#include "gl43_raytracer.h"

#include "core/math/math_funcs.h"
#include "core/object.h"
#include "drivers/gles3/gl43_effect_helpers.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>

#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_STATIC_DRAW
#define GL_STATIC_DRAW 0x88E4
#endif
#ifndef GL_DYNAMIC_DRAW
#define GL_DYNAMIC_DRAW 0x88E8
#endif

static const char *GL43_RAYTRACER_COMMON_GLSL = R"GLSL(
struct BVHNode {
	vec4 aabbMin;
	vec4 aabbMax;
};

struct RTTriangle {
	vec4 p0;
	vec4 p1;
	vec4 p2;
	vec4 n0;
	vec4 n1;
	vec4 n2;
	vec4 uv01;
	vec4 uv2pad;
};

struct RTInstance {
	mat4 invModel;
	mat4 model;
	vec4 aabbMin;
	vec4 aabbMax;
	ivec4 flags;
};

layout (std430, binding = 0) readonly buffer BLASNodeBuf { BVHNode blasNodes[]; };
layout (std430, binding = 1) readonly buffer TriangleBuf { RTTriangle triangles[]; };
layout (std430, binding = 2) readonly buffer TLASNodeBuf { BVHNode tlasNodes[]; };
layout (std430, binding = 3) readonly buffer TLASIdxBuf  { int tlasIndices[]; };
layout (std430, binding = 4) readonly buffer InstanceBuf { RTInstance instances[]; };

vec3 safeInvDir(vec3 d) {
	return vec3(
		1.0 / (abs(d.x) < 1e-8 ? (d.x < 0.0 ? -1e-8 : 1e-8) : d.x),
		1.0 / (abs(d.y) < 1e-8 ? (d.y < 0.0 ? -1e-8 : 1e-8) : d.y),
		1.0 / (abs(d.z) < 1e-8 ? (d.z < 0.0 ? -1e-8 : 1e-8) : d.z)
	);
}

float intersectAABB(vec3 bMin, vec3 bMax, vec3 ro, vec3 invRd, float maxT) {
	vec3 t0 = (bMin - ro) * invRd;
	vec3 t1 = (bMax - ro) * invRd;
	vec3 tMinV = min(t0, t1);
	vec3 tMaxV = max(t0, t1);
	float tNear = max(max(tMinV.x, tMinV.y), tMinV.z);
	float tFar  = min(min(tMaxV.x, tMaxV.y), tMaxV.z);
	return (tFar >= max(tNear, 0.0) && tNear < maxT) ? max(tNear, 0.0) : 1e30;
}

bool rayTriangleGeom(RTTriangle tri, vec3 ro, vec3 rd, float maxT, out float outDet, out float outT, out float outU, out float outV) {
	vec3 e1 = tri.p1.xyz - tri.p0.xyz;
	vec3 e2 = tri.p2.xyz - tri.p0.xyz;
	vec3 pvec = cross(rd, e2);
	float det = dot(e1, pvec);
	if (abs(det) < 1e-7) return false;
	float invDet = 1.0 / det;
	vec3 tvec = ro - tri.p0.xyz;
	float u = dot(tvec, pvec) * invDet;
	if (u < 0.0 || u > 1.0) return false;
	vec3 qvec = cross(tvec, e1);
	float v = dot(rd, qvec) * invDet;
	if (v < 0.0 || u + v > 1.0) return false;
	float t = dot(e2, qvec) * invDet;
	if (t <= 0.001 || t >= maxT) return false;
	outDet = det;
	outT = t;
	outU = u;
	outV = v;
	return true;
}

bool rayTriangleGeomIdx(int triIdx, vec3 ro, vec3 rd, float maxT, out float outDet, out float outT, out float outU, out float outV) {
	vec3 p0 = triangles[triIdx].p0.xyz;
	vec3 e1 = triangles[triIdx].p1.xyz - p0;
	vec3 e2 = triangles[triIdx].p2.xyz - p0;
	vec3 pvec = cross(rd, e2);
	float det = dot(e1, pvec);
	if (abs(det) < 1e-7) return false;
	float invDet = 1.0 / det;
	vec3 tvec = ro - p0;
	float u = dot(tvec, pvec) * invDet;
	if (u < 0.0 || u > 1.0) return false;
	vec3 qvec = cross(tvec, e1);
	float v = dot(rd, qvec) * invDet;
	if (v < 0.0 || u + v > 1.0) return false;
	float t = dot(e2, qvec) * invDet;
	if (t <= 0.001 || t >= maxT) return false;
	outDet = det;
	outT = t;
	outU = u;
	outV = v;
	return true;
}
)GLSL";

const char *GL43Raytracer::get_glsl_common() {
	return GL43_RAYTRACER_COMMON_GLSL;
}

uint64_t GL43Raytracer::SimplificationConfig::get_signature() const {
	uint64_t h = 0x14650FB0739D0383ULL;
	h = GL43EffectHelpers::hash_combine_u64(h, enabled ? 1ULL : 0ULL);
	if (!enabled) {
		return h;
	}
	h = GL43EffectHelpers::hash_combine_u64(h, (uint64_t)mode);
	h = GL43EffectHelpers::hash_float_u64(h, ratio);
	h = GL43EffectHelpers::hash_combine_u64(h, (uint64_t)min_triangles);
	h = GL43EffectHelpers::hash_float_u64(h, max_error);
	h = GL43EffectHelpers::hash_combine_u64(h, preserve_seams ? 1ULL : 0ULL);
	h = GL43EffectHelpers::hash_combine_u64(h, simplify_skinned ? 1ULL : 0ULL);
	h = GL43EffectHelpers::hash_combine_u64(h, simplify_gbuffer ? 1ULL : 0ULL);
	return h;
}

bool GL43Raytracer::init_gl() {
	if (ssbo_blas_nodes != 0) {
		return true;
	}
	GLuint bufs[5] = {};
	glGenBuffers(5, bufs);
	ssbo_blas_nodes = (uint32_t)bufs[0];
	ssbo_triangles = (uint32_t)bufs[1];
	ssbo_tlas_nodes = (uint32_t)bufs[2];
	ssbo_tlas_indices = (uint32_t)bufs[3];
	ssbo_instances = (uint32_t)bufs[4];
	blas_buffers_dirty = true;
	return true;
}

void GL43Raytracer::cleanup_gl() {
	for (auto &kv : blas_cache) {
		if (kv.second.gl_vao) {
			GLuint v = (GLuint)kv.second.gl_vao;
			glDeleteVertexArrays(1, &v);
		}
		if (kv.second.gl_vbo) {
			GLuint b = (GLuint)kv.second.gl_vbo;
			glDeleteBuffers(1, &b);
		}
	}
	blas_cache.clear();
	blas_order.clear();

	for (auto &kv : skinned_gl_cache) {
		if (kv.second.gl_vao) {
			GLuint v = (GLuint)kv.second.gl_vao;
			glDeleteVertexArrays(1, &v);
		}
		if (kv.second.gl_vbo) {
			GLuint b = (GLuint)kv.second.gl_vbo;
			glDeleteBuffers(1, &b);
		}
	}
	skinned_gl_cache.clear();

	for (size_t i = 0; i < gl_buffers_to_delete.size(); i++) {
		if (gl_buffers_to_delete[i].vao) {
			GLuint v = (GLuint)gl_buffers_to_delete[i].vao;
			glDeleteVertexArrays(1, &v);
		}
		if (gl_buffers_to_delete[i].vbo) {
			GLuint b = (GLuint)gl_buffers_to_delete[i].vbo;
			glDeleteBuffers(1, &b);
		}
	}
	gl_buffers_to_delete.clear();

	uint32_t *ssbos[5] = { &ssbo_blas_nodes, &ssbo_triangles, &ssbo_tlas_nodes, &ssbo_tlas_indices, &ssbo_instances };
	for (int i = 0; i < 5; i++) {
		if (*ssbos[i]) {
			GLuint b = (GLuint)(*ssbos[i]);
			glDeleteBuffers(1, &b);
			*ssbos[i] = 0;
		}
	}

	cached_static_nodes.clear();
	cached_static_tris.clear();
	blas_buffers_dirty = true;
	last_frame_had_skinned = false;
}

void GL43Raytracer::invalidate_blas_cache() {
	for (auto &kv : blas_cache) {
		kv.second.simpl_signature = ~kv.second.simpl_signature;
	}
	blas_buffers_dirty = true;
}

void GL43Raytracer::_simplify_mesh_data(
		const SimplificationConfig &p_cfg,
		std::vector<Vector3> &io_pos,
		std::vector<Vector3> &io_norm,
		std::vector<Vector2> &io_uv,
		std::vector<int32_t> &io_bones,
		std::vector<float> &io_weights,
		bool p_has_skin,
		int p_bpv,
		std::vector<uint32_t> &io_tri_verts,
		std::vector<int32_t> &io_tri_surf) const {

	int num_tris = (int)io_tri_surf.size();
	int num_verts = (int)io_pos.size();
	if (!p_cfg.enabled || num_tris <= p_cfg.min_triangles || p_cfg.ratio >= 0.999f || num_verts < 4) {
		return;
	}
	if (p_has_skin && !p_cfg.simplify_skinned) {
		return;
	}

	int target_tris = MAX(4, MAX(p_cfg.min_triangles / 2, (int)std::round((double)num_tris * p_cfg.ratio)));
	if (target_tris >= num_tris) {
		return;
	}

	AABB bounds(io_pos[0], Vector3());
	for (int i = 1; i < num_verts; i++) {
		bounds.expand_to(io_pos[i]);
	}
	float diag = (float)bounds.size.length();
	if (diag < 1e-6f) {
		return;
	}

	if (p_cfg.mode == SIMPLIFY_VERTEX_CLUSTERING) {
		struct ClusterKey {
			int32_t ix, iy, iz, surf, uq, vq;
			bool operator==(const ClusterKey &o) const {
				return ix == o.ix && iy == o.iy && iz == o.iz && surf == o.surf && uq == o.uq && vq == o.vq;
			}
		};
		struct ClusterHash {
			size_t operator()(const ClusterKey &k) const {
				uint64_t h = (uint64_t)(uint32_t)k.ix * 73856093ULL;
				h ^= (uint64_t)(uint32_t)k.iy * 19349663ULL;
				h ^= (uint64_t)(uint32_t)k.iz * 83492791ULL;
				h ^= (uint64_t)(uint32_t)k.surf * 2654435761ULL;
				h ^= (uint64_t)(uint32_t)k.uq * 2246822519ULL;
				h ^= (uint64_t)(uint32_t)k.vq * 3266489917ULL;
				return (size_t)h;
			}
		};

		int grid_n = CLAMP((int)std::round(std::sqrt((double)target_tris * 0.45)), 4, 512);
		float max_axis = MAX(MAX((float)bounds.size.x, (float)bounds.size.y), (float)bounds.size.z);
		float cell_size = MAX(max_axis / (float)grid_n, 1e-5f);
		float inv_cell = 1.0f / cell_size;

		std::vector<int32_t> vert_surf(num_verts, 0);
		for (int t = 0; t < num_tris; t++) {
			int32_t s = io_tri_surf[t];
			vert_surf[io_tri_verts[t * 3 + 0]] = s;
			vert_surf[io_tri_verts[t * 3 + 1]] = s;
			vert_surf[io_tri_verts[t * 3 + 2]] = s;
		}

		struct ClusterAccum {
			Vector3 pos_sum;
			Vector3 norm_sum;
			Vector2 uv_sum;
			int count = 0;
			int first_vert = -1;
		};

		std::unordered_map<ClusterKey, int32_t, ClusterHash> cluster_map;
		cluster_map.reserve((size_t)num_verts);
		std::vector<ClusterAccum> clusters;
		std::vector<int32_t> vert_to_cluster(num_verts, -1);

		for (int v = 0; v < num_verts; v++) {
			Vector3 rel = io_pos[v] - bounds.position;
			ClusterKey key;
			key.ix = (int32_t)std::floor(rel.x * inv_cell);
			key.iy = (int32_t)std::floor(rel.y * inv_cell);
			key.iz = (int32_t)std::floor(rel.z * inv_cell);
			key.surf = vert_surf[v];
			if (p_cfg.preserve_seams) {
				key.uq = (int32_t)std::floor(io_uv[v].x * 4.0f);
				key.vq = (int32_t)std::floor(io_uv[v].y * 4.0f);
			} else {
				key.uq = 0;
				key.vq = 0;
			}

			auto it = cluster_map.find(key);
			int32_t cid = -1;
			if (it == cluster_map.end()) {
				cid = (int32_t)clusters.size();
				cluster_map[key] = cid;
				ClusterAccum acc;
				acc.first_vert = v;
				clusters.push_back(acc);
			} else {
				cid = it->second;
			}

			ClusterAccum &c = clusters[cid];
			c.pos_sum += io_pos[v];
			c.norm_sum += io_norm[v];
			c.uv_sum += io_uv[v];
			c.count++;
			vert_to_cluster[v] = cid;
		}

		std::vector<uint32_t> new_tri_verts;
		std::vector<int32_t> new_tri_surf;
		new_tri_verts.reserve(io_tri_verts.size());
		new_tri_surf.reserve(io_tri_surf.size());

		for (int t = 0; t < num_tris; t++) {
			int32_t c0 = vert_to_cluster[io_tri_verts[t * 3 + 0]];
			int32_t c1 = vert_to_cluster[io_tri_verts[t * 3 + 1]];
			int32_t c2 = vert_to_cluster[io_tri_verts[t * 3 + 2]];
			if (c0 == c1 || c1 == c2 || c0 == c2 || c0 < 0 || c1 < 0 || c2 < 0) {
				continue;
			}
			new_tri_verts.push_back((uint32_t)c0);
			new_tri_verts.push_back((uint32_t)c1);
			new_tri_verts.push_back((uint32_t)c2);
			new_tri_surf.push_back(io_tri_surf[t]);
		}

		if (new_tri_surf.empty() || (int)new_tri_surf.size() >= num_tris) {
			return;
		}

		int nc = (int)clusters.size();
		std::vector<Vector3> new_pos(nc);
		std::vector<Vector3> new_norm(nc);
		std::vector<Vector2> new_uv(nc);
		std::vector<int32_t> new_bones;
		std::vector<float> new_weights;
		if (p_has_skin) {
			new_bones.resize((size_t)nc * p_bpv);
			new_weights.resize((size_t)nc * p_bpv);
		}

		for (int i = 0; i < nc; i++) {
			const ClusterAccum &c = clusters[i];
			float inv_n = 1.0f / (float)c.count;
			new_pos[i] = c.pos_sum * inv_n;
			Vector3 n = c.norm_sum;
			float nlen = n.length();
			new_norm[i] = (nlen > 1e-6f) ? (n / nlen) : Vector3(0, 1, 0);
			new_uv[i] = c.uv_sum * inv_n;
			if (p_has_skin && c.first_vert >= 0) {
				for (int k = 0; k < p_bpv; k++) {
					new_bones[i * p_bpv + k] = io_bones[c.first_vert * p_bpv + k];
					new_weights[i * p_bpv + k] = io_weights[c.first_vert * p_bpv + k];
				}
			}
		}

		io_pos.swap(new_pos);
		io_norm.swap(new_norm);
		io_uv.swap(new_uv);
		if (p_has_skin) {
			io_bones.swap(new_bones);
			io_weights.swap(new_weights);
		}
		io_tri_verts.swap(new_tri_verts);
		io_tri_surf.swap(new_tri_surf);
		return;
	}

	// QEM Edge Collapse (Garland & Heckbert Quadric Error Metric)
	struct PosKey {
		int64_t x, y, z;
		bool operator==(const PosKey &o) const { return x == o.x && y == o.y && z == o.z; }
	};
	struct PosHash {
		size_t operator()(const PosKey &k) const {
			uint64_t h = (uint64_t)k.x * 73856093ULL ^ (uint64_t)k.y * 19349663ULL ^ (uint64_t)k.z * 83492791ULL;
			return (size_t)h;
		}
	};

	float weld_scale = 10000.0f / MAX(diag, 1e-4f);
	std::unordered_map<PosKey, int32_t, PosHash> pos_map;
	pos_map.reserve((size_t)num_verts);
	std::vector<Vector3> pos_coords;
	std::vector<int32_t> vert_to_pos(num_verts);
	std::vector<std::vector<int32_t>> pos_attrs;

	for (int v = 0; v < num_verts; v++) {
		Vector3 rel = io_pos[v] - bounds.position;
		PosKey k = {
			(int64_t)std::llround((double)rel.x * weld_scale),
			(int64_t)std::llround((double)rel.y * weld_scale),
			(int64_t)std::llround((double)rel.z * weld_scale)
		};
		auto it = pos_map.find(k);
		int32_t pid;
		if (it == pos_map.end()) {
			pid = (int32_t)pos_coords.size();
			pos_map[k] = pid;
			pos_coords.push_back(io_pos[v]);
			pos_attrs.emplace_back();
		} else {
			pid = it->second;
		}
		vert_to_pos[v] = pid;
		pos_attrs[pid].push_back(v);
	}

	int num_pos = (int)pos_coords.size();
	if (num_pos < 4) {
		return;
	}

	struct SymMat {
		double m[10] = {};
		double weight = 0.0;
		void add_plane(double a, double b, double c, double d, double w, bool add_to_weight) {
			m[0] += a * a * w; m[1] += a * b * w; m[2] += a * c * w; m[3] += a * d * w;
			m[4] += b * b * w; m[5] += b * c * w; m[6] += b * d * w;
			m[7] += c * c * w; m[8] += c * d * w;
			m[9] += d * d * w;
			if (add_to_weight) weight += w;
		}
		void operator+=(const SymMat &o) {
			for (int i = 0; i < 10; i++) m[i] += o.m[i];
			weight += o.weight;
		}
		double eval(const Vector3 &v) const {
			double x = v.x, y = v.y, z = v.z;
			double e = m[0] * x * x + 2.0 * m[1] * x * y + 2.0 * m[2] * x * z + 2.0 * m[3] * x
					 + m[4] * y * y + 2.0 * m[5] * y * z + 2.0 * m[6] * y
					 + m[7] * z * z + 2.0 * m[8] * z + m[9];
			return e > 0.0 ? e : 0.0;
		}
	};

	struct SimpTri {
		uint32_t v[3];
		int32_t p[3];
		int32_t surf;
		bool alive;
	};

	std::vector<SimpTri> tris(num_tris);
	std::vector<SymMat> quadrics(num_pos);
	std::vector<std::vector<int32_t>> pos_tris(num_pos);
	std::vector<int32_t> vert_surf(num_verts, -1);
	std::vector<uint8_t> pos_seam(num_pos, 0);

	auto make_edge_key = [](int32_t a, int32_t b) -> uint64_t {
		if (a > b) std::swap(a, b);
		return ((uint64_t)(uint32_t)a << 32) | (uint64_t)(uint32_t)b;
	};
	std::unordered_map<uint64_t, int32_t> edge_tri_count;
	edge_tri_count.reserve((size_t)num_tris * 2);

	for (int t = 0; t < num_tris; t++) {
		SimpTri &st = tris[t];
		st.surf = io_tri_surf[t];
		st.alive = true;
		for (int k = 0; k < 3; k++) {
			st.v[k] = io_tri_verts[t * 3 + k];
			st.p[k] = vert_to_pos[st.v[k]];
			vert_surf[st.v[k]] = st.surf;
		}
		if (st.p[0] == st.p[1] || st.p[1] == st.p[2] || st.p[0] == st.p[2]) {
			st.alive = false;
			continue;
		}

		for (int k = 0; k < 3; k++) {
			pos_tris[st.p[k]].push_back(t);
			edge_tri_count[make_edge_key(st.p[k], st.p[(k + 1) % 3])]++;
		}

		const Vector3 &p0 = pos_coords[st.p[0]];
		const Vector3 &p1 = pos_coords[st.p[1]];
		const Vector3 &p2 = pos_coords[st.p[2]];
		Vector3 cross_n = (p1 - p0).cross(p2 - p0);
		double area2 = cross_n.length();
		if (area2 > 1e-12) {
			Vector3 n = cross_n / (float)area2;
			double d = -(double)n.dot(p0);
			double area = 0.5 * area2;
			for (int k = 0; k < 3; k++) {
				quadrics[st.p[k]].add_plane(n.x, n.y, n.z, d, area, true);
			}
		}
	}

	for (int p = 0; p < num_pos; p++) {
		const auto &attrs = pos_attrs[p];
		if (attrs.size() > 1) {
			int ref_v = attrs[0];
			for (size_t i = 1; i < attrs.size(); i++) {
				int av = attrs[i];
				if (vert_surf[av] != vert_surf[ref_v]
					|| io_norm[av].dot(io_norm[ref_v]) < 0.85f
					|| (io_uv[av] - io_uv[ref_v]).length_squared() > 1e-4f) {
					pos_seam[p] = 1;
					break;
				}
			}
		}
	}

	for (int t = 0; t < num_tris; t++) {
		if (!tris[t].alive) continue;
		const Vector3 &p0 = pos_coords[tris[t].p[0]];
		const Vector3 &p1 = pos_coords[tris[t].p[1]];
		const Vector3 &p2 = pos_coords[tris[t].p[2]];
		Vector3 tri_n = (p1 - p0).cross(p2 - p0);
		float len = tri_n.length();
		if (len < 1e-8f) continue;
		tri_n /= len;

		for (int k = 0; k < 3; k++) {
			int32_t pa = tris[t].p[k];
			int32_t pb = tris[t].p[(k + 1) % 3];
			if (edge_tri_count[make_edge_key(pa, pb)] == 1) {
				pos_seam[pa] = 1;
				pos_seam[pb] = 1;
				Vector3 ed = pos_coords[pb] - pos_coords[pa];
				float elen = ed.length();
				if (elen > 1e-6f) {
					Vector3 bn = ed.cross(tri_n).normalized();
					double bd = -(double)bn.dot(pos_coords[pa]);
					double bw = (double)elen * (double)elen * (p_cfg.preserve_seams ? 16.0 : 4.0);
					quadrics[pa].add_plane(bn.x, bn.y, bn.z, bd, bw, false);
					quadrics[pb].add_plane(bn.x, bn.y, bn.z, bd, bw, false);
				}
			}
		}
	}

	struct EdgeCandidate {
		double cost;
		float rms_norm;
		int32_t pa, pb;
		uint32_t ver_a, ver_b;
		uint8_t choice;
		Vector3 pos;
		bool operator>(const EdgeCandidate &o) const { return cost > o.cost; }
	};

	std::vector<int32_t> redirect(num_pos);
	std::vector<uint32_t> pos_ver(num_pos, 0);
	for (int i = 0; i < num_pos; i++) redirect[i] = i;

	auto eval_edge = [&](int32_t a, int32_t b, EdgeCandidate &out) -> bool {
		if (a == b) return false;
		SymMat q = quadrics[a];
		q += quadrics[b];

		Vector3 cands[3] = {
			pos_coords[a],
			pos_coords[b],
			(pos_coords[a] + pos_coords[b]) * 0.5f
		};

		int best_c = 2;
		double best_e = 1e30;
		bool seam_a = pos_seam[a] != 0;
		bool seam_b = pos_seam[b] != 0;

		for (int c = 0; c < 3; c++) {
			if (p_cfg.preserve_seams) {
				if (seam_a && !seam_b && c != 0) continue;
				if (seam_b && !seam_a && c != 1) continue;
			}
			double e = q.eval(cands[c]);
			if (e < best_e) {
				best_e = e;
				best_c = c;
			}
		}

		double wsum = MAX(q.weight, 1e-10);
		float rms_norm = (float)(std::sqrt(best_e / wsum)) / diag;
		double penalty = (p_cfg.preserve_seams && seam_a && seam_b) ? 4.0 : 1.0;

		out.cost = best_e * penalty;
		out.rms_norm = rms_norm;
		out.pa = a;
		out.pb = b;
		out.ver_a = pos_ver[a];
		out.ver_b = pos_ver[b];
		out.choice = (uint8_t)best_c;
		out.pos = cands[best_c];
		return true;
	};

	std::priority_queue<EdgeCandidate, std::vector<EdgeCandidate>, std::greater<EdgeCandidate>> heap;
	for (const auto &kv : edge_tri_count) {
		int32_t pa = (int32_t)(uint32_t)(kv.first >> 32);
		int32_t pb = (int32_t)(uint32_t)(kv.first & 0xFFFFFFFFULL);
		EdgeCandidate ec;
		if (eval_edge(pa, pb, ec)) {
			heap.push(ec);
		}
	}

	int live_tris = 0;
	for (int t = 0; t < num_tris; t++) {
		if (tris[t].alive) live_tris++;
	}

	std::vector<uint32_t> tri_seen_stamp(num_tris, 0);
	std::vector<int32_t> pos_seen_stamp(num_pos, 0);
	uint32_t cur_stamp = 1;
	std::vector<int32_t> incident_tris;
	std::vector<int32_t> neighbor_pos;

	while (!heap.empty() && live_tris > target_tris) {
		EdgeCandidate ec = heap.top();
		heap.pop();

		int32_t pa = ec.pa;
		int32_t pb = ec.pb;
		if (redirect[pa] != pa || redirect[pb] != pb) continue;
		if (pos_ver[pa] != ec.ver_a || pos_ver[pb] != ec.ver_b) continue;
		if (p_cfg.max_error > 0.0f && ec.rms_norm > p_cfg.max_error) {
			break;
		}

		cur_stamp++;
		incident_tris.clear();
		for (int pass = 0; pass < 2; pass++) {
			int32_t p = (pass == 0) ? pa : pb;
			for (int32_t tid : pos_tris[p]) {
				if (tris[tid].alive && tri_seen_stamp[tid] != cur_stamp) {
					tri_seen_stamp[tid] = cur_stamp;
					incident_tris.push_back(tid);
				}
			}
		}

		bool valid_collapse = true;
		for (int32_t tid : incident_tris) {
			const SimpTri &st = tris[tid];
			bool has_a = (st.p[0] == pa || st.p[1] == pa || st.p[2] == pa);
			bool has_b = (st.p[0] == pb || st.p[1] == pb || st.p[2] == pb);
			if (has_a && has_b) continue;

			Vector3 old_p[3] = { pos_coords[st.p[0]], pos_coords[st.p[1]], pos_coords[st.p[2]] };
			Vector3 new_p[3] = {
				(st.p[0] == pa || st.p[0] == pb) ? ec.pos : old_p[0],
				(st.p[1] == pa || st.p[1] == pb) ? ec.pos : old_p[1],
				(st.p[2] == pa || st.p[2] == pb) ? ec.pos : old_p[2]
			};
			Vector3 old_n = (old_p[1] - old_p[0]).cross(old_p[2] - old_p[0]);
			Vector3 new_n = (new_p[1] - new_p[0]).cross(new_p[2] - new_p[0]);
			float olen = old_n.length();
			float nlen = new_n.length();
			if (nlen < 1e-7f) {
				valid_collapse = false;
				break;
			}
			if (olen > 1e-7f && (old_n / olen).dot(new_n / nlen) < 0.25f) {
				valid_collapse = false;
				break;
			}
		}
		if (!valid_collapse) continue;

		pos_coords[pa] = ec.pos;
		quadrics[pa] += quadrics[pb];
		pos_seam[pa] |= pos_seam[pb];
		redirect[pb] = pa;
		pos_ver[pa]++;
		pos_ver[pb]++;

		if (ec.choice != 0) {
			for (int32_t va : pos_attrs[pa]) {
				io_pos[va] = ec.pos;
			}
		}

		std::vector<int32_t> merged_tris;
		merged_tris.reserve(incident_tris.size());
		neighbor_pos.clear();

		for (int32_t tid : incident_tris) {
			SimpTri &st = tris[tid];
			bool has_a = (st.p[0] == pa || st.p[1] == pa || st.p[2] == pa);
			bool has_b = (st.p[0] == pb || st.p[1] == pb || st.p[2] == pb);
			if (has_a && has_b) {
				st.alive = false;
				live_tris--;
				continue;
			}

			if (has_b) {
				for (int k = 0; k < 3; k++) {
					if (st.p[k] == pb) {
						st.p[k] = pa;
						uint32_t vb = st.v[k];

						int32_t best_va = -1;
						float best_uv_d2 = 0.04f;
						for (int32_t va : pos_attrs[pa]) {
							if (vert_surf[va] != st.surf) continue;
							if (io_norm[va].dot(io_norm[vb]) < 0.7f) continue;
							float d2 = (io_uv[va] - io_uv[vb]).length_squared();
							if (d2 < best_uv_d2) {
								best_uv_d2 = d2;
								best_va = va;
							}
						}

						if (best_va >= 0) {
							if (ec.choice == 2) {
								io_norm[best_va] = (io_norm[best_va] + io_norm[vb]).normalized();
								io_uv[best_va] = (io_uv[best_va] + io_uv[vb]) * 0.5f;
							} else if (ec.choice == 1) {
								io_norm[best_va] = io_norm[vb];
								io_uv[best_va] = io_uv[vb];
							}
							st.v[k] = (uint32_t)best_va;
						} else {
							io_pos[vb] = ec.pos;
							pos_attrs[pa].push_back((int32_t)vb);
						}
					}
				}
			}

			merged_tris.push_back(tid);
			for (int k = 0; k < 3; k++) {
				int32_t np = st.p[k];
				if (np != pa && pos_seen_stamp[np] != (int32_t)cur_stamp) {
					pos_seen_stamp[np] = (int32_t)cur_stamp;
					neighbor_pos.push_back(np);
				}
			}
		}

		pos_tris[pa].swap(merged_tris);
		std::vector<int32_t>().swap(pos_tris[pb]);
		std::vector<int32_t>().swap(pos_attrs[pb]);

		for (int32_t np : neighbor_pos) {
			EdgeCandidate nec;
			if (eval_edge(pa, np, nec)) {
				heap.push(nec);
			}
		}
	}

	if (live_tris >= num_tris || live_tris <= 0) {
		return;
	}

	std::vector<int32_t> vert_remap(num_verts, -1);
	std::vector<Vector3> out_pos;
	std::vector<Vector3> out_norm;
	std::vector<Vector2> out_uv;
	std::vector<int32_t> out_bones;
	std::vector<float> out_weights;
	std::vector<uint32_t> out_tri_verts;
	std::vector<int32_t> out_tri_surf;

	out_tri_verts.reserve((size_t)live_tris * 3);
	out_tri_surf.reserve((size_t)live_tris);

	for (int t = 0; t < num_tris; t++) {
		if (!tris[t].alive) continue;
		uint32_t nv[3];
		for (int k = 0; k < 3; k++) {
			uint32_t old_v = tris[t].v[k];
			if (vert_remap[old_v] < 0) {
				int32_t idx = (int32_t)out_pos.size();
				vert_remap[old_v] = idx;
				out_pos.push_back(io_pos[old_v]);
				out_norm.push_back(io_norm[old_v]);
				out_uv.push_back(io_uv[old_v]);
				if (p_has_skin) {
					for (int b = 0; b < p_bpv; b++) {
						out_bones.push_back(io_bones[old_v * p_bpv + b]);
						out_weights.push_back(io_weights[old_v * p_bpv + b]);
					}
				}
			}
			nv[k] = (uint32_t)vert_remap[old_v];
		}
		if (nv[0] == nv[1] || nv[1] == nv[2] || nv[0] == nv[2]) continue;
		out_tri_verts.push_back(nv[0]);
		out_tri_verts.push_back(nv[1]);
		out_tri_verts.push_back(nv[2]);
		out_tri_surf.push_back(tris[t].surf);
	}

	if (out_tri_surf.empty()) {
		return;
	}

	io_pos.swap(out_pos);
	io_norm.swap(out_norm);
	io_uv.swap(out_uv);
	if (p_has_skin) {
		io_bones.swap(out_bones);
		io_weights.swap(out_weights);
	}
	io_tri_verts.swap(out_tri_verts);
	io_tri_surf.swap(out_tri_surf);
}

void GL43Raytracer::rebuild_blas_cache(
		const std::vector<MeshInstance *> &p_meshes,
		const std::vector<Ref<Mesh>> &p_extra_meshes,
		const SimplificationConfig &p_simpl,
		int p_blas_builds_per_frame) {

	uint64_t simpl_sig = p_simpl.get_signature();
	int builds_done = 0;

	for (auto it = blas_cache.begin(); it != blas_cache.end(); ) {
		if (it->second.cached_mesh_instance_id != 0 && ObjectDB::get_instance((ObjectID)it->second.cached_mesh_instance_id) == nullptr) {
			if (it->second.gl_vao || it->second.gl_vbo) {
				gl_buffers_to_delete.push_back({ it->second.gl_vao, it->second.gl_vbo });
			}
			blas_order.erase(std::remove(blas_order.begin(), blas_order.end(), it->first), blas_order.end());
			it = blas_cache.erase(it);
			blas_buffers_dirty = true;
		} else {
			++it;
		}
	}

	auto process_mesh = [&](const Ref<Mesh> &mesh) {
		if (mesh.is_null() || !mesh->get_rid().is_valid()) {
			return;
		}
		uint32_t mid = mesh->get_rid().get_id();
		uint64_t m_inst_id = (uint64_t)mesh->get_instance_id();
		AABB mesh_aabb = mesh->get_aabb();

		auto existing = blas_cache.find(mid);
		if (existing != blas_cache.end()) {
			if (existing->second.cached_mesh_instance_id == m_inst_id &&
					GL43EffectHelpers::aabb_same(existing->second.cached_mesh_aabb, mesh_aabb) &&
					existing->second.simpl_signature == simpl_sig) {
				return;
			}
			if (p_blas_builds_per_frame > 0 && builds_done >= p_blas_builds_per_frame) {
				return;
			}
			if (existing->second.gl_vao || existing->second.gl_vbo) {
				gl_buffers_to_delete.push_back({ existing->second.gl_vao, existing->second.gl_vbo });
			}
			blas_cache.erase(existing);
			blas_order.erase(std::remove(blas_order.begin(), blas_order.end(), mid), blas_order.end());
		} else {
			if (p_blas_builds_per_frame > 0 && builds_done >= p_blas_builds_per_frame) {
				return;
			}
		}

		builds_done++;

		MeshBLAS blas;
		blas.mesh_id = mid;
		blas.cached_mesh_instance_id = m_inst_id;
		blas.cached_mesh_aabb = mesh_aabb;
		blas.simpl_signature = simpl_sig;

		std::vector<Vector3> orig_pos;
		std::vector<Vector3> orig_norm;
		std::vector<Vector2> orig_uv;
		std::vector<int32_t> orig_bones;
		std::vector<float> orig_weights;
		std::vector<uint32_t> orig_tri_verts;
		std::vector<int32_t> orig_tri_surf;

		int surf_count = mesh->get_surface_count();
		for (int s = 0; s < surf_count; s++) {
			if (mesh->surface_get_primitive_type(s) != Mesh::PRIMITIVE_TRIANGLES) {
				continue;
			}
			Array arrays = mesh->surface_get_arrays(s);
			if (arrays.size() <= Mesh::ARRAY_VERTEX) {
				continue;
			}

			PoolVector<Vector3> verts = arrays[Mesh::ARRAY_VERTEX];
			int vcount = verts.size();
			if (vcount < 3) {
				continue;
			}

			PoolVector<Vector3> norms;
			bool has_norms = arrays[Mesh::ARRAY_NORMAL].get_type() == Variant::POOL_VECTOR3_ARRAY;
			if (has_norms) norms = arrays[Mesh::ARRAY_NORMAL];
			if (norms.size() != vcount) has_norms = false;

			PoolVector<Vector2> uvs;
			bool has_uvs = arrays.size() > Mesh::ARRAY_TEX_UV && arrays[Mesh::ARRAY_TEX_UV].get_type() == Variant::POOL_VECTOR2_ARRAY;
			if (has_uvs) uvs = arrays[Mesh::ARRAY_TEX_UV];
			if (uvs.size() != vcount) has_uvs = false;

			PoolVector<int> bones;
			PoolVector<real_t> weights;
			bool has_skin = arrays.size() > Mesh::ARRAY_WEIGHTS &&
							arrays[Mesh::ARRAY_BONES].get_type() == Variant::POOL_INT_ARRAY &&
							arrays[Mesh::ARRAY_WEIGHTS].get_type() == Variant::POOL_REAL_ARRAY;
			int bpv = 4;
			if (has_skin) {
				bones = arrays[Mesh::ARRAY_BONES];
				weights = arrays[Mesh::ARRAY_WEIGHTS];
				if (bones.size() >= vcount * 8 && weights.size() >= vcount * 8) bpv = 8;
				else if (bones.size() < vcount * 4 || weights.size() < vcount * 4) has_skin = false;
			}
			if (has_skin) {
				blas.has_skin_weights = true;
				blas.bones_per_vert = MAX(blas.bones_per_vert, bpv);
			}

			PoolVector<int> indices;
			bool has_indices = arrays[Mesh::ARRAY_INDEX].get_type() == Variant::POOL_INT_ARRAY;
			if (has_indices) indices = arrays[Mesh::ARRAY_INDEX];

			PoolVector<Vector3>::Read vr = verts.read();
			PoolVector<Vector3>::Read nr = norms.read();
			PoolVector<Vector2>::Read uvr = uvs.read();
			PoolVector<int>::Read br = bones.read();
			PoolVector<real_t>::Read wr = weights.read();
			PoolVector<int>::Read ir = indices.read();

			uint32_t v_base = (uint32_t)orig_pos.size();
			for (int v = 0; v < vcount; v++) {
				orig_pos.push_back(vr[v]);
				orig_norm.push_back(has_norms ? nr[v] : Vector3(0, 1, 0));
				orig_uv.push_back(has_uvs ? uvr[v] : Vector2());
				if (has_skin) {
					for (int k = 0; k < bpv; k++) {
						orig_bones.push_back(br[v * bpv + k]);
						orig_weights.push_back((float)wr[v * bpv + k]);
					}
				}
			}

			int tri_verts = has_indices ? indices.size() : vcount;
			for (int i = 0; i + 2 < tri_verts; i += 3) {
				int i0 = has_indices ? ir[i + 0] : (i + 0);
				int i1 = has_indices ? ir[i + 1] : (i + 1);
				int i2 = has_indices ? ir[i + 2] : (i + 2);
				if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= vcount || i1 >= vcount || i2 >= vcount) {
					continue;
				}

				Vector3 p0 = vr[i0], p1 = vr[i1], p2 = vr[i2];
				Vector3 fn = (p1 - p0).cross(p2 - p0);
				if (fn.length_squared() < 1e-12f) continue;
				fn.normalize();

				if (!has_norms) {
					orig_norm[v_base + i0] = fn;
					orig_norm[v_base + i1] = fn;
					orig_norm[v_base + i2] = fn;
				}

				orig_tri_verts.push_back(v_base + (uint32_t)i0);
				orig_tri_verts.push_back(v_base + (uint32_t)i1);
				orig_tri_verts.push_back(v_base + (uint32_t)i2);
				orig_tri_surf.push_back(s);
			}
		}

		if (orig_tri_surf.empty()) {
			return;
		}

		if (blas.has_skin_weights && orig_bones.size() < orig_pos.size() * (size_t)blas.bones_per_vert) {
			orig_bones.resize(orig_pos.size() * (size_t)blas.bones_per_vert, 0);
			orig_weights.resize(orig_pos.size() * (size_t)blas.bones_per_vert, 0.0f);
		}

		bool use_orig_gbuf = !p_simpl.simplify_gbuffer && p_simpl.enabled;
		if (use_orig_gbuf && !blas.has_skin_weights) {
			int orig_tris = (int)orig_tri_surf.size();
			blas.gbuf_verts.reserve((size_t)orig_tris * 27);
			for (int t = 0; t < orig_tris; t++) {
				float sf = (float)orig_tri_surf[t];
				for (int k = 0; k < 3; k++) {
					uint32_t vi = orig_tri_verts[t * 3 + k];
					blas.gbuf_verts.push_back((float)orig_pos[vi].x);
					blas.gbuf_verts.push_back((float)orig_pos[vi].y);
					blas.gbuf_verts.push_back((float)orig_pos[vi].z);
					blas.gbuf_verts.push_back((float)orig_norm[vi].x);
					blas.gbuf_verts.push_back((float)orig_norm[vi].y);
					blas.gbuf_verts.push_back((float)orig_norm[vi].z);
					blas.gbuf_verts.push_back((float)orig_uv[vi].x);
					blas.gbuf_verts.push_back((float)orig_uv[vi].y);
					blas.gbuf_verts.push_back(sf);
				}
			}
			blas.vertex_count = orig_tris * 3;
		}

		_simplify_mesh_data(p_simpl, orig_pos, orig_norm, orig_uv, orig_bones, orig_weights, blas.has_skin_weights, blas.bones_per_vert, orig_tri_verts, orig_tri_surf);

		int final_tris = (int)orig_tri_surf.size();
		std::vector<Triangle> raw_tris(final_tris);
		std::vector<AABB> tri_aabbs(final_tris);
		bool first_aabb = true;

		bool emit_gbuf_from_simplified = (!use_orig_gbuf || blas.has_skin_weights);
		if (emit_gbuf_from_simplified) {
			blas.gbuf_verts.reserve((size_t)final_tris * 27);
		}

		for (int t = 0; t < final_tris; t++) {
			uint32_t i0 = orig_tri_verts[t * 3 + 0];
			uint32_t i1 = orig_tri_verts[t * 3 + 1];
			uint32_t i2 = orig_tri_verts[t * 3 + 2];

			const Vector3 &p0 = orig_pos[i0], &p1 = orig_pos[i1], &p2 = orig_pos[i2];
			const Vector3 &n0 = orig_norm[i0], &n1 = orig_norm[i1], &n2 = orig_norm[i2];
			const Vector2 &uv0 = orig_uv[i0], &uv1 = orig_uv[i1], &uv2 = orig_uv[i2];
			int32_t s = orig_tri_surf[t];

			Triangle &tri = raw_tris[t];
			tri = {};
			tri.p0[0] = (float)p0.x; tri.p0[1] = (float)p0.y; tri.p0[2] = (float)p0.z; tri.surface_idx = s;
			tri.p1[0] = (float)p1.x; tri.p1[1] = (float)p1.y; tri.p1[2] = (float)p1.z;
			tri.p2[0] = (float)p2.x; tri.p2[1] = (float)p2.y; tri.p2[2] = (float)p2.z;
			tri.n0[0] = (float)n0.x; tri.n0[1] = (float)n0.y; tri.n0[2] = (float)n0.z;
			tri.n1[0] = (float)n1.x; tri.n1[1] = (float)n1.y; tri.n1[2] = (float)n1.z;
			tri.n2[0] = (float)n2.x; tri.n2[1] = (float)n2.y; tri.n2[2] = (float)n2.z;
			tri.uv01[0] = (float)uv0.x; tri.uv01[1] = (float)uv0.y;
			tri.uv01[2] = (float)uv1.x; tri.uv01[3] = (float)uv1.y;
			tri.uv2_pad[0] = (float)uv2.x; tri.uv2_pad[1] = (float)uv2.y;

			AABB tb(p0, Vector3());
			tb.expand_to(p1);
			tb.expand_to(p2);
			tb.grow_by(0.001f);
			tri_aabbs[t] = tb;

			if (first_aabb) {
				blas.local_aabb = tb;
				first_aabb = false;
			} else {
				blas.local_aabb.merge_with(tb);
			}

			if (emit_gbuf_from_simplified) {
				float sf = (float)s;
				const Vector3 *pts[3] = { &p0, &p1, &p2 };
				const Vector3 *nms[3] = { &n0, &n1, &n2 };
				const Vector2 *uvs[3] = { &uv0, &uv1, &uv2 };
				for (int k = 0; k < 3; k++) {
					blas.gbuf_verts.push_back((float)pts[k]->x);
					blas.gbuf_verts.push_back((float)pts[k]->y);
					blas.gbuf_verts.push_back((float)pts[k]->z);
					blas.gbuf_verts.push_back((float)nms[k]->x);
					blas.gbuf_verts.push_back((float)nms[k]->y);
					blas.gbuf_verts.push_back((float)nms[k]->z);
					blas.gbuf_verts.push_back((float)uvs[k]->x);
					blas.gbuf_verts.push_back((float)uvs[k]->y);
					blas.gbuf_verts.push_back(sf);
				}
			}
		}

		if (emit_gbuf_from_simplified) {
			blas.vertex_count = final_tris * 3;
		}

		blas.bind_pos = std::move(orig_pos);
		blas.bind_norm = std::move(orig_norm);
		blas.bind_uv = std::move(orig_uv);
		if (blas.has_skin_weights) {
			blas.bind_bones = std::move(orig_bones);
			blas.bind_weights = std::move(orig_weights);
		}

		std::vector<int32_t> tri_order;
		GL43BVHHelper::build_bvh(tri_aabbs, tri_order, blas.nodes);

		blas.triangles.resize(raw_tris.size());
		blas.ordered_tri_verts.resize(raw_tris.size() * 3);
		blas.ordered_tri_surf.resize(raw_tris.size());
		for (size_t i = 0; i < tri_order.size(); i++) {
			int idx = tri_order[i];
			blas.triangles[i] = raw_tris[idx];
			blas.ordered_tri_verts[i * 3 + 0] = orig_tri_verts[idx * 3 + 0];
			blas.ordered_tri_verts[i * 3 + 1] = orig_tri_verts[idx * 3 + 1];
			blas.ordered_tri_verts[i * 3 + 2] = orig_tri_verts[idx * 3 + 2];
			blas.ordered_tri_surf[i] = orig_tri_surf[idx];
		}

		blas_order.push_back(mid);
		blas_cache[mid] = std::move(blas);
		blas_buffers_dirty = true;
	};

	for (MeshInstance *mi : p_meshes) {
		process_mesh(mi->get_mesh());
	}
	for (const Ref<Mesh> &m : p_extra_meshes) {
		process_mesh(m);
	}
}

const GL43Raytracer::MeshBLAS *GL43Raytracer::get_blas(uint32_t p_mesh_id) const {
	auto it = blas_cache.find(p_mesh_id);
	return (it != blas_cache.end()) ? &it->second : nullptr;
}

bool GL43Raytracer::_build_skinned_snapshot(MeshInstance *p_mi, const MeshBLAS &p_blas, SkinnedSnapshot &r_snap, AABB &r_local_aabb) const {
	if (!p_blas.has_skin_weights) {
		return false;
	}

	std::vector<Vector3> posed_pos;
	std::vector<Vector3> posed_norm;
	if (!GL43SkinningHelper::pose_mesh_positions_and_normals(p_mi, p_blas.bind_pos, p_blas.bind_norm, p_blas.bind_bones, p_blas.bind_weights, p_blas.bones_per_vert, posed_pos, posed_norm)) {
		return false;
	}

	int num_tris = (int)p_blas.triangles.size();
	r_snap.obj_id = (uint64_t)p_mi->get_instance_id();
	r_snap.posed_tris.resize(num_tris);
	r_snap.posed_gbuf_verts.resize((size_t)num_tris * 27);
	r_snap.vertex_count = num_tris * 3;
	r_snap.refit_nodes = p_blas.nodes;

	for (int t = 0; t < num_tris; t++) {
		uint32_t i0 = p_blas.ordered_tri_verts[t * 3 + 0];
		uint32_t i1 = p_blas.ordered_tri_verts[t * 3 + 1];
		uint32_t i2 = p_blas.ordered_tri_verts[t * 3 + 2];

		const Vector3 &p0 = posed_pos[i0], &p1 = posed_pos[i1], &p2 = posed_pos[i2];
		const Vector3 &n0 = posed_norm[i0], &n1 = posed_norm[i1], &n2 = posed_norm[i2];
		const Vector2 &uv0 = p_blas.bind_uv[i0], &uv1 = p_blas.bind_uv[i1], &uv2 = p_blas.bind_uv[i2];
		int32_t s = p_blas.ordered_tri_surf[t];

		Triangle &tri = r_snap.posed_tris[t];
		tri.p0[0] = (float)p0.x; tri.p0[1] = (float)p0.y; tri.p0[2] = (float)p0.z; tri.surface_idx = s;
		tri.p1[0] = (float)p1.x; tri.p1[1] = (float)p1.y; tri.p1[2] = (float)p1.z;
		tri.p2[0] = (float)p2.x; tri.p2[1] = (float)p2.y; tri.p2[2] = (float)p2.z;
		tri.n0[0] = (float)n0.x; tri.n0[1] = (float)n0.y; tri.n0[2] = (float)n0.z;
		tri.n1[0] = (float)n1.x; tri.n1[1] = (float)n1.y; tri.n1[2] = (float)n1.z;
		tri.n2[0] = (float)n2.x; tri.n2[1] = (float)n2.y; tri.n2[2] = (float)n2.z;
		tri.uv01[0] = (float)uv0.x; tri.uv01[1] = (float)uv0.y;
		tri.uv01[2] = (float)uv1.x; tri.uv01[3] = (float)uv1.y;
		tri.uv2_pad[0] = (float)uv2.x; tri.uv2_pad[1] = (float)uv2.y;

		float sf = (float)s;
		float *gv = &r_snap.posed_gbuf_verts[(size_t)t * 27];
		const Vector3 *pts[3] = { &p0, &p1, &p2 };
		const Vector3 *nms[3] = { &n0, &n1, &n2 };
		const Vector2 *uvs[3] = { &uv0, &uv1, &uv2 };
		for (int k = 0; k < 3; k++) {
			gv[k * 9 + 0] = (float)pts[k]->x; gv[k * 9 + 1] = (float)pts[k]->y; gv[k * 9 + 2] = (float)pts[k]->z;
			gv[k * 9 + 3] = (float)nms[k]->x; gv[k * 9 + 4] = (float)nms[k]->y; gv[k * 9 + 5] = (float)nms[k]->z;
			gv[k * 9 + 6] = (float)uvs[k]->x; gv[k * 9 + 7] = (float)uvs[k]->y; gv[k * 9 + 8] = sf;
		}
	}

	GL43BVHHelper::refit_bvh_bottom_up(r_snap.refit_nodes, r_snap.posed_tris.data(), sizeof(Triangle));

	const BVHNode &root = r_snap.refit_nodes[0];
	r_local_aabb = AABB(
			Vector3(root.min_x, root.min_y, root.min_z),
			Vector3(root.max_x - root.min_x, root.max_y - root.min_y, root.max_z - root.min_z));
	return true;
}

bool GL43Raytracer::passes_distance_cull(
		const AABB &p_world_aabb,
		const Vector3 &p_cam_pos,
		float p_cull_distance,
		float p_cull_min_angle,
		float p_res_scale) {

	if (p_cull_min_angle <= 0.0f) {
		return true;
	}
	Vector3 center = p_world_aabb.position + p_world_aabb.size * 0.5f;
	float radius = (float)p_world_aabb.size.length() * 0.5f;
	float dist = MAX(0.0f, (float)center.distance_to(p_cam_pos) - radius);
	float min_ang = p_cull_min_angle / MAX(p_res_scale, 0.25f);
	if (dist > p_cull_distance && (radius / MAX(dist, 0.1f)) < min_ang) {
		return false;
	}
	return true;
}

void GL43Raytracer::begin_frame_instances() {
	instances.clear();
	skinned_instances.clear();
	tlas_candidates.clear();
	tlas_nodes.clear();
	tlas_indices.clear();
}

bool GL43Raytracer::add_mesh_instance(
		MeshInstance *p_mi,
		int32_t p_mat_offset,
		const Vector3 &p_cam_pos,
		float p_cull_distance,
		float p_cull_min_angle,
		float p_res_scale,
		bool p_enable_skinning,
		bool p_add_to_tlas,
		bool p_draw_in_gbuf,
		uint32_t p_user_flags,
		int32_t *r_out_index) {

	if (!p_mi || p_mi->get_mesh().is_null() || !p_mi->get_mesh()->get_rid().is_valid()) {
		return false;
	}
	uint32_t mid = p_mi->get_mesh()->get_rid().get_id();
	auto it = blas_cache.find(mid);
	if (it == blas_cache.end()) {
		return false;
	}
	const MeshBLAS &blas = it->second;

	GeometryInstance::ShadowCastingSetting shadow_mode = p_mi->get_cast_shadows_setting();
	bool shadows_only = (shadow_mode == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
	bool cast_shadows = (shadow_mode != GeometryInstance::SHADOW_CASTING_SETTING_OFF);
	bool double_sided_shadow = (shadow_mode == GeometryInstance::SHADOW_CASTING_SETTING_DOUBLE_SIDED);

	InstanceEntry inst;
	inst.mesh_id = mid;
	inst.world_xform = p_mi->get_global_transform();
	inst.inv_xform = inst.world_xform.affine_inverse();
	GL43EffectHelpers::transform_to_float16(inst.world_xform, inst.model_mat);
	inst.mat_offset = p_mat_offset;
	inst.layer_mask = p_mi->get_layer_mask();
	inst.draw_in_gbuf = p_draw_in_gbuf && !shadows_only;
	inst.visible_in_trace = p_add_to_tlas && !shadows_only;
	inst.cast_shadows = cast_shadows;
	inst.double_sided_shadow = double_sided_shadow;
	inst.flip_facing = (inst.world_xform.basis.determinant() < 0.0f);
	inst.user_flags = p_user_flags;

	AABB local_bounds = blas.local_aabb;
	SkinnedSnapshot sk_snap;
	if (p_enable_skinning && _build_skinned_snapshot(p_mi, blas, sk_snap, local_bounds)) {
		inst.is_skinned = true;
		inst.skinned_snap_idx = (int32_t)skinned_instances.size();
		skinned_instances.push_back(std::move(sk_snap));
	}

	inst.world_aabb = inst.world_xform.xform(local_bounds);
	inst.world_aabb.grow_by(0.002f);

	if (!passes_distance_cull(inst.world_aabb, p_cam_pos, p_cull_distance, p_cull_min_angle, p_res_scale)) {
		if (inst.is_skinned) {
			skinned_instances.pop_back();
		}
		return false;
	}

	int32_t snap_idx = (int32_t)instances.size();
	instances.push_back(std::move(inst));
	if (r_out_index) {
		*r_out_index = snap_idx;
	}

	if (p_add_to_tlas) {
		Vector3 center = instances[snap_idx].world_aabb.position + instances[snap_idx].world_aabb.size * 0.5f;
		float rad = (float)instances[snap_idx].world_aabb.size.length() * 0.5f;
		float dist = MAX(0.0f, (float)center.distance_to(p_cam_pos) - rad);
		tlas_candidates.push_back({ snap_idx, dist, instances[snap_idx].world_aabb });
	}
	return true;
}

bool GL43Raytracer::add_multimesh_instance(
		MultiMeshInstance *p_mmi,
		int32_t p_shared_mat_offset,
		const Vector3 &p_cam_pos,
		float p_cull_distance,
		float p_cull_min_angle,
		float p_res_scale,
		bool p_add_to_tlas,
		bool p_draw_in_gbuf,
		uint32_t p_user_flags) {

	if (!p_mmi || p_mmi->get_multimesh().is_null()) {
		return false;
	}
	Ref<MultiMesh> mm = p_mmi->get_multimesh();
	if (mm->get_transform_format() != MultiMesh::TRANSFORM_3D || mm->get_mesh().is_null() || !mm->get_mesh()->get_rid().is_valid()) {
		return false;
	}
	uint32_t mid = mm->get_mesh()->get_rid().get_id();
	auto it = blas_cache.find(mid);
	if (it == blas_cache.end()) {
		return false;
	}
	const MeshBLAS &blas = it->second;

	GeometryInstance::ShadowCastingSetting shadow_mode = p_mmi->get_cast_shadows_setting();
	bool shadows_only = (shadow_mode == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
	bool cast_shadows = (shadow_mode != GeometryInstance::SHADOW_CASTING_SETTING_OFF);
	bool double_sided_shadow = (shadow_mode == GeometryInstance::SHADOW_CASTING_SETTING_DOUBLE_SIDED);
	uint32_t lm = p_mmi->get_layer_mask();

	Transform base_xf = p_mmi->get_global_transform();
	int count = mm->get_visible_instance_count();
	if (count < 0) {
		count = mm->get_instance_count();
	}

	bool any_added = false;
	for (int i = 0; i < count; i++) {
		Transform xf = base_xf * mm->get_instance_transform(i);
		AABB world_aabb = xf.xform(blas.local_aabb);
		world_aabb.grow_by(0.002f);

		if (!passes_distance_cull(world_aabb, p_cam_pos, p_cull_distance, p_cull_min_angle, p_res_scale)) {
			continue;
		}

		InstanceEntry inst;
		inst.mesh_id = mid;
		inst.world_xform = xf;
		inst.inv_xform = xf.affine_inverse();
		GL43EffectHelpers::transform_to_float16(xf, inst.model_mat);
		inst.world_aabb = world_aabb;
		inst.mat_offset = p_shared_mat_offset;
		inst.layer_mask = lm;
		inst.draw_in_gbuf = p_draw_in_gbuf && !shadows_only;
		inst.visible_in_trace = p_add_to_tlas && !shadows_only;
		inst.cast_shadows = cast_shadows;
		inst.double_sided_shadow = double_sided_shadow;
		inst.flip_facing = (xf.basis.determinant() < 0.0f);
		inst.user_flags = p_user_flags;

		int32_t snap_idx = (int32_t)instances.size();
		instances.push_back(std::move(inst));
		any_added = true;

		if (p_add_to_tlas) {
			Vector3 center = world_aabb.position + world_aabb.size * 0.5f;
			float rad = (float)world_aabb.size.length() * 0.5f;
			float dist = MAX(0.0f, (float)center.distance_to(p_cam_pos) - rad);
			tlas_candidates.push_back({ snap_idx, dist, world_aabb });
		}
	}
	return any_added;
}

bool GL43Raytracer::add_custom_mesh_instance(
		const Ref<Mesh> &p_mesh,
		const Transform &p_world_xf,
		uint32_t p_layer_mask,
		GeometryInstance::ShadowCastingSetting p_shadow_mode,
		int32_t p_mat_offset,
		const Vector3 &p_cam_pos,
		float p_cull_distance,
		float p_cull_min_angle,
		float p_res_scale,
		bool p_add_to_tlas,
		bool p_draw_in_gbuf,
		uint32_t p_user_flags,
		int32_t *r_out_index) {

	if (p_mesh.is_null() || !p_mesh->get_rid().is_valid()) {
		return false;
	}
	uint32_t mid = p_mesh->get_rid().get_id();
	auto it = blas_cache.find(mid);
	if (it == blas_cache.end()) {
		return false;
	}
	const MeshBLAS &blas = it->second;

	AABB world_aabb = p_world_xf.xform(blas.local_aabb);
	world_aabb.grow_by(0.002f);

	if (!passes_distance_cull(world_aabb, p_cam_pos, p_cull_distance, p_cull_min_angle, p_res_scale)) {
		return false;
	}

	bool shadows_only = (p_shadow_mode == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
	bool cast_shadows = (p_shadow_mode != GeometryInstance::SHADOW_CASTING_SETTING_OFF);
	bool double_sided_shadow = (p_shadow_mode == GeometryInstance::SHADOW_CASTING_SETTING_DOUBLE_SIDED);

	InstanceEntry inst;
	inst.mesh_id = mid;
	inst.world_xform = p_world_xf;
	inst.inv_xform = p_world_xf.affine_inverse();
	GL43EffectHelpers::transform_to_float16(p_world_xf, inst.model_mat);
	inst.world_aabb = world_aabb;
	inst.mat_offset = p_mat_offset;
	inst.layer_mask = p_layer_mask;
	inst.draw_in_gbuf = p_draw_in_gbuf && !shadows_only;
	inst.visible_in_trace = p_add_to_tlas && !shadows_only;
	inst.cast_shadows = cast_shadows;
	inst.double_sided_shadow = double_sided_shadow;
	inst.flip_facing = (p_world_xf.basis.determinant() < 0.0f);
	inst.user_flags = p_user_flags;

	int32_t snap_idx = (int32_t)instances.size();
	instances.push_back(std::move(inst));
	if (r_out_index) {
		*r_out_index = snap_idx;
	}

	if (p_add_to_tlas) {
		Vector3 center = world_aabb.position + world_aabb.size * 0.5f;
		float rad = (float)world_aabb.size.length() * 0.5f;
		float dist = MAX(0.0f, (float)center.distance_to(p_cam_pos) - rad);
		tlas_candidates.push_back({ snap_idx, dist, world_aabb });
	}
	return true;
}

void GL43Raytracer::build_tlas(int p_max_tlas_instances) {
	if (p_max_tlas_instances > 0 && (int)tlas_candidates.size() > p_max_tlas_instances) {
		std::nth_element(
				tlas_candidates.begin(),
				tlas_candidates.begin() + p_max_tlas_instances,
				tlas_candidates.end(),
				[](const TlasCandidate &a, const TlasCandidate &b) { return a.sort_dist < b.sort_dist; });
		tlas_candidates.resize(p_max_tlas_instances);
	}

	std::vector<AABB> bvh_aabbs;
	std::vector<int32_t> bvh_to_snap;
	bvh_aabbs.reserve(tlas_candidates.size());
	bvh_to_snap.reserve(tlas_candidates.size());
	for (const TlasCandidate &tc : tlas_candidates) {
		bvh_aabbs.push_back(tc.world_aabb);
		bvh_to_snap.push_back(tc.snap_idx);
	}

	GL43BVHHelper::build_bvh(bvh_aabbs, tlas_indices, tlas_nodes);
	for (int32_t &idx : tlas_indices) {
		idx = bvh_to_snap[idx];
	}
}

void GL43Raytracer::sync_gpu_buffers() {
	if (ssbo_blas_nodes == 0) {
		init_gl();
	}

	for (size_t i = 0; i < gl_buffers_to_delete.size(); i++) {
		if (gl_buffers_to_delete[i].vao) {
			GLuint v = (GLuint)gl_buffers_to_delete[i].vao;
			glDeleteVertexArrays(1, &v);
		}
		if (gl_buffers_to_delete[i].vbo) {
			GLuint b = (GLuint)gl_buffers_to_delete[i].vbo;
			glDeleteBuffers(1, &b);
		}
	}
	gl_buffers_to_delete.clear();

	bool static_dirty = blas_buffers_dirty;
	if (static_dirty) {
		cached_static_nodes.clear();
		cached_static_tris.clear();

		for (uint32_t mid : blas_order) {
			MeshBLAS &b = blas_cache[mid];
			b.blas_node_offset = (int32_t)cached_static_nodes.size();
			int32_t tri_offset = (int32_t)cached_static_tris.size();

			for (BVHNode n : b.nodes) {
				if (n.count > 0) n.left_or_first += tri_offset;
				else n.left_or_first += b.blas_node_offset;
				cached_static_nodes.push_back(n);
			}
			cached_static_tris.insert(cached_static_tris.end(), b.triangles.begin(), b.triangles.end());

			if (!b.gpu_uploaded && !b.gbuf_verts.empty()) {
				GLuint vao = 0, vbo = 0;
				glGenVertexArrays(1, &vao);
				glGenBuffers(1, &vbo);
				b.gl_vao = (uint32_t)vao;
				b.gl_vbo = (uint32_t)vbo;
				glBindVertexArray(vao);
				glBindBuffer(GL_ARRAY_BUFFER, vbo);
				glBufferData(GL_ARRAY_BUFFER, b.gbuf_verts.size() * sizeof(float), b.gbuf_verts.data(), GL_STATIC_DRAW);
				glEnableVertexAttribArray(0);
				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)0);
				glEnableVertexAttribArray(1);
				glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(3 * sizeof(float)));
				glEnableVertexAttribArray(2);
				glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(6 * sizeof(float)));
				glEnableVertexAttribArray(3);
				glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(8 * sizeof(float)));
				b.gpu_uploaded = true;
			}
		}

		if (skinned_instances.empty()) {
			GL43EffectHelpers::upload_ssbo(ssbo_blas_nodes, cached_static_nodes.data(), cached_static_nodes.size() * sizeof(BVHNode), sizeof(BVHNode), GL_STATIC_DRAW);
			GL43EffectHelpers::upload_ssbo(ssbo_triangles, cached_static_tris.data(), cached_static_tris.size() * sizeof(Triangle), sizeof(Triangle), GL_STATIC_DRAW);
			last_frame_had_skinned = false;
		}
		blas_buffers_dirty = false;
	}

	frame_counter++;
	std::vector<int32_t> skinned_blas_roots(skinned_instances.size(), 0);

	if (!skinned_instances.empty()) {
		std::vector<BVHNode> frame_nodes = cached_static_nodes;
		std::vector<Triangle> frame_tris = cached_static_tris;

		for (size_t s = 0; s < skinned_instances.size(); s++) {
			const SkinnedSnapshot &sk = skinned_instances[s];
			int32_t node_off = (int32_t)frame_nodes.size();
			int32_t tri_off = (int32_t)frame_tris.size();
			skinned_blas_roots[s] = node_off;

			for (BVHNode n : sk.refit_nodes) {
				if (n.count > 0) n.left_or_first += tri_off;
				else n.left_or_first += node_off;
				frame_nodes.push_back(n);
			}
			frame_tris.insert(frame_tris.end(), sk.posed_tris.begin(), sk.posed_tris.end());

			SkinnedMeshGL &sgl = skinned_gl_cache[sk.obj_id];
			if (sgl.gl_vao == 0) {
				GLuint vao = 0, vbo = 0;
				glGenVertexArrays(1, &vao);
				glGenBuffers(1, &vbo);
				sgl.gl_vao = (uint32_t)vao;
				sgl.gl_vbo = (uint32_t)vbo;
				glBindVertexArray(vao);
				glBindBuffer(GL_ARRAY_BUFFER, vbo);
				glEnableVertexAttribArray(0);
				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)0);
				glEnableVertexAttribArray(1);
				glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(3 * sizeof(float)));
				glEnableVertexAttribArray(2);
				glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(6 * sizeof(float)));
				glEnableVertexAttribArray(3);
				glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void *)(8 * sizeof(float)));
			} else {
				glBindBuffer(GL_ARRAY_BUFFER, (GLuint)sgl.gl_vbo);
			}
			glBufferData(GL_ARRAY_BUFFER, sk.posed_gbuf_verts.size() * sizeof(float), sk.posed_gbuf_verts.data(), GL_DYNAMIC_DRAW);
			sgl.vertex_count = sk.vertex_count;
			sgl.last_frame_seen = frame_counter;
		}

		GL43EffectHelpers::upload_ssbo(ssbo_blas_nodes, frame_nodes.data(), frame_nodes.size() * sizeof(BVHNode), sizeof(BVHNode), GL_DYNAMIC_DRAW);
		GL43EffectHelpers::upload_ssbo(ssbo_triangles, frame_tris.data(), frame_tris.size() * sizeof(Triangle), sizeof(Triangle), GL_DYNAMIC_DRAW);
		last_frame_had_skinned = true;
	} else if (!static_dirty && last_frame_had_skinned) {
		GL43EffectHelpers::upload_ssbo(ssbo_blas_nodes, cached_static_nodes.data(), cached_static_nodes.size() * sizeof(BVHNode), sizeof(BVHNode), GL_STATIC_DRAW);
		GL43EffectHelpers::upload_ssbo(ssbo_triangles, cached_static_tris.data(), cached_static_tris.size() * sizeof(Triangle), sizeof(Triangle), GL_STATIC_DRAW);
		last_frame_had_skinned = false;
	}

	for (auto it = skinned_gl_cache.begin(); it != skinned_gl_cache.end(); ) {
		if (frame_counter - it->second.last_frame_seen > 60) {
			if (it->second.gl_vao) {
				GLuint v = (GLuint)it->second.gl_vao;
				glDeleteVertexArrays(1, &v);
			}
			if (it->second.gl_vbo) {
				GLuint b = (GLuint)it->second.gl_vbo;
				glDeleteBuffers(1, &b);
			}
			it = skinned_gl_cache.erase(it);
		} else {
			++it;
		}
	}

	std::vector<Instance> gpu_insts(instances.size());
	for (size_t i = 0; i < instances.size(); i++) {
		const InstanceEntry &src = instances[i];
		Instance &dst = gpu_insts[i];
		GL43EffectHelpers::transform_to_float16(src.inv_xform, dst.inv_model);
		std::memcpy(dst.model, src.model_mat, sizeof(dst.model));
		dst.aabb_min[0] = (float)src.world_aabb.position.x;
		dst.aabb_min[1] = (float)src.world_aabb.position.y;
		dst.aabb_min[2] = (float)src.world_aabb.position.z;
		dst.blas_root = src.is_skinned ? skinned_blas_roots[src.skinned_snap_idx] : blas_cache[src.mesh_id].blas_node_offset;
		Vector3 wmax = src.world_aabb.position + src.world_aabb.size;
		dst.aabb_max[0] = (float)wmax.x;
		dst.aabb_max[1] = (float)wmax.y;
		dst.aabb_max[2] = (float)wmax.z;
		dst.mat_offset = src.mat_offset;

		int32_t iflags = 0;
		if (src.visible_in_trace) iflags |= 1;
		if (src.cast_shadows) iflags |= 2;
		if (src.double_sided_shadow) iflags |= 4;
		dst.flags[0] = iflags;
		dst.flags[1] = 0;
		dst.flags[2] = 0;
		dst.flags[3] = 0;
	}

	bool valid_tlas = has_tlas();
	GL43EffectHelpers::upload_ssbo(ssbo_tlas_nodes, valid_tlas ? tlas_nodes.data() : nullptr, valid_tlas ? tlas_nodes.size() * sizeof(BVHNode) : 0, sizeof(BVHNode), GL_DYNAMIC_DRAW);
	GL43EffectHelpers::upload_ssbo(ssbo_tlas_indices, valid_tlas ? tlas_indices.data() : nullptr, valid_tlas ? tlas_indices.size() * sizeof(int32_t) : 0, sizeof(int32_t), GL_DYNAMIC_DRAW);
	GL43EffectHelpers::upload_ssbo(ssbo_instances, gpu_insts.data(), gpu_insts.size() * sizeof(Instance), sizeof(Instance), GL_DYNAMIC_DRAW);
}

void GL43Raytracer::bind_ssbos(
		uint32_t p_blas_node_binding,
		uint32_t p_tri_binding,
		uint32_t p_tlas_node_binding,
		uint32_t p_tlas_idx_binding,
		uint32_t p_inst_binding) const {

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_blas_node_binding, (GLuint)ssbo_blas_nodes);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_tri_binding, (GLuint)ssbo_triangles);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_tlas_node_binding, (GLuint)ssbo_tlas_nodes);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_tlas_idx_binding, (GLuint)ssbo_tlas_indices);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_inst_binding, (GLuint)ssbo_instances);
}

void GL43Raytracer::unbind_ssbos(
		uint32_t p_blas_node_binding,
		uint32_t p_tri_binding,
		uint32_t p_tlas_node_binding,
		uint32_t p_tlas_idx_binding,
		uint32_t p_inst_binding) const {

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_blas_node_binding, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_tri_binding, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_tlas_node_binding, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_tlas_idx_binding, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, p_inst_binding, 0);
}

bool GL43Raytracer::get_instance_draw_data(size_t p_index, uint32_t &r_vao, int32_t &r_vertex_count) const {
	if (p_index >= instances.size()) {
		return false;
	}
	const InstanceEntry &inst = instances[p_index];
	if (inst.is_skinned) {
		const SkinnedSnapshot &sk = skinned_instances[inst.skinned_snap_idx];
		auto sit = skinned_gl_cache.find(sk.obj_id);
		if (sit == skinned_gl_cache.end() || sit->second.gl_vao == 0 || sit->second.vertex_count <= 0) {
			return false;
		}
		r_vao = sit->second.gl_vao;
		r_vertex_count = sit->second.vertex_count;
		return true;
	}

	auto bit = blas_cache.find(inst.mesh_id);
	if (bit == blas_cache.end() || !bit->second.gpu_uploaded || bit->second.gl_vao == 0 || bit->second.vertex_count <= 0) {
		return false;
	}
	r_vao = bit->second.gl_vao;
	r_vertex_count = bit->second.vertex_count;
	return true;
}
