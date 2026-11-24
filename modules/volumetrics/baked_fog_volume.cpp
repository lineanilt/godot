#include "baked_fog_volume.h"

#include "core/math/math_funcs.h"
#include "local_fog_volume.h"
#include "modules/modules_enabled.gen.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/resources/mesh.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

struct BakeTri {
	Vector3 p0, p1, p2;
	uint32_t layer_mask = 0xFFFFFFFF;
};

struct BakeBVHNode {
	AABB bounds;
	int left_or_first = 0;
	int count = 0;
};

static bool ray_aabb_hit(const AABB &p_box, const Vector3 &p_ro, const Vector3 &p_inv_rd, float p_max_t) {
	Vector3 bmin = p_box.position;
	Vector3 bmax = p_box.position + p_box.size;
	float tmin = 0.0f;
	float tmax = p_max_t;
	for (int i = 0; i < 3; i++) {
		float t1 = (bmin[i] - p_ro[i]) * p_inv_rd[i];
		float t2 = (bmax[i] - p_ro[i]) * p_inv_rd[i];
		if (t1 > t2) {
			float tmp = t1;
			t1 = t2;
			t2 = tmp;
		}
		tmin = MAX(tmin, t1);
		tmax = MIN(tmax, t2);
		if (tmin > tmax) {
			return false;
		}
	}
	return tmax >= 0.0f && tmin < p_max_t;
}

static bool ray_tri_hit(const BakeTri &p_tri, const Vector3 &p_ro, const Vector3 &p_rd, float p_max_t) {
	Vector3 e1 = p_tri.p1 - p_tri.p0;
	Vector3 e2 = p_tri.p2 - p_tri.p0;
	Vector3 pvec = p_rd.cross(e2);
	float det = e1.dot(pvec);
	if (Math::abs(det) < 1e-7f) {
		return false;
	}
	float inv_det = 1.0f / det;
	Vector3 tvec = p_ro - p_tri.p0;
	float u = tvec.dot(pvec) * inv_det;
	if (u < 0.0f || u > 1.0f) {
		return false;
	}
	Vector3 qvec = tvec.cross(e1);
	float v = p_rd.dot(qvec) * inv_det;
	if (v < 0.0f || u + v > 1.0f) {
		return false;
	}
	float t = e2.dot(qvec) * inv_det;
	return t > 0.001f && t < p_max_t;
}

class BakeRaycaster {
	std::vector<BakeTri> tris;
	std::vector<BakeBVHNode> nodes;

	int build_recursive(std::vector<AABB> &p_aabbs, std::vector<int> &p_indices, int p_first, int p_count, int p_depth) {
		int node_idx = (int)nodes.size();
		nodes.push_back(BakeBVHNode());

		AABB bounds = p_aabbs[p_indices[p_first]];
		AABB cent_bounds(bounds.position + bounds.size * 0.5f, Vector3());
		for (int i = 1; i < p_count; i++) {
			const AABB &b = p_aabbs[p_indices[p_first + i]];
			bounds.merge_with(b);
			cent_bounds.expand_to(b.position + b.size * 0.5f);
		}
		nodes[node_idx].bounds = bounds;

		int axis = cent_bounds.get_longest_axis_index();
		if (p_count <= 4 || p_depth >= 18 || cent_bounds.size[axis] < 1e-5f) {
			nodes[node_idx].left_or_first = p_first;
			nodes[node_idx].count = p_count;
			return node_idx;
		}

		float split = cent_bounds.position[axis] + cent_bounds.size[axis] * 0.5f;
		auto it = std::partition(p_indices.begin() + p_first, p_indices.begin() + p_first + p_count, [&](int idx) {
			return (p_aabbs[idx].position[axis] + p_aabbs[idx].size[axis] * 0.5f) < split;
		});
		int mid = (int)(it - p_indices.begin());
		if (mid == p_first || mid == p_first + p_count) {
			mid = p_first + p_count / 2;
		}

		build_recursive(p_aabbs, p_indices, p_first, mid - p_first, p_depth + 1);
		int right_idx = build_recursive(p_aabbs, p_indices, mid, p_first + p_count - mid, p_depth + 1);
		nodes[node_idx].left_or_first = right_idx;
		nodes[node_idx].count = 0;
		return node_idx;
	}

static void append_mesh(const Ref<Mesh> &p_mesh, const Transform &p_xf, uint32_t p_layer_mask, std::vector<BakeTri> &r_raw_tris, std::vector<AABB> &r_aabbs) {
		if (p_mesh.is_null()) {
			return;
		}
		int sc = p_mesh->get_surface_count();
		for (int s = 0; s < sc; s++) {
			if (p_mesh->surface_get_primitive_type(s) != Mesh::PRIMITIVE_TRIANGLES) {
				continue;
			}
			Array arrays = p_mesh->surface_get_arrays(s);
			if (arrays.size() <= Mesh::ARRAY_VERTEX) {
				continue;
			}
			PoolVector<Vector3> verts = arrays[Mesh::ARRAY_VERTEX];
			int vcount = verts.size();
			if (vcount < 3) {
				continue;
			}
			PoolVector<int> indices;
			bool has_indices = arrays.size() > Mesh::ARRAY_INDEX && arrays[Mesh::ARRAY_INDEX].get_type() == Variant::POOL_INT_ARRAY;
			if (has_indices) {
				indices = arrays[Mesh::ARRAY_INDEX];
			}
			PoolVector<Vector3>::Read vr = verts.read();
			PoolVector<int>::Read ir = indices.read();
			int count = has_indices ? indices.size() : vcount;

			for (int i = 0; i + 2 < count; i += 3) {
				int i0 = has_indices ? ir[i + 0] : (i + 0);
				int i1 = has_indices ? ir[i + 1] : (i + 1);
				int i2 = has_indices ? ir[i + 2] : (i + 2);
				if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= vcount || i1 >= vcount || i2 >= vcount) {
					continue;
				}
				BakeTri t;
				t.p0 = p_xf.xform(vr[i0]);
				t.p1 = p_xf.xform(vr[i1]);
				t.p2 = p_xf.xform(vr[i2]);
				t.layer_mask = p_layer_mask;
				if ((t.p1 - t.p0).cross(t.p2 - t.p0).length_squared() < 1e-12f) {
					continue;
				}
				AABB tb(t.p0, Vector3());
				tb.expand_to(t.p1);
				tb.expand_to(t.p2);
				tb.grow_by(0.002f);
				r_raw_tris.push_back(t);
				r_aabbs.push_back(tb);
			}
		}
	}

public:
	void build(const std::vector<MeshInstance *> &p_meshes,
#ifdef MODULE_CSG_ENABLED
			const std::vector<CSGShape *> &p_csgs,
#endif
			uint32_t p_shadow_mask) {
		std::vector<BakeTri> raw_tris;
		std::vector<AABB> aabbs;

		for (size_t m = 0; m < p_meshes.size(); m++) {
			MeshInstance *mi = p_meshes[m];
			if (!mi->is_visible_in_tree() || !(mi->get_layer_mask() & p_shadow_mask)) {
				continue;
			}
			if (mi->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_OFF) {
				continue;
			}
			append_mesh(mi->get_mesh(), mi->get_global_transform(), mi->get_layer_mask(), raw_tris, aabbs);
		}

#ifdef MODULE_CSG_ENABLED
		for (size_t m = 0; m < p_csgs.size(); m++) {
			CSGShape *csg = p_csgs[m];
			if (!csg->is_visible_in_tree() || !csg->is_root_shape() || !(csg->get_layer_mask() & p_shadow_mask)) {
				continue;
			}
			if (csg->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_OFF) {
				continue;
			}
			Array csg_meshes = csg->get_meshes();
			if (csg_meshes.size() < 2) {
				continue;
			}
			Ref<Mesh> mesh = csg_meshes[1];
			Transform local_xf = csg_meshes[0];
			append_mesh(mesh, csg->get_global_transform() * local_xf, csg->get_layer_mask(), raw_tris, aabbs);
		}
#endif

		if (raw_tris.empty()) {
			return;
		}
		std::vector<int> order(raw_tris.size());
		for (size_t i = 0; i < order.size(); i++) {
			order[i] = (int)i;
		}
		build_recursive(aabbs, order, 0, (int)order.size(), 0);
		tris.resize(order.size());
		for (size_t i = 0; i < order.size(); i++) {
			tris[i] = raw_tris[order[i]];
		}
	}

	bool is_occluded(const Vector3 &p_ro, const Vector3 &p_rd, float p_max_t, uint32_t p_light_shadow_mask = 0xFFFFFFFF) const {
		if (nodes.empty()) {
			return false;
		}
		Vector3 inv_rd(
				1.0f / (ABS(p_rd.x) < 1e-8f ? (p_rd.x < 0.0f ? -1e-8f : 1e-8f) : p_rd.x),
				1.0f / (ABS(p_rd.y) < 1e-8f ? (p_rd.y < 0.0f ? -1e-8f : 1e-8f) : p_rd.y),
				1.0f / (ABS(p_rd.z) < 1e-8f ? (p_rd.z < 0.0f ? -1e-8f : 1e-8f) : p_rd.z));

		if (!ray_aabb_hit(nodes[0].bounds, p_ro, inv_rd, p_max_t)) {
			return false;
		}

		int stack[32];
		int sp = 0;
		stack[sp++] = 0;
		while (sp > 0) {
			const BakeBVHNode &node = nodes[stack[--sp]];
			if (node.count > 0) {
				for (int i = 0; i < node.count; i++) {
					const BakeTri &tri = tris[node.left_or_first + i];
					if ((tri.layer_mask & p_light_shadow_mask) != 0 && ray_tri_hit(tri, p_ro, p_rd, p_max_t)) {
						return true;
					}
				}
			} else {
				int left = stack[sp];
				(void)left;
				int cur = (&node - nodes.data());
				int l_idx = cur + 1;
				int r_idx = node.left_or_first;
				if (sp < 31 && ray_aabb_hit(nodes[l_idx].bounds, p_ro, inv_rd, p_max_t)) {
					stack[sp++] = l_idx;
				}
				if (sp < 31 && ray_aabb_hit(nodes[r_idx].bounds, p_ro, inv_rd, p_max_t)) {
					stack[sp++] = r_idx;
				}
			}
		}
		return false;
	}
};

static float hash31(int x, int y, int z) {
	uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + (uint32_t)z * 2147483647u;
	h = (h ^ (h >> 13)) * 1274126177u;
	return (float)(h & 0xFFFFu) / 65535.0f;
}

static float value_noise_3d(const Vector3 &p) {
	int ix = (int)Math::floor(p.x);
	int iy = (int)Math::floor(p.y);
	int iz = (int)Math::floor(p.z);
	float fx = p.x - ix;
	float fy = p.y - iy;
	float fz = p.z - iz;
	fx = fx * fx * (3.0f - 2.0f * fx);
	fy = fy * fy * (3.0f - 2.0f * fy);
	fz = fz * fz * (3.0f - 2.0f * fz);

	float n000 = hash31(ix, iy, iz);
	float n100 = hash31(ix + 1, iy, iz);
	float n010 = hash31(ix, iy + 1, iz);
	float n110 = hash31(ix + 1, iy + 1, iz);
	float n001 = hash31(ix, iy, iz + 1);
	float n101 = hash31(ix + 1, iy, iz + 1);
	float n011 = hash31(ix, iy + 1, iz + 1);
	float n111 = hash31(ix + 1, iy + 1, iz + 1);

	float nx00 = Math::lerp(n000, n100, fx);
	float nx10 = Math::lerp(n010, n110, fx);
	float nx01 = Math::lerp(n001, n101, fx);
	float nx11 = Math::lerp(n011, n111, fx);
	return Math::lerp(Math::lerp(nx00, nx10, fy), Math::lerp(nx01, nx11, fy), fz);
}

static float fbm_noise_3d(const Vector3 &p) {
	return value_noise_3d(p) * 0.65f + value_noise_3d(p * 2.03f) * 0.35f;
}

} // namespace

BakedFogVolume::BakedFogVolume() {
}

void BakedFogVolume::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &BakedFogVolume::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &BakedFogVolume::is_enabled);

	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &BakedFogVolume::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &BakedFogVolume::get_extents);

	ClassDB::bind_method(D_METHOD("set_resolution", "resolution"), &BakedFogVolume::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &BakedFogVolume::get_resolution);

	ClassDB::bind_method(D_METHOD("set_baked_texture", "texture"), &BakedFogVolume::set_baked_texture);
	ClassDB::bind_method(D_METHOD("get_baked_texture"), &BakedFogVolume::get_baked_texture);

	ClassDB::bind_method(D_METHOD("set_baked_data", "data"), &BakedFogVolume::set_baked_data);
	ClassDB::bind_method(D_METHOD("get_baked_data"), &BakedFogVolume::get_baked_data);

	ClassDB::bind_method(D_METHOD("set_baked_data_resolution", "resolution"), &BakedFogVolume::set_baked_data_resolution);
	ClassDB::bind_method(D_METHOD("get_baked_data_resolution"), &BakedFogVolume::get_baked_data_resolution);

	ClassDB::bind_method(D_METHOD("set_density_multiplier", "multiplier"), &BakedFogVolume::set_density_multiplier);
	ClassDB::bind_method(D_METHOD("get_density_multiplier"), &BakedFogVolume::get_density_multiplier);

	ClassDB::bind_method(D_METHOD("set_emission_multiplier", "multiplier"), &BakedFogVolume::set_emission_multiplier);
	ClassDB::bind_method(D_METHOD("get_emission_multiplier"), &BakedFogVolume::get_emission_multiplier);

	ClassDB::bind_method(D_METHOD("set_albedo_tint", "tint"), &BakedFogVolume::set_albedo_tint);
	ClassDB::bind_method(D_METHOD("get_albedo_tint"), &BakedFogVolume::get_albedo_tint);

	ClassDB::bind_method(D_METHOD("set_edge_falloff", "falloff"), &BakedFogVolume::set_edge_falloff);
	ClassDB::bind_method(D_METHOD("get_edge_falloff"), &BakedFogVolume::get_edge_falloff);

	ClassDB::bind_method(D_METHOD("set_bake_base_density", "density"), &BakedFogVolume::set_bake_base_density);
	ClassDB::bind_method(D_METHOD("get_bake_base_density"), &BakedFogVolume::get_bake_base_density);

	ClassDB::bind_method(D_METHOD("set_bake_base_albedo", "albedo"), &BakedFogVolume::set_bake_base_albedo);
	ClassDB::bind_method(D_METHOD("get_bake_base_albedo"), &BakedFogVolume::get_bake_base_albedo);

	ClassDB::bind_method(D_METHOD("set_bake_base_emission", "emission"), &BakedFogVolume::set_bake_base_emission);
	ClassDB::bind_method(D_METHOD("get_bake_base_emission"), &BakedFogVolume::get_bake_base_emission);

	ClassDB::bind_method(D_METHOD("set_bake_local_volumes", "enable"), &BakedFogVolume::set_bake_local_volumes);
	ClassDB::bind_method(D_METHOD("get_bake_local_volumes"), &BakedFogVolume::get_bake_local_volumes);

	ClassDB::bind_method(D_METHOD("set_bake_lights", "enable"), &BakedFogVolume::set_bake_lights);
	ClassDB::bind_method(D_METHOD("get_bake_lights"), &BakedFogVolume::get_bake_lights);

	ClassDB::bind_method(D_METHOD("set_bake_shadows", "enable"), &BakedFogVolume::set_bake_shadows);
	ClassDB::bind_method(D_METHOD("get_bake_shadows"), &BakedFogVolume::get_bake_shadows);

	ClassDB::bind_method(D_METHOD("set_bake_shadow_samples", "samples"), &BakedFogVolume::set_bake_shadow_samples);
	ClassDB::bind_method(D_METHOD("get_bake_shadow_samples"), &BakedFogVolume::get_bake_shadow_samples);

	ClassDB::bind_method(D_METHOD("set_bake_shadow_bias", "bias"), &BakedFogVolume::set_bake_shadow_bias);
	ClassDB::bind_method(D_METHOD("get_bake_shadow_bias"), &BakedFogVolume::get_bake_shadow_bias);

	ClassDB::bind_method(D_METHOD("set_bake_noise_strength", "strength"), &BakedFogVolume::set_bake_noise_strength);
	ClassDB::bind_method(D_METHOD("get_bake_noise_strength"), &BakedFogVolume::get_bake_noise_strength);

	ClassDB::bind_method(D_METHOD("set_bake_noise_scale", "scale"), &BakedFogVolume::set_bake_noise_scale);
	ClassDB::bind_method(D_METHOD("get_bake_noise_scale"), &BakedFogVolume::get_bake_noise_scale);

	ClassDB::bind_method(D_METHOD("set_bake_filter", "filter"), &BakedFogVolume::set_bake_filter);
	ClassDB::bind_method(D_METHOD("get_bake_filter"), &BakedFogVolume::get_bake_filter);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &BakedFogVolume::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &BakedFogVolume::get_cull_mask);

	ClassDB::bind_method(D_METHOD("set_shadow_cull_mask", "mask"), &BakedFogVolume::set_shadow_cull_mask);
	ClassDB::bind_method(D_METHOD("get_shadow_cull_mask"), &BakedFogVolume::get_shadow_cull_mask);

	ClassDB::bind_method(D_METHOD("bake"), &BakedFogVolume::bake);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents"), "set_extents", "get_extents");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "resolution"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "baked_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture3D", PROPERTY_USAGE_EDITOR), "set_baked_texture", "get_baked_texture");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "baked_data_resolution", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_baked_data_resolution", "get_baked_data_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_BYTE_ARRAY, "baked_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_baked_data", "get_baked_data");

	ADD_GROUP("Playback", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "density_multiplier", PROPERTY_HINT_RANGE, "0,8,0.05,or_greater"), "set_density_multiplier", "get_density_multiplier");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "emission_multiplier", PROPERTY_HINT_RANGE, "0,8,0.05,or_greater"), "set_emission_multiplier", "get_emission_multiplier");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "albedo_tint", PROPERTY_HINT_COLOR_NO_ALPHA), "set_albedo_tint", "get_albedo_tint");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "edge_falloff", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_edge_falloff", "get_edge_falloff");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");

	ADD_GROUP("Bake Settings", "bake_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_base_density", PROPERTY_HINT_RANGE, "0,8,0.01,or_greater"), "set_bake_base_density", "get_bake_base_density");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "bake_base_albedo", PROPERTY_HINT_COLOR_NO_ALPHA), "set_bake_base_albedo", "get_bake_base_albedo");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "bake_base_emission", PROPERTY_HINT_COLOR_NO_ALPHA), "set_bake_base_emission", "get_bake_base_emission");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bake_local_volumes"), "set_bake_local_volumes", "get_bake_local_volumes");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bake_lights"), "set_bake_lights", "get_bake_lights");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bake_shadows"), "set_bake_shadows", "get_bake_shadows");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bake_shadow_samples", PROPERTY_HINT_RANGE, "1,8,1"), "set_bake_shadow_samples", "get_bake_shadow_samples");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_shadow_bias", PROPERTY_HINT_RANGE, "0.001,1.0,0.005"), "set_bake_shadow_bias", "get_bake_shadow_bias");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_noise_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_bake_noise_strength", "get_bake_noise_strength");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "bake_noise_scale", PROPERTY_HINT_RANGE, "0.01,10,0.05,or_greater"), "set_bake_noise_scale", "get_bake_noise_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bake_filter"), "set_bake_filter", "get_bake_filter");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shadow_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_shadow_cull_mask", "get_shadow_cull_mask");
}

void BakedFogVolume::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	update_gizmo();
}

void BakedFogVolume::set_extents(const Vector3 &p_extents) {
	extents = Vector3(MAX(p_extents.x, 0.1f), MAX(p_extents.y, 0.1f), MAX(p_extents.z, 0.1f));
	update_gizmo();
}

void BakedFogVolume::set_resolution(const Vector3 &p_res) {
	resolution = Vector3(
			CLAMP((int)Math::round(p_res.x), 4, 128),
			CLAMP((int)Math::round(p_res.y), 4, 128),
			CLAMP((int)Math::round(p_res.z), 4, 128));
}

void BakedFogVolume::set_density_multiplier(float p_mult) {
	density_multiplier = MAX(p_mult, 0.0f);
}

void BakedFogVolume::set_emission_multiplier(float p_mult) {
	emission_multiplier = MAX(p_mult, 0.0f);
}

void BakedFogVolume::set_edge_falloff(float p_falloff) {
	edge_falloff = CLAMP(p_falloff, 0.0f, 1.0f);
}

void BakedFogVolume::set_bake_base_density(float p_density) {
	bake_base_density = MAX(p_density, 0.0f);
}

void BakedFogVolume::set_bake_shadow_samples(int p_samples) {
	bake_shadow_samples = CLAMP(p_samples, 1, 8);
}

void BakedFogVolume::set_bake_shadow_bias(float p_bias) {
	bake_shadow_bias = MAX(p_bias, 0.001f);
}

void BakedFogVolume::set_bake_noise_strength(float p_strength) {
	bake_noise_strength = CLAMP(p_strength, 0.0f, 1.0f);
}

void BakedFogVolume::set_bake_noise_scale(float p_scale) {
	bake_noise_scale = MAX(p_scale, 0.001f);
}

void BakedFogVolume::set_baked_texture(const Ref<Texture3D> &p_tex) {
	baked_texture = p_tex;
	if (baked_texture.is_null()) {
		baked_data.resize(0);
		baked_data_resolution = Vector3(0, 0, 0);
	}
}

void BakedFogVolume::set_baked_data(const PoolByteArray &p_data) {
	baked_data = p_data;
	_rebuild_texture_from_baked_data();
}

void BakedFogVolume::set_baked_data_resolution(const Vector3 &p_res) {
	baked_data_resolution = p_res;
	_rebuild_texture_from_baked_data();
}

void BakedFogVolume::_rebuild_texture_from_baked_data() {
	int rx = (int)Math::round(baked_data_resolution.x);
	int ry = (int)Math::round(baked_data_resolution.y);
	int rz = (int)Math::round(baked_data_resolution.z);
	if (rx < 1 || ry < 1 || rz < 1) {
		return;
	}
	int slice_bytes = rx * ry * 8; // Image::FORMAT_RGBAH = 4 * 2 bytes per pixel
	if (baked_data.size() != slice_bytes * rz) {
		return;
	}

	Ref<Texture3D> tex;
	tex.instance();
	tex->create(rx, ry, rz, Image::FORMAT_RGBAH, bake_filter ? Texture::FLAG_FILTER : 0);

	PoolByteArray::Read r = baked_data.read();
	for (int z = 0; z < rz; z++) {
		PoolByteArray slice_buf;
		slice_buf.resize(slice_bytes);
		{
			PoolByteArray::Write w = slice_buf.write();
			memcpy(w.ptr(), r.ptr() + z * slice_bytes, slice_bytes);
		}
		Ref<Image> slice;
		slice.instance();
		slice->create(rx, ry, false, Image::FORMAT_RGBAH, slice_buf);
		tex->set_layer_data(slice, z);
	}

	baked_texture = tex;
	_change_notify();
}

AABB BakedFogVolume::get_world_aabb() const {
	AABB local(-extents, extents * 2.0f);
	return get_global_transform().xform(local);
}

bool BakedFogVolume::bake() {
	ERR_FAIL_COND_V_MSG(!is_inside_tree(), false, "BakedFogVolume must be inside the SceneTree to bake.");

	Viewport *vp = get_viewport();
	Node *search_root = vp ? (Node *)vp : (Node *)get_tree()->get_root();
	ERR_FAIL_COND_V(!search_root, false);

	std::vector<LocalFogVolume *> local_vols;
	std::vector<Light *> lights;
	std::vector<MeshInstance *> meshes;
#ifdef MODULE_CSG_ENABLED
	std::vector<CSGShape *> csg_shapes;
#endif

	struct Collector {
		static void visit(Node *n, bool collect_vols, bool collect_lights, bool collect_meshes,
				std::vector<LocalFogVolume *> &out_vols, std::vector<Light *> &out_lights, std::vector<MeshInstance *> &out_meshes
#ifdef MODULE_CSG_ENABLED
				, std::vector<CSGShape *> &out_csgs
#endif
				) {
			if (collect_vols) {
				if (LocalFogVolume *lv = Object::cast_to<LocalFogVolume>(n)) {
					if (lv->is_enabled() && lv->is_visible_in_tree()) {
						out_vols.push_back(lv);
					}
				}
			}
			if (collect_lights) {
				if (Light *l = Object::cast_to<Light>(n)) {
					if (l->is_visible_in_tree() && l->get_param(Light::PARAM_ENERGY) > 0.0001f) {
						out_lights.push_back(l);
					}
				}
			}
			if (collect_meshes) {
				if (MeshInstance *mi = Object::cast_to<MeshInstance>(n)) {
					if (mi->is_visible_in_tree() && mi->get_mesh().is_valid()) {
						out_meshes.push_back(mi);
					}
				}
#ifdef MODULE_CSG_ENABLED
				else if (CSGShape *csg = Object::cast_to<CSGShape>(n)) {
					if (csg->is_visible_in_tree() && csg->is_root_shape()) {
						out_csgs.push_back(csg);
					}
				}
#endif
			}
			for (int i = 0; i < n->get_child_count(); i++) {
				visit(n->get_child(i), collect_vols, collect_lights, collect_meshes, out_vols, out_lights, out_meshes
#ifdef MODULE_CSG_ENABLED
						, out_csgs
#endif
				);
			}
		}
	};

	Collector::visit(search_root, bake_local_volumes, bake_lights, bake_lights && bake_shadows, local_vols, lights, meshes
#ifdef MODULE_CSG_ENABLED
			, csg_shapes
#endif
	);

	BakeRaycaster raycaster;
#ifdef MODULE_CSG_ENABLED
	if (bake_lights && bake_shadows && (!meshes.empty() || !csg_shapes.empty())) {
		raycaster.build(meshes, csg_shapes, shadow_cull_mask);
	}
#else
	if (bake_lights && bake_shadows && !meshes.empty()) {
		raycaster.build(meshes, shadow_cull_mask);
	}
#endif

	int rx = CLAMP((int)Math::round(resolution.x), 4, 128);
	int ry = CLAMP((int)Math::round(resolution.y), 4, 128);
	int rz = CLAMP((int)Math::round(resolution.z), 4, 128);

	Ref<Texture3D> tex;
	tex.instance();
	tex->create(rx, ry, rz, Image::FORMAT_RGBAH, bake_filter ? Texture::FLAG_FILTER : 0);

	int slice_bytes = rx * ry * 8;
	PoolByteArray raw_bytes;
	raw_bytes.resize(slice_bytes * rz);
	PoolByteArray::Write raw_w = raw_bytes.write();

	Transform global_xf = get_global_transform();
	Color base_alb_lin = bake_base_albedo.to_linear();
	Color base_emi_lin = bake_base_emission.to_linear();

	for (int z = 0; z < rz; z++) {
		Ref<Image> slice;
		slice.instance();
		slice->create(rx, ry, false, Image::FORMAT_RGBAH);
		slice->lock();

		float uz = ((float)z + 0.5f) / (float)rz - 0.5f;
		for (int y = 0; y < ry; y++) {
			float uy = ((float)y + 0.5f) / (float)ry - 0.5f;
			for (int x = 0; x < rx; x++) {
				float ux = ((float)x + 0.5f) / (float)rx - 0.5f;
				Vector3 local_pos(ux * 2.0f * extents.x, uy * 2.0f * extents.y, uz * 2.0f * extents.z);
				Vector3 world_pos = global_xf.xform(local_pos);

				float density = bake_base_density;
				Vector3 albedo(base_alb_lin.r, base_alb_lin.g, base_alb_lin.b);
				Vector3 emission(base_emi_lin.r, base_emi_lin.g, base_emi_lin.b);

				if (bake_noise_strength > 0.001f) {
					float n = fbm_noise_3d(world_pos * bake_noise_scale);
					density *= Math::lerp(1.0f, n * 1.6f, bake_noise_strength);
				}

				for (size_t i = 0; i < local_vols.size(); i++) {
					LocalFogVolume *lv = local_vols[i];
					float factor = lv->sample_shape_factor(world_pos);
					if (factor <= 0.0001f) {
						continue;
					}
					float lv_d = lv->get_density() * factor;
					Color lv_alb = lv->get_albedo().to_linear();
					Color lv_emi = lv->get_emission().to_linear();
					Vector3 lv_a(lv_alb.r, lv_alb.g, lv_alb.b);
					Vector3 lv_e = Vector3(lv_emi.r, lv_emi.g, lv_emi.b) * lv->get_emission_energy() * factor;

					switch (lv->get_blend_mode()) {
						case LocalFogVolume::BLEND_ADDITIVE: {
							density += lv_d;
							float w = CLAMP(lv_d / MAX(density, 0.0001f), 0.0f, 1.0f);
							albedo = albedo.linear_interpolate(lv_a, w);
							emission += lv_e;
						} break;
						case LocalFogVolume::BLEND_MAX: {
							if (lv_d > density) {
								density = lv_d;
								albedo = lv_a;
							}
							emission = Vector3(MAX(emission.x, lv_e.x), MAX(emission.y, lv_e.y), MAX(emission.z, lv_e.z));
						} break;
						case LocalFogVolume::BLEND_MULTIPLY: {
							density *= Math::lerp(1.0f, lv->get_density(), factor);
						} break;
						case LocalFogVolume::BLEND_SUBTRACT: {
							density = MAX(0.0f, density - lv_d);
						} break;
					}
				}

				Vector3 light_accum(0.0f, 0.0f, 0.0f);
				if (bake_lights && density > 1e-5f) {
					Vector3 voxel_step(
							(2.0f * extents.x) / (float)rx,
							(2.0f * extents.y) / (float)ry,
							(2.0f * extents.z) / (float)rz);

					for (size_t li = 0; li < lights.size(); li++) {
						Light *l = lights[li];
						if (!(l->get_cull_mask() & cull_mask)) {
							continue;
						}
						Transform lxf = l->get_global_transform();
						Color lc = l->get_color().to_linear();
						float energy = l->get_param(Light::PARAM_ENERGY);
						if (l->is_negative()) {
							energy = -energy;
						}
						Vector3 l_col = Vector3(lc.r, lc.g, lc.b) * energy;

						Vector3 ray_dir;
						float max_t = 200.0f;
						float atten = 1.0f;
						bool is_point_light = false;

						if (Object::cast_to<DirectionalLight>(l)) {
							ray_dir = lxf.basis.get_axis(2).normalized();
							max_t = 500.0f;
						} else if (OmniLight *ol = Object::cast_to<OmniLight>(l)) {
							is_point_light = true;
							Vector3 to_light = lxf.origin - world_pos;
							float range = MAX(ol->get_param(Light::PARAM_RANGE), 0.01f);
							float dist = to_light.length();
							float decay = MAX(ol->get_param(Light::PARAM_ATTENUATION), 0.01f);

							if (ol->is_square_shape()) {
								Vector3 lp = lxf.affine_inverse().xform(world_pos);
								float sq_dist = MAX(ABS(lp.x), MAX(ABS(lp.y), ABS(lp.z)));
								if (sq_dist >= range) {
									continue;
								}
								atten = Math::pow(MAX(1.0f - sq_dist / range, 0.0f), decay);
							} else {
								if (dist >= range) {
									continue;
								}
								atten = Math::pow(MAX(1.0f - dist / range, 0.0f), decay);
							}

							if (dist < 1e-5f) {
								ray_dir = Vector3(0.0f, 1.0f, 0.0f);
								max_t = 0.0f;
							} else {
								ray_dir = to_light / dist;
								max_t = dist;
							}
						} else if (SpotLight *sl = Object::cast_to<SpotLight>(l)) {
							is_point_light = true;
							Vector3 to_light = lxf.origin - world_pos;
							float range = MAX(sl->get_param(Light::PARAM_RANGE), 0.01f);
							float dist = to_light.length();
							if (dist >= range) {
								continue;
							}
							atten = Math::pow(MAX(1.0f - dist / range, 0.0f), MAX(sl->get_param(Light::PARAM_ATTENUATION), 0.01f));

							if (dist < 1e-5f) {
								ray_dir = Vector3(0.0f, 1.0f, 0.0f);
								max_t = 0.0f;
							} else {
								ray_dir = to_light / dist;
								max_t = dist;
							}

							Vector3 l_local = lxf.affine_inverse().xform(world_pos);
							if (l_local.z >= -0.001f) {
								continue;
							}
							float spot_angle_rad = Math::deg2rad(sl->get_param(Light::PARAM_SPOT_ANGLE));
							float spot_cutoff = Math::cos(spot_angle_rad);
							float spot_atten_p = sl->get_param(Light::PARAM_SPOT_ATTENUATION);
							float rim = 1.0f;
							if (sl->is_square_shape()) {
								float tan_a = Math::sqrt(MAX(0.0f, 1.0f - spot_cutoff * spot_cutoff)) / MAX(0.001f, spot_cutoff);
								Vector2 suv(l_local.x / (-l_local.z * tan_a), l_local.y / (-l_local.z * tan_a));
								if (MAX(ABS(suv.x), ABS(suv.y)) > 1.0f) {
									continue;
								}
								float ux2 = suv.x * suv.x;
								float uy2 = suv.y * suv.y;
								rim = CLAMP(ux2 + uy2 - ux2 * uy2, 0.0001f, 1.0f);
							} else {
								Vector3 spot_fwd = (-lxf.basis.get_axis(2)).normalized();
								float scos = (-ray_dir).dot(spot_fwd);
								if (scos <= spot_cutoff) {
									continue;
								}
								rim = CLAMP((1.0f - scos) / MAX(1.0f - spot_cutoff, 0.0001f), 0.0001f, 1.0f);
							}
							atten *= MAX(0.0f, 1.0f - Math::pow(rim, MAX(spot_atten_p, 0.001f)));
						} else {
							continue;
						}

						if (atten <= 0.0001f) {
							continue;
						}

						float vis = 1.0f;
						if (bake_shadows && l->has_shadow() && (!is_point_light || max_t > bake_shadow_bias)) {
							uint32_t light_shadow_mask = l->get_shadow_cull_mask();
							float light_radius = is_point_light ? MAX(l->get_param(Light::PARAM_SIZE), 0.0f) : 0.0f;
							int occ_hits = 0;

							for (int s = 0; s < bake_shadow_samples; s++) {
								Vector3 jitter(0.0f, 0.0f, 0.0f);
								if (bake_shadow_samples > 1) {
									jitter = Vector3(
											(hash31(x + s * 17, y, z) - 0.5f) * voxel_step.x,
											(hash31(x, y + s * 31, z) - 0.5f) * voxel_step.y,
											(hash31(x, y, z + s * 47) - 0.5f) * voxel_step.z);
								}
								Vector3 sample_pos = world_pos + jitter;
								Vector3 sample_dir = ray_dir;
								float sample_max_t = max_t;

								if (is_point_light) {
									Vector3 light_target = lxf.origin;
									if (bake_shadow_samples > 1 && light_radius > 0.0001f) {
										float u = hash31(x + s * 53, y + s * 11, z);
										float v = hash31(x, y + s * 29, z + s * 61);
										float phi = u * (float)Math_TAU;
										float cos_t = 1.0f - 2.0f * v;
										float sin_t = Math::sqrt(MAX(0.0f, 1.0f - cos_t * cos_t));
										light_target += Vector3(sin_t * Math::cos(phi), cos_t, sin_t * Math::sin(phi)) * light_radius;
									}
									Vector3 to_target = light_target - sample_pos;
									float target_dist = to_target.length();
									if (target_dist <= bake_shadow_bias) {
										continue;
									}
									sample_dir = to_target / target_dist;
									sample_max_t = target_dist - bake_shadow_bias;
								}

								if (raycaster.is_occluded(sample_pos + sample_dir * bake_shadow_bias, sample_dir, sample_max_t, light_shadow_mask)) {
									occ_hits++;
								}
							}
							vis = 1.0f - (float)occ_hits / (float)bake_shadow_samples;
						}

						Color sc = l->get_shadow_color().to_linear();
						Vector3 shadow_tint(
								Math::lerp(sc.r, 1.0f, vis),
								Math::lerp(sc.g, 1.0f, vis),
								Math::lerp(sc.b, 1.0f, vis));
						light_accum += Vector3(l_col.x * shadow_tint.x, l_col.y * shadow_tint.y, l_col.z * shadow_tint.z) * atten;
					}
					light_accum.x = MAX(light_accum.x, 0.0f);
					light_accum.y = MAX(light_accum.y, 0.0f);
					light_accum.z = MAX(light_accum.z, 0.0f);
				}

				Vector3 stored_rgb(
						albedo.x * light_accum.x + emission.x,
						albedo.y * light_accum.y + emission.y,
						albedo.z * light_accum.z + emission.z);
				slice->set_pixel(x, y, Color(stored_rgb.x, stored_rgb.y, stored_rgb.z, MAX(density, 0.0f)));
			}
		}

		slice->unlock();
		PoolByteArray slice_data = slice->get_data();
		if (slice_data.size() == slice_bytes) {
			PoolByteArray::Read sr = slice_data.read();
			memcpy(raw_w.ptr() + z * slice_bytes, sr.ptr(), slice_bytes);
		}
		tex->set_layer_data(slice, z);
	}

	raw_w.release();
	baked_data_resolution = Vector3((float)rx, (float)ry, (float)rz);
	baked_data = raw_bytes;
	baked_texture = tex;
	_change_notify();
	return true;
}
