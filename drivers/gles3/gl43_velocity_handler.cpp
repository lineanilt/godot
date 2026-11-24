#include "gl43_velocity_handler.h"
#include "gl43_effect_helpers.h"

#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"
#include "modules/modules_enabled.gen.h"
#include "scene/3d/camera.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif

#include <cstring>

#ifndef GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020
#endif
#ifndef GL_TEXTURE_FETCH_BARRIER_BIT
#define GL_TEXTURE_FETCH_BARRIER_BIT 0x00000008
#endif
#ifndef GL_WRITE_ONLY
#define GL_WRITE_ONLY 0x88B9
#endif
#ifndef GL_FRAMEBUFFER_SRGB
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#endif
#ifndef GL_RGBA16F
#define GL_RGBA16F 0x881A
#endif
#ifndef GL_DEPTH_COMPONENT24
#define GL_DEPTH_COMPONENT24 0x81A6
#endif
#ifndef GL_TEXTURE_COMPARE_MODE
#define GL_TEXTURE_COMPARE_MODE 0x884C
#endif

namespace {

static const char *GL43_VEL_VERT_GLSL = R"GLSL(
#version 430 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aUv;
layout (location = 2) in vec3 aPrevPos;

uniform mat4 uCurrMVP;
uniform mat4 uPrevMVP;

out vec4 vCurrClip;
out vec4 vPrevClip;
out vec2 vUv;

void main() {
	vUv = aUv;
	vCurrClip = uCurrMVP * vec4(aPos, 1.0);
	vPrevClip = uPrevMVP * vec4(aPrevPos, 1.0);
	gl_Position = vCurrClip;
}
)GLSL";

static const char *GL43_VEL_FRAG_GLSL = R"GLSL(
#version 430 core
in vec4 vCurrClip;
in vec4 vPrevClip;
in vec2 vUv;

layout (location = 0) out vec4 outVel;

uniform sampler2D uAlbedoTex;
uniform int uUseAlphaScissor;
uniform float uAlphaThreshold;

void main() {
	if (uUseAlphaScissor != 0) {
		if (texture(uAlbedoTex, vUv).a < uAlphaThreshold) {
			discard;
		}
	}

	if (vCurrClip.w <= 0.0001 || vPrevClip.w <= 0.0001) {
		outVel = vec4(0.0, 0.0, 0.0, 1.0);
		return;
	}

	vec2 currNDC = vCurrClip.xy / vCurrClip.w;
	vec2 prevNDC = vPrevClip.xy / vPrevClip.w;
	vec2 velUV = (currNDC - prevNDC) * 0.5;
	outVel = vec4(velUV, 0.0, 1.0);
}
)GLSL";

static const char *GL43_VEL_RESOLVE_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutVelocity;

uniform sampler2D uVelRawTex;
uniform sampler2D uDepthTex;

uniform mat4 uInvProj;
uniform mat4 uInvCurrVP;
uniform mat4 uPrevVP;

uniform vec2 uViewportSize;
uniform float uVelScale;
uniform float uMaxVelPx;
uniform int uCameraMotion;

void main() {
	ivec2 px = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutVelocity);
	if (px.x >= sz.x || px.y >= sz.y) {
		return;
	}

	vec2 uv = (vec2(px) + 0.5) / vec2(sz);
	float rawDepth = texelFetch(uDepthTex, px, 0).r;

	float clampedDepth = min(rawDepth, 0.99999);
	vec4 vpPos = uInvProj * vec4(uv * 2.0 - 1.0, clampedDepth * 2.0 - 1.0, 1.0);
	float linearZ = max(-vpPos.z / max(vpPos.w, 0.0001), 0.001);

	vec4 rawV = texelFetch(uVelRawTex, px, 0);
	vec2 velUV = vec2(0.0);

	if (rawV.a > 0.5 && rawDepth < 0.99999) {
		velUV = rawV.rg * uVelScale;
	} else if (uCameraMotion != 0) {
		vec4 wp = uInvCurrVP * vec4(uv * 2.0 - 1.0, 0.99999 * 2.0 - 1.0, 1.0);
		if (abs(wp.w) > 0.0001) {
			wp.xyz /= wp.w;
			vec4 prevClip = uPrevVP * vec4(wp.xyz, 1.0);
			if (prevClip.w > 0.0001) {
				vec2 currNDC = uv * 2.0 - 1.0;
				vec2 prevNDC = prevClip.xy / prevClip.w;
				velUV = (currNDC - prevNDC) * 0.5 * uVelScale;
			}
		}
	}

	vec2 velPx = velUV * uViewportSize;
	float speedPx = length(velPx);
	if (uMaxVelPx > 0.0 && speedPx > uMaxVelPx && speedPx > 0.0001) {
		velUV *= (uMaxVelPx / speedPx);
		speedPx = uMaxVelPx;
	}

	imageStore(uOutVelocity, px, vec4(velUV.x, velUV.y, linearZ, speedPx));
}
)GLSL";

} // namespace

bool GL43VelocityHandler::init_gl(const char *p_tag) {
	if (vel_program != 0 && resolve_program != 0) {
		return true;
	}
	vel_program = GL43EffectHelpers::link_vf_program(GL43_VEL_VERT_GLSL, GL43_VEL_FRAG_GLSL, p_tag, "Velocity");
	resolve_program = GL43EffectHelpers::link_compute_program(GL43_VEL_RESOLVE_COMPUTE_GLSL, p_tag, "VelocityResolve");
	return vel_program != 0 && resolve_program != 0;
}

void GL43VelocityHandler::cleanup_gl() {
	for (auto &kv : mesh_cache) {
		if (kv.second.gl_vao) {
			GLuint v = (GLuint)kv.second.gl_vao;
			glDeleteVertexArrays(1, &v);
		}
		if (kv.second.gl_vbo) {
			GLuint b = (GLuint)kv.second.gl_vbo;
			glDeleteBuffers(1, &b);
		}
	}
	mesh_cache.clear();

	for (auto &kv : skinned_gl_cache) {
		if (kv.second.gl_vao) {
			GLuint v = (GLuint)kv.second.gl_vao;
			glDeleteVertexArrays(1, &v);
		}
		if (kv.second.gl_curr_vbo) {
			GLuint b = (GLuint)kv.second.gl_curr_vbo;
			glDeleteBuffers(1, &b);
		}
		if (kv.second.gl_prev_vbo) {
			GLuint b = (GLuint)kv.second.gl_prev_vbo;
			glDeleteBuffers(1, &b);
		}
	}
	skinned_gl_cache.clear();

	for (size_t i = 0; i < gl_buffers_to_delete.size(); i++) {
		if (gl_buffers_to_delete[i].vao) {
			GLuint v = (GLuint)gl_buffers_to_delete[i].vao;
			glDeleteVertexArrays(1, &v);
		}
		if (gl_buffers_to_delete[i].vbo_a) {
			GLuint b = (GLuint)gl_buffers_to_delete[i].vbo_a;
			glDeleteBuffers(1, &b);
		}
		if (gl_buffers_to_delete[i].vbo_b) {
			GLuint b = (GLuint)gl_buffers_to_delete[i].vbo_b;
			glDeleteBuffers(1, &b);
		}
	}
	gl_buffers_to_delete.clear();

	for (auto &kv : vp_gl_states) {
		ViewportGLState &st = kv.second;
		if (st.vel_fbo) {
			GLuint f = (GLuint)st.vel_fbo;
			glDeleteFramebuffers(1, &f);
			st.vel_fbo = 0;
		}
		if (st.vel_raw_tex) {
			GLuint t = (GLuint)st.vel_raw_tex;
			glDeleteTextures(1, &t);
			st.vel_raw_tex = 0;
		}
		if (st.vel_depth_tex) {
			GLuint t = (GLuint)st.vel_depth_tex;
			glDeleteTextures(1, &t);
			st.vel_depth_tex = 0;
		}
	}
	vp_gl_states.clear();

	if (vel_program) {
		glDeleteProgram((GLuint)vel_program);
		vel_program = 0;
	}
	if (resolve_program) {
		glDeleteProgram((GLuint)resolve_program);
		resolve_program = 0;
	}
}

void GL43VelocityHandler::reset_history() {
	has_prev_frame = false;
	for (auto &kv : vp_gl_states) {
		kv.second.has_prev_frame = false;
	}
	prev_xforms.clear();
	prev_skinned_poses.clear();
	valid = false;
}

void GL43VelocityHandler::begin_collection() {
	snapshot_counter++;
	pinned_textures.clear();
	viewport_targets.clear();
	instances.clear();
	skinned_instances.clear();
}

void GL43VelocityHandler::add_viewport_target(
		uint32_t p_vp_id,
		Camera *p_cam,
		const Size2 &p_vp_size,
		int p_vel_w,
		int p_vel_h,
		bool p_camera_motion,
		float p_eff_vel_scale,
		float p_max_vel_px,
		RID p_velocity_tex_rid,
		uint64_t p_collect_counter) {

	ViewportGLState &vst = vp_gl_states[p_vp_id];
	vst.last_seen = p_collect_counter;

	CameraMatrix v_view, v_proj, v_vp, v_inv_proj, v_inv_vp;
	GL43EffectHelpers::compute_camera_matrices(p_cam, p_vp_size, v_view, v_proj, v_vp, v_inv_proj, v_inv_vp);

	ViewportTarget vt;
	vt.vp_id = p_vp_id;
	vt.viewport_w = (int)p_vp_size.x;
	vt.viewport_h = (int)p_vp_size.y;
	vt.vel_w = p_vel_w;
	vt.vel_h = p_vel_h;
	vt.eff_vel_scale = p_eff_vel_scale;
	vt.max_vel_px = p_max_vel_px;
	vt.camera_motion = p_camera_motion;
	vt.velocity_tex_rid = p_velocity_tex_rid;

	GL43EffectHelpers::camera_matrix_to_float16(v_vp, vt.curr_vp_mat);
	GL43EffectHelpers::camera_matrix_to_float16(v_inv_proj, vt.inv_proj_mat);
	GL43EffectHelpers::camera_matrix_to_float16(v_inv_vp, vt.inv_curr_vp_mat);
	GL43EffectHelpers::extract_frustum_planes(v_vp, vt.frustum_planes);

	if (p_camera_motion && vst.has_prev_frame) {
		std::memcpy(vt.prev_vp_mat, vst.prev_vp_mat, sizeof(vt.prev_vp_mat));
	} else {
		std::memcpy(vt.prev_vp_mat, vt.curr_vp_mat, sizeof(vt.curr_vp_mat));
	}
	std::memcpy(vst.prev_vp_mat, vt.curr_vp_mat, sizeof(vst.prev_vp_mat));
	vst.has_prev_frame = true;

	viewport_targets.push_back(vt);
}

void GL43VelocityHandler::prune_stale_viewports(uint64_t p_collect_counter, bool p_gl_active_on_current_thread) {
	for (auto it = vp_gl_states.begin(); it != vp_gl_states.end(); ) {
		if (it->second.last_seen != p_collect_counter) {
			it->second.has_prev_frame = false;
			if (p_collect_counter - it->second.last_seen > 60ULL) {
				if (p_gl_active_on_current_thread) {
					SharedGL43Context::ScopedScope gl43;
					if (gl43.is_active()) {
						if (it->second.vel_fbo) {
							GLuint f = (GLuint)it->second.vel_fbo;
							glDeleteFramebuffers(1, &f);
						}
						if (it->second.vel_raw_tex) {
							GLuint t = (GLuint)it->second.vel_raw_tex;
							glDeleteTextures(1, &t);
						}
						if (it->second.vel_depth_tex) {
							GLuint t = (GLuint)it->second.vel_depth_tex;
							glDeleteTextures(1, &t);
						}
					}
				}
				it = vp_gl_states.erase(it);
				continue;
			}
		}
		++it;
	}
}

void GL43VelocityHandler::_rebuild_mesh_cache(const std::vector<MeshInstance *> &p_meshes, const std::vector<Ref<Mesh>> &p_extra_meshes) {
	for (auto it = mesh_cache.begin(); it != mesh_cache.end(); ) {
		if (it->second.cached_mesh_instance_id != 0 && ObjectDB::get_instance((ObjectID)it->second.cached_mesh_instance_id) == nullptr) {
			PendingGLDelete del;
			del.vao = it->second.gl_vao;
			del.vbo_a = it->second.gl_vbo;
			if (del.vao || del.vbo_a) gl_buffers_to_delete.push_back(del);
			it = mesh_cache.erase(it);
		} else {
			++it;
		}
	}

	auto process_mesh = [&](const Ref<Mesh> &p_mesh) {
		if (p_mesh.is_null() || !p_mesh->get_rid().is_valid()) return;
		uint32_t mid = p_mesh->get_rid().get_id();
		uint64_t m_inst_id = (uint64_t)p_mesh->get_instance_id();
		AABB mesh_aabb = p_mesh->get_aabb();

		auto existing = mesh_cache.find(mid);
		if (existing != mesh_cache.end()) {
			if (existing->second.cached_mesh_instance_id == m_inst_id &&
					GL43EffectHelpers::aabb_same(existing->second.cached_mesh_aabb, mesh_aabb)) {
				return;
			}
			PendingGLDelete del;
			del.vao = existing->second.gl_vao;
			del.vbo_a = existing->second.gl_vbo;
			if (del.vao || del.vbo_a) gl_buffers_to_delete.push_back(del);
			mesh_cache.erase(existing);
		}

		MeshGeom geom;
		geom.mesh_id = mid;
		geom.cached_mesh_instance_id = m_inst_id;
		geom.cached_mesh_aabb = mesh_aabb;
		bool first_aabb = true;

		int surf_count = p_mesh->get_surface_count();
		for (int s = 0; s < surf_count; s++) {
			if (p_mesh->surface_get_primitive_type(s) != Mesh::PRIMITIVE_TRIANGLES) continue;
			Array arrays = p_mesh->surface_get_arrays(s);
			if (arrays.size() <= Mesh::ARRAY_VERTEX) continue;

			PoolVector<Vector3> verts = arrays[Mesh::ARRAY_VERTEX];
			int vcount = verts.size();
			if (vcount < 3) continue;

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
				geom.has_skin_weights = true;
				geom.bones_per_vert = MAX(geom.bones_per_vert, bpv);
			}

			PoolVector<int> indices;
			bool has_indices = arrays[Mesh::ARRAY_INDEX].get_type() == Variant::POOL_INT_ARRAY;
			if (has_indices) indices = arrays[Mesh::ARRAY_INDEX];

			PoolVector<Vector3>::Read vr = verts.read();
			PoolVector<Vector2>::Read uvr = uvs.read();
			PoolVector<int>::Read br = bones.read();
			PoolVector<real_t>::Read wr = weights.read();
			PoolVector<int>::Read ir = indices.read();

			uint32_t bind_base = (uint32_t)geom.bind_pos.size();
			for (int v = 0; v < vcount; v++) {
				geom.bind_pos.push_back(vr[v]);
				geom.bind_uv.push_back(has_uvs ? uvr[v] : Vector2());
				if (first_aabb) {
					geom.local_aabb = AABB(vr[v], Vector3());
					first_aabb = false;
				} else {
					geom.local_aabb.expand_to(vr[v]);
				}
				if (has_skin) {
					for (int k = 0; k < bpv; k++) {
						geom.bind_bones.push_back(br[v * bpv + k]);
						geom.bind_weights.push_back((float)wr[v * bpv + k]);
					}
				}
			}

			SurfaceRange sr;
			sr.first_vert = geom.total_verts;

			int tri_idx_count = has_indices ? indices.size() : vcount;
			for (int i = 0; i + 2 < tri_idx_count; i += 3) {
				int i0 = has_indices ? ir[i + 0] : (i + 0);
				int i1 = has_indices ? ir[i + 1] : (i + 1);
				int i2 = has_indices ? ir[i + 2] : (i + 2);
				if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= vcount || i1 >= vcount || i2 >= vcount) {
					continue;
				}
				const Vector3 &p0 = vr[i0];
				const Vector3 &p1 = vr[i1];
				const Vector3 &p2 = vr[i2];
				if ((p1 - p0).cross(p2 - p0).length_squared() < 1e-12f) continue;

				const Vector2 uv0 = has_uvs ? uvr[i0] : Vector2();
				const Vector2 uv1 = has_uvs ? uvr[i1] : Vector2();
				const Vector2 uv2 = has_uvs ? uvr[i2] : Vector2();

				geom.verts.push_back((float)p0.x); geom.verts.push_back((float)p0.y); geom.verts.push_back((float)p0.z);
				geom.verts.push_back((float)uv0.x); geom.verts.push_back((float)uv0.y);

				geom.verts.push_back((float)p1.x); geom.verts.push_back((float)p1.y); geom.verts.push_back((float)p1.z);
				geom.verts.push_back((float)uv1.x); geom.verts.push_back((float)uv1.y);

				geom.verts.push_back((float)p2.x); geom.verts.push_back((float)p2.y); geom.verts.push_back((float)p2.z);
				geom.verts.push_back((float)uv2.x); geom.verts.push_back((float)uv2.y);

				geom.tri_verts.push_back(bind_base + (uint32_t)i0);
				geom.tri_verts.push_back(bind_base + (uint32_t)i1);
				geom.tri_verts.push_back(bind_base + (uint32_t)i2);
				geom.total_verts += 3;
			}

			sr.vert_count = geom.total_verts - sr.first_vert;
			geom.surfaces.push_back(sr);
		}

		if (geom.total_verts == 0) return;

		if (geom.has_skin_weights && geom.bind_bones.size() < geom.bind_pos.size() * (size_t)geom.bones_per_vert) {
			geom.bind_bones.resize(geom.bind_pos.size() * (size_t)geom.bones_per_vert, 0);
			geom.bind_weights.resize(geom.bind_pos.size() * (size_t)geom.bones_per_vert, 0.0f);
		}

		geom.local_aabb.grow_by(0.002f);
		mesh_cache[mid] = std::move(geom);
	};

	for (MeshInstance *mi : p_meshes) {
		process_mesh(mi->get_mesh());
	}
	for (const Ref<Mesh> &m : p_extra_meshes) {
		process_mesh(m);
	}
}

void GL43VelocityHandler::collect_scene_meshes(
		Node *p_search_root,
		uint32_t p_cull_mask,
		bool p_alpha_scissor_support,
		bool p_object_motion,
		bool p_skinned_meshes) {

	std::vector<MeshInstance *> meshes;
	std::vector<MultiMeshInstance *> multimeshes;
	std::vector<Ref<Mesh>> extra_meshes;
#ifdef MODULE_CSG_ENABLED
	struct CsgEntry {
		CSGShape *csg;
		Transform local_xf;
		Ref<Mesh> mesh;
	};
	std::vector<CsgEntry> csg_entries;
#endif

	struct SceneCollector {
		static void visit(Node *p_node, uint32_t p_mask,
				std::vector<MeshInstance *> &r_meshes,
				std::vector<MultiMeshInstance *> &r_multimeshes,
				std::vector<Ref<Mesh>> &r_extra_meshes
#ifdef MODULE_CSG_ENABLED
				, std::vector<CsgEntry> &r_csgs
#endif
				) {
			if (MeshInstance *mi = Object::cast_to<MeshInstance>(p_node)) {
				if (mi->is_visible_in_tree() && (mi->get_layer_mask() & p_mask) &&
						mi->get_cast_shadows_setting() != GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY &&
						mi->get_mesh().is_valid()) {
					r_meshes.push_back(mi);
				}
			} else if (MultiMeshInstance *mmi = Object::cast_to<MultiMeshInstance>(p_node)) {
				if (mmi->is_visible_in_tree() && (mmi->get_layer_mask() & p_mask) &&
						mmi->get_cast_shadows_setting() != GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY &&
						mmi->get_multimesh().is_valid() && mmi->get_multimesh()->get_mesh().is_valid()) {
					r_multimeshes.push_back(mmi);
					r_extra_meshes.push_back(mmi->get_multimesh()->get_mesh());
				}
			}
#ifdef MODULE_CSG_ENABLED
			else if (CSGShape *csg = Object::cast_to<CSGShape>(p_node)) {
				if (csg->is_visible_in_tree() && csg->is_root_shape() && (csg->get_layer_mask() & p_mask) &&
						csg->get_cast_shadows_setting() != GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY) {
					Array csg_meshes = csg->get_meshes();
					if (csg_meshes.size() >= 2) {
						Ref<Mesh> m = csg_meshes[1];
						if (m.is_valid()) {
							Transform lxf = csg_meshes[0];
							r_csgs.push_back({ csg, lxf, m });
							r_extra_meshes.push_back(m);
						}
					}
				}
			}
#endif
			for (int i = 0; i < p_node->get_child_count(); i++) {
				visit(p_node->get_child(i), p_mask, r_meshes, r_multimeshes, r_extra_meshes
#ifdef MODULE_CSG_ENABLED
						, r_csgs
#endif
				);
			}
		}
	};

	if (p_search_root) {
		SceneCollector::visit(p_search_root, p_cull_mask, meshes, multimeshes, extra_meshes
#ifdef MODULE_CSG_ENABLED
				, csg_entries
#endif
		);
	}

	_rebuild_mesh_cache(meshes, extra_meshes);

	auto resolve_surfaces = [&](const MeshGeom &p_geom, auto get_mat_fn, std::vector<SurfaceRange> &r_surfaces) {
		r_surfaces.clear();
		for (size_t s = 0; s < p_geom.surfaces.size(); s++) {
			SurfaceRange sr = p_geom.surfaces[s];
			Ref<SpatialMaterial> smat = get_mat_fn((int)s);
			if (smat.is_valid()) {
				bool is_transparent = smat->get_feature(SpatialMaterial::FEATURE_TRANSPARENT);
				bool is_scissor = smat->get_flag(SpatialMaterial::FLAG_USE_ALPHA_SCISSOR);
				if (is_transparent && !is_scissor) {
					continue;
				}
				sr.cull_mode = (int32_t)smat->get_cull_mode();
				if (p_alpha_scissor_support && is_scissor) {
					Ref<Texture> alb_tex = smat->get_texture(SpatialMaterial::TEXTURE_ALBEDO);
					if (alb_tex.is_valid() && alb_tex->get_rid().is_valid()) {
						sr.alpha_scissor = true;
						sr.alpha_threshold = smat->get_alpha_scissor_threshold();
						sr.albedo_tex_rid = alb_tex->get_rid();
						pinned_textures.push_back(alb_tex);
					}
				}
			}
			r_surfaces.push_back(sr);
		}
	};

	for (MeshInstance *mi : meshes) {
		Ref<Mesh> mesh = mi->get_mesh();
		uint32_t mid = mesh->get_rid().get_id();
		auto it = mesh_cache.find(mid);
		if (it == mesh_cache.end()) continue;
		const MeshGeom &geom = it->second;

		Transform xf = mi->get_global_transform();
		float curr_model[16];
		GL43EffectHelpers::transform_to_float16(xf, curr_model);

		PrevKey key = { (uint64_t)mi->get_instance_id(), 0 };
		float prev_model[16];
		auto pit = prev_xforms.find(key);
		if (has_prev_frame && pit != prev_xforms.end()) {
			std::memcpy(prev_model, pit->second.model_mat, sizeof(prev_model));
		} else {
			std::memcpy(prev_model, curr_model, sizeof(curr_model));
		}
		PrevObjectState &pstate = prev_xforms[key];
		std::memcpy(pstate.model_mat, curr_model, sizeof(curr_model));
		pstate.last_seen = snapshot_counter;

		AABB local_bounds = geom.local_aabb;
		std::vector<float> curr_skin_verts;
		std::vector<float> curr_skin_xyz;
		bool is_skin = false;
		if (p_skinned_meshes && geom.has_skin_weights) {
			std::vector<Vector3> posed_pos;
			if (GL43SkinningHelper::pose_mesh_positions(mi, geom.bind_pos, geom.bind_bones, geom.bind_weights, geom.bones_per_vert, posed_pos, &local_bounds)) {
				GL43SkinningHelper::pack_pos_uv_triangles(posed_pos, geom.bind_uv, geom.tri_verts, curr_skin_verts, &curr_skin_xyz);
				is_skin = true;
			}
		}

		AABB world_aabb = xf.xform(local_bounds);

		VelocityInstance inst;
		inst.mesh_id = mid;
		inst.world_aabb = world_aabb;
		inst.flip_facing = (xf.basis.determinant() < 0.0f);
		resolve_surfaces(geom, [&](int s) { return mi->get_active_material(s); }, inst.surfaces);
		if (inst.surfaces.empty()) {
			if (is_skin) {
				PrevSkinnedState &ps = prev_skinned_poses[(uint64_t)mi->get_instance_id()];
				ps.posed_xyz = std::move(curr_skin_xyz);
				ps.last_seen = snapshot_counter;
			}
			continue;
		}

		if (is_skin) {
			uint64_t oid = (uint64_t)mi->get_instance_id();
			SkinnedSnapshot sk;
			sk.obj_id = oid;
			sk.vertex_count = (int32_t)(curr_skin_xyz.size() / 3);
			sk.curr_verts = std::move(curr_skin_verts);

			auto ps_it = prev_skinned_poses.find(oid);
			if (p_object_motion && has_prev_frame && ps_it != prev_skinned_poses.end() && ps_it->second.posed_xyz.size() == curr_skin_xyz.size()) {
				sk.prev_verts = ps_it->second.posed_xyz;
			} else {
				sk.prev_verts = curr_skin_xyz;
			}

			PrevSkinnedState &ps = prev_skinned_poses[oid];
			ps.posed_xyz = std::move(curr_skin_xyz);
			ps.last_seen = snapshot_counter;

			inst.is_skinned = true;
			inst.skinned_snap_idx = (int32_t)skinned_instances.size();
			skinned_instances.push_back(std::move(sk));
		}

		const float *eff_prev_model = p_object_motion ? prev_model : curr_model;
		std::memcpy(inst.curr_model, curr_model, sizeof(inst.curr_model));
		std::memcpy(inst.prev_model, eff_prev_model, sizeof(inst.prev_model));
		instances.push_back(std::move(inst));
	}

	for (MultiMeshInstance *mmi : multimeshes) {
		Ref<MultiMesh> mm = mmi->get_multimesh();
		if (mm->get_transform_format() != MultiMesh::TRANSFORM_3D) continue;
		Ref<Mesh> mesh = mm->get_mesh();
		uint32_t mid = mesh->get_rid().get_id();
		auto it = mesh_cache.find(mid);
		if (it == mesh_cache.end()) continue;
		const MeshGeom &geom = it->second;

		std::vector<SurfaceRange> mmi_surfaces;
		Ref<Material> override_mat = mmi->get_material_override();
		resolve_surfaces(geom, [&](int s) {
			return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
		}, mmi_surfaces);
		if (mmi_surfaces.empty()) continue;

		Transform base_xf = mmi->get_global_transform();
		int count = mm->get_visible_instance_count();
		if (count < 0) count = mm->get_instance_count();
		uint64_t mmi_id = (uint64_t)mmi->get_instance_id();

		for (int i = 0; i < count; i++) {
			Transform xf = base_xf * mm->get_instance_transform(i);
			float curr_model[16];
			GL43EffectHelpers::transform_to_float16(xf, curr_model);

			PrevKey key = { mmi_id, (uint32_t)(i + 1) };
			float prev_model[16];
			auto pit = prev_xforms.find(key);
			if (has_prev_frame && pit != prev_xforms.end()) {
				std::memcpy(prev_model, pit->second.model_mat, sizeof(prev_model));
			} else {
				std::memcpy(prev_model, curr_model, sizeof(curr_model));
			}
			PrevObjectState &pstate = prev_xforms[key];
			std::memcpy(pstate.model_mat, curr_model, sizeof(curr_model));
			pstate.last_seen = snapshot_counter;

			VelocityInstance inst;
			inst.mesh_id = mid;
			inst.world_aabb = xf.xform(geom.local_aabb);
			inst.flip_facing = (xf.basis.determinant() < 0.0f);
			inst.surfaces = mmi_surfaces;

			const float *eff_prev_model = p_object_motion ? prev_model : curr_model;
			std::memcpy(inst.curr_model, curr_model, sizeof(inst.curr_model));
			std::memcpy(inst.prev_model, eff_prev_model, sizeof(inst.prev_model));
			instances.push_back(std::move(inst));
		}
	}

#ifdef MODULE_CSG_ENABLED
	for (const CsgEntry &entry : csg_entries) {
		CSGShape *csg = entry.csg;
		Ref<Mesh> mesh = entry.mesh;
		uint32_t mid = mesh->get_rid().get_id();
		auto it = mesh_cache.find(mid);
		if (it == mesh_cache.end()) continue;
		const MeshGeom &geom = it->second;

		std::vector<SurfaceRange> csg_surfaces;
		Ref<Material> override_mat = csg->get_material_override();
		resolve_surfaces(geom, [&](int s) {
			return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s);
		}, csg_surfaces);
		if (csg_surfaces.empty()) continue;

		Transform xf = csg->get_global_transform() * entry.local_xf;
		float curr_model[16];
		GL43EffectHelpers::transform_to_float16(xf, curr_model);

		PrevKey key = { (uint64_t)csg->get_instance_id(), 0 };
		float prev_model[16];
		auto pit = prev_xforms.find(key);
		if (has_prev_frame && pit != prev_xforms.end()) {
			std::memcpy(prev_model, pit->second.model_mat, sizeof(prev_model));
		} else {
			std::memcpy(prev_model, curr_model, sizeof(curr_model));
		}
		PrevObjectState &pstate = prev_xforms[key];
		std::memcpy(pstate.model_mat, curr_model, sizeof(curr_model));
		pstate.last_seen = snapshot_counter;

		VelocityInstance inst;
		inst.mesh_id = mid;
		inst.world_aabb = xf.xform(geom.local_aabb);
		inst.flip_facing = (xf.basis.determinant() < 0.0f);
		inst.surfaces = std::move(csg_surfaces);

		const float *eff_prev_model = p_object_motion ? prev_model : curr_model;
		std::memcpy(inst.curr_model, curr_model, sizeof(inst.curr_model));
		std::memcpy(inst.prev_model, eff_prev_model, sizeof(inst.prev_model));
		instances.push_back(std::move(inst));
	}
#endif

	if ((snapshot_counter % 60ULL) == 0ULL) {
		for (auto it = prev_xforms.begin(); it != prev_xforms.end(); ) {
			if (snapshot_counter - it->second.last_seen > 60ULL) {
				it = prev_xforms.erase(it);
			} else {
				++it;
			}
		}
		for (auto it = prev_skinned_poses.begin(); it != prev_skinned_poses.end(); ) {
			if (snapshot_counter - it->second.last_seen > 60ULL) {
				it = prev_skinned_poses.erase(it);
			} else {
				++it;
			}
		}
	}
}

void GL43VelocityHandler::end_collection(bool p_valid) {
	if (p_valid) {
		has_prev_frame = true;
	}
	valid = p_valid;
}

void GL43VelocityHandler::_ensure_vel_fbo_size(ViewportGLState &r_state, int p_w, int p_h) {
	if (r_state.vel_fbo != 0 && r_state.alloc_vel_w == p_w && r_state.alloc_vel_h == p_h) {
		return;
	}
	r_state.alloc_vel_w = p_w;
	r_state.alloc_vel_h = p_h;

	if (r_state.vel_fbo == 0) {
		GLuint f = 0;
		glGenFramebuffers(1, &f);
		r_state.vel_fbo = (uint32_t)f;
	}

	glActiveTexture(GL_TEXTURE0);
	GL43EffectHelpers::ensure_rgba16f_texture(r_state.vel_raw_tex, p_w, p_h, GL_NEAREST);
	GL43EffectHelpers::ensure_depth24_texture(r_state.vel_depth_tex, p_w, p_h);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)r_state.vel_fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, (GLuint)r_state.vel_raw_tex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, (GLuint)r_state.vel_depth_tex, 0);
	GLenum buf = GL_COLOR_ATTACHMENT0;
	glDrawBuffers(1, &buf);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool GL43VelocityHandler::render_velocities(RasterizerStorageGLES3 *p_rs) {
	if (!p_rs || !valid || viewport_targets.empty() || vel_program == 0 || resolve_program == 0) {
		return false;
	}

	for (size_t i = 0; i < gl_buffers_to_delete.size(); i++) {
		if (gl_buffers_to_delete[i].vao) {
			GLuint v = (GLuint)gl_buffers_to_delete[i].vao;
			glDeleteVertexArrays(1, &v);
		}
		if (gl_buffers_to_delete[i].vbo_a) {
			GLuint b = (GLuint)gl_buffers_to_delete[i].vbo_a;
			glDeleteBuffers(1, &b);
		}
		if (gl_buffers_to_delete[i].vbo_b) {
			GLuint b = (GLuint)gl_buffers_to_delete[i].vbo_b;
			glDeleteBuffers(1, &b);
		}
	}
	gl_buffers_to_delete.clear();

	glDisable(GL_FRAMEBUFFER_SRGB);
	frame_counter++;

	for (auto &kv : mesh_cache) {
		MeshGeom &g = kv.second;
		if (!g.gpu_uploaded && !g.verts.empty()) {
			GLuint vao = 0, vbo = 0;
			glGenVertexArrays(1, &vao);
			glGenBuffers(1, &vbo);
			g.gl_vao = (uint32_t)vao;
			g.gl_vbo = (uint32_t)vbo;
			glBindVertexArray(vao);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, g.verts.size() * sizeof(float), g.verts.data(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
			g.gpu_uploaded = true;
		}
	}

	for (size_t s = 0; s < skinned_instances.size(); s++) {
		const SkinnedSnapshot &sk = skinned_instances[s];
		SkinnedMeshGL &sgl = skinned_gl_cache[sk.obj_id];
		if (sgl.gl_vao == 0) {
			GLuint vao = 0, curr_vbo = 0, prev_vbo = 0;
			glGenVertexArrays(1, &vao);
			glGenBuffers(1, &curr_vbo);
			glGenBuffers(1, &prev_vbo);
			sgl.gl_vao = (uint32_t)vao;
			sgl.gl_curr_vbo = (uint32_t)curr_vbo;
			sgl.gl_prev_vbo = (uint32_t)prev_vbo;
			glBindVertexArray(vao);

			glBindBuffer(GL_ARRAY_BUFFER, curr_vbo);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(3 * sizeof(float)));

			glBindBuffer(GL_ARRAY_BUFFER, prev_vbo);
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
		}
		glBindBuffer(GL_ARRAY_BUFFER, (GLuint)sgl.gl_curr_vbo);
		glBufferData(GL_ARRAY_BUFFER, sk.curr_verts.size() * sizeof(float), sk.curr_verts.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, (GLuint)sgl.gl_prev_vbo);
		glBufferData(GL_ARRAY_BUFFER, sk.prev_verts.size() * sizeof(float), sk.prev_verts.data(), GL_DYNAMIC_DRAW);

		sgl.vertex_count = sk.vertex_count;
		sgl.last_frame_seen = frame_counter;
	}

	for (auto it = skinned_gl_cache.begin(); it != skinned_gl_cache.end(); ) {
		if (frame_counter - it->second.last_frame_seen > 60) {
			if (it->second.gl_vao) {
				GLuint v = (GLuint)it->second.gl_vao;
				glDeleteVertexArrays(1, &v);
			}
			if (it->second.gl_curr_vbo) {
				GLuint b = (GLuint)it->second.gl_curr_vbo;
				glDeleteBuffers(1, &b);
			}
			if (it->second.gl_prev_vbo) {
				GLuint b = (GLuint)it->second.gl_prev_vbo;
				glDeleteBuffers(1, &b);
			}
			it = skinned_gl_cache.erase(it);
		} else {
			++it;
		}
	}

	SharedGL43Context *gl43_ctx = SharedGL43Context::get_singleton();

	for (size_t vi = 0; vi < viewport_targets.size(); vi++) {
		const ViewportTarget &vt = viewport_targets[vi];
		auto st_it = vp_gl_states.find(vt.vp_id);
		if (st_it == vp_gl_states.end()) continue;
		ViewportGLState &vst = st_it->second;

		RasterizerStorageGLES3::Texture *godot_vel_tex = p_rs->texture_owner.getornull(vt.velocity_tex_rid);
		if (!godot_vel_tex || godot_vel_tex->tex_id == 0) continue;
		if (godot_vel_tex->width != vt.vel_w || godot_vel_tex->height != vt.vel_h) continue;

		_ensure_vel_fbo_size(vst, vt.vel_w, vt.vel_h);
		glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)vst.vel_fbo);
		glViewport(0, 0, vt.vel_w, vt.vel_h);
		glFrontFace(GL_CW);
		glDisable(GL_SCISSOR_TEST);
		glDisable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(GL_TRUE);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glClearDepth(1.0);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram((GLuint)vel_program);
		GLint loc_curr_mvp = glGetUniformLocation((GLuint)vel_program, "uCurrMVP");
		GLint loc_prev_mvp = glGetUniformLocation((GLuint)vel_program, "uPrevMVP");
		GLint loc_alb_tex = glGetUniformLocation((GLuint)vel_program, "uAlbedoTex");
		GLint loc_use_scissor = glGetUniformLocation((GLuint)vel_program, "uUseAlphaScissor");
		GLint loc_alpha_thresh = glGetUniformLocation((GLuint)vel_program, "uAlphaThreshold");
		glUniform1i(loc_alb_tex, 0);

		for (size_t i = 0; i < instances.size(); i++) {
			const VelocityInstance &inst = instances[i];
			if (!GL43EffectHelpers::aabb_in_frustum(inst.world_aabb, vt.frustum_planes)) continue;
			uint32_t vao = 0;
			if (inst.is_skinned) {
				const SkinnedSnapshot &sk = skinned_instances[inst.skinned_snap_idx];
				vao = skinned_gl_cache[sk.obj_id].gl_vao;
			} else {
				auto it = mesh_cache.find(inst.mesh_id);
				if (it == mesh_cache.end() || !it->second.gpu_uploaded) continue;
				vao = it->second.gl_vao;
			}
			if (vao == 0) continue;

			float curr_mvp[16], prev_mvp[16];
			GL43EffectHelpers::mat4_mul(vt.curr_vp_mat, inst.curr_model, curr_mvp);
			GL43EffectHelpers::mat4_mul(vt.prev_vp_mat, inst.prev_model, prev_mvp);
			glUniformMatrix4fv(loc_curr_mvp, 1, GL_FALSE, curr_mvp);
			glUniformMatrix4fv(loc_prev_mvp, 1, GL_FALSE, prev_mvp);
			glBindVertexArray((GLuint)vao);

			for (size_t s = 0; s < inst.surfaces.size(); s++) {
				const SurfaceRange &sr = inst.surfaces[s];
				if (sr.vert_count <= 0) continue;

				if (sr.cull_mode == 2) {
					glDisable(GL_CULL_FACE);
				} else {
					glEnable(GL_CULL_FACE);
					bool cull_front = (sr.cull_mode == 1);
					if (inst.flip_facing) cull_front = !cull_front;
					glCullFace(cull_front ? GL_FRONT : GL_BACK);
				}

				bool do_scissor = false;
				if (sr.alpha_scissor && sr.albedo_tex_rid.is_valid()) {
					RasterizerStorageGLES3::Texture *gt = p_rs->texture_owner.getornull(sr.albedo_tex_rid);
					if (gt) gt = gt->get_ptr();
					if (gt && gt->tex_id != 0) {
						glActiveTexture(GL_TEXTURE0);
						glBindTexture(GL_TEXTURE_2D, gt->tex_id);
						glUniform1i(loc_use_scissor, 1);
						glUniform1f(loc_alpha_thresh, sr.alpha_threshold);
						do_scissor = true;
					}
				}
				if (!do_scissor) {
					glUniform1i(loc_use_scissor, 0);
				}

				glDrawArrays(GL_TRIANGLES, sr.first_vert, sr.vert_count);
			}
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		GLuint gx = (GLuint)((vt.vel_w + 7) / 8);
		GLuint gy = (GLuint)((vt.vel_h + 7) / 8);

		glUseProgram((GLuint)resolve_program);
		gl43_ctx->bind_image_texture(0, godot_vel_tex->tex_id, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, (GLuint)vst.vel_raw_tex);
		glUniform1i(glGetUniformLocation((GLuint)resolve_program, "uVelRawTex"), 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, (GLuint)vst.vel_depth_tex);
		glUniform1i(glGetUniformLocation((GLuint)resolve_program, "uDepthTex"), 1);

		glUniformMatrix4fv(glGetUniformLocation((GLuint)resolve_program, "uInvProj"), 1, GL_FALSE, vt.inv_proj_mat);
		glUniformMatrix4fv(glGetUniformLocation((GLuint)resolve_program, "uInvCurrVP"), 1, GL_FALSE, vt.inv_curr_vp_mat);
		glUniformMatrix4fv(glGetUniformLocation((GLuint)resolve_program, "uPrevVP"), 1, GL_FALSE, vt.prev_vp_mat);

		glUniform2f(glGetUniformLocation((GLuint)resolve_program, "uViewportSize"), (float)vt.viewport_w, (float)vt.viewport_h);
		glUniform1f(glGetUniformLocation((GLuint)resolve_program, "uVelScale"), vt.eff_vel_scale);
		glUniform1f(glGetUniformLocation((GLuint)resolve_program, "uMaxVelPx"), vt.max_vel_px);
		glUniform1i(glGetUniformLocation((GLuint)resolve_program, "uCameraMotion"), vt.camera_motion ? 1 : 0);

		gl43_ctx->dispatch_compute(gx, gy, 1);
		gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, godot_vel_tex->tex_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	return true;
}
