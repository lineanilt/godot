#ifndef GL43_VELOCITY_HANDLER_H
#define GL43_VELOCITY_HANDLER_H

#include "core/math/aabb.h"
#include "core/math/camera_matrix.h"
#include "core/math/vector2.h"
#include "core/rid.h"
#include "core/math/transform.h"
#include "scene/resources/mesh.h"
#include "scene/resources/texture.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

class Camera;
class MeshInstance;
class MultiMeshInstance;
class Node;
class RasterizerStorageGLES3;

class GL43VelocityHandler {
public:
	struct SurfaceRange {
		int32_t first_vert = 0;
		int32_t vert_count = 0;
		int32_t cull_mode = 0; // 0=back, 1=front, 2=disabled
		bool alpha_scissor = false;
		float alpha_threshold = 0.5f;
		RID albedo_tex_rid;
	};

	struct MeshGeom {
		uint32_t mesh_id = 0;
		uint64_t cached_mesh_instance_id = 0;
		AABB local_aabb;
		AABB cached_mesh_aabb;
		std::vector<float> verts; // 5 floats per vertex: x, y, z, u, v
		std::vector<SurfaceRange> surfaces;
		int32_t total_verts = 0;

		bool has_skin_weights = false;
		int bones_per_vert = 4;
		std::vector<Vector3> bind_pos;
		std::vector<Vector2> bind_uv;
		std::vector<int32_t> bind_bones;
		std::vector<float> bind_weights;
		std::vector<uint32_t> tri_verts;

		uint32_t gl_vao = 0;
		uint32_t gl_vbo = 0;
		bool gpu_uploaded = false;
	};

	struct SkinnedMeshGL {
		uint32_t gl_vao = 0;
		uint32_t gl_curr_vbo = 0;
		uint32_t gl_prev_vbo = 0;
		int32_t vertex_count = 0;
		uint64_t last_frame_seen = 0;
	};

	struct SkinnedSnapshot {
		uint64_t obj_id = 0;
		std::vector<float> curr_verts; // 5 floats/vert: x, y, z, u, v
		std::vector<float> prev_verts; // 3 floats/vert: px, py, pz
		int32_t vertex_count = 0;
	};

	struct VelocityInstance {
		uint32_t mesh_id = 0;
		bool is_skinned = false;
		int32_t skinned_snap_idx = -1;
		float curr_model[16] = {};
		float prev_model[16] = {};
		AABB world_aabb;
		bool flip_facing = false;
		std::vector<SurfaceRange> surfaces;
	};

	struct ViewportTarget {
		uint32_t vp_id = 0;
		int viewport_w = 0;
		int viewport_h = 0;
		int vel_w = 0;
		int vel_h = 0;
		float eff_vel_scale = 1.0f;
		float max_vel_px = 0.0f; // <= 0 disables clamping
		bool camera_motion = true;

		RID velocity_tex_rid;
		float curr_vp_mat[16] = {};
		float prev_vp_mat[16] = {};
		float inv_curr_vp_mat[16] = {};
		float inv_proj_mat[16] = {};
		float frustum_planes[6][4] = {};
	};

	struct ViewportGLState {
		uint32_t vel_fbo = 0;
		uint32_t vel_raw_tex = 0;
		uint32_t vel_depth_tex = 0;
		int alloc_vel_w = 0;
		int alloc_vel_h = 0;

		bool has_prev_frame = false;
		float prev_vp_mat[16] = {};
		uint64_t last_seen = 0;
	};

private:
	struct PrevKey {
		uint64_t obj_id = 0;
		uint32_t sub_idx = 0;
		bool operator==(const PrevKey &p_other) const {
			return obj_id == p_other.obj_id && sub_idx == p_other.sub_idx;
		}
	};

	struct PrevKeyHash {
		size_t operator()(const PrevKey &p_key) const {
			uint64_t h = p_key.obj_id * 0x9e3779b97f4a7c15ULL;
			h ^= (uint64_t)p_key.sub_idx * 0xbf58476d1ce4e5b9ULL;
			return (size_t)(h ^ (h >> 30));
		}
	};

	struct PrevObjectState {
		float model_mat[16] = {};
		uint64_t last_seen = 0;
	};

	struct PrevSkinnedState {
		std::vector<float> posed_xyz;
		uint64_t last_seen = 0;
	};

	struct PendingGLDelete {
		uint32_t vao = 0;
		uint32_t vbo_a = 0;
		uint32_t vbo_b = 0;
	};

	std::unordered_map<uint32_t, MeshGeom> mesh_cache;
	std::unordered_map<uint64_t, SkinnedMeshGL> skinned_gl_cache;
	std::unordered_map<PrevKey, PrevObjectState, PrevKeyHash> prev_xforms;
	std::unordered_map<uint64_t, PrevSkinnedState> prev_skinned_poses;
	std::unordered_map<uint32_t, ViewportGLState> vp_gl_states;
	std::vector<PendingGLDelete> gl_buffers_to_delete;
	std::vector<Ref<Texture>> pinned_textures;

	std::vector<VelocityInstance> instances;
	std::vector<SkinnedSnapshot> skinned_instances;
	std::vector<ViewportTarget> viewport_targets;

	bool valid = false;
	bool has_prev_frame = false;
	uint64_t snapshot_counter = 0;
	uint64_t frame_counter = 0;

	uint32_t vel_program = 0;
	uint32_t resolve_program = 0;

	void _rebuild_mesh_cache(const std::vector<MeshInstance *> &p_meshes, const std::vector<Ref<Mesh>> &p_extra_meshes);
	void _ensure_vel_fbo_size(ViewportGLState &r_state, int p_w, int p_h);

public:
	bool init_gl(const char *p_tag);
	void cleanup_gl();

	void begin_collection();
	void add_viewport_target(
			uint32_t p_vp_id,
			Camera *p_cam,
			const Size2 &p_vp_size,
			int p_vel_w,
			int p_vel_h,
			bool p_camera_motion,
			float p_eff_vel_scale,
			float p_max_vel_px,
			RID p_velocity_tex_rid,
			uint64_t p_collect_counter);
	void collect_scene_meshes(
			Node *p_search_root,
			uint32_t p_cull_mask,
			bool p_alpha_scissor_support,
			bool p_object_motion,
			bool p_skinned_meshes);
	void prune_stale_viewports(uint64_t p_collect_counter, bool p_gl_active_on_current_thread);
	void end_collection(bool p_valid);

	bool render_velocities(RasterizerStorageGLES3 *p_rs);
	void reset_history();

	bool is_valid() const { return valid; }
	const std::vector<ViewportTarget> &get_viewport_targets() const { return viewport_targets; }
};

#endif // GL43_VELOCITY_HANDLER_H
