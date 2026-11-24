#ifndef GL43_RAYTRACER_H
#define GL43_RAYTRACER_H

#include "core/math/aabb.h"
#include "core/math/transform.h"
#include "drivers/gles3/gl43_bvh_helper.h"
#include "scene/3d/visual_instance.h"
#include "scene/resources/mesh.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

class MeshInstance;
class MultiMeshInstance;

class GL43Raytracer {
public:
	enum SimplificationMode {
		SIMPLIFY_QEM = 0,
		SIMPLIFY_VERTEX_CLUSTERING = 1,
	};

	struct SimplificationConfig {
		bool enabled = false;
		SimplificationMode mode = SIMPLIFY_QEM;
		float ratio = 0.5f;
		int min_triangles = 128;
		float max_error = 0.05f;
		bool preserve_seams = true;
		bool simplify_skinned = false;
		bool simplify_gbuffer = false;

		uint64_t get_signature() const;
	};

	using BVHNode = GL43BVHNode;

	// 8 x vec4 = 128 bytes (std430)
	struct Triangle {
		float p0[3]; int32_t surface_idx;
		float p1[3]; float _pad1;
		float p2[3]; float _pad2;
		float n0[3]; float _pad3;
		float n1[3]; float _pad4;
		float n2[3]; float _pad5;
		float uv01[4];
		float uv2_pad[4];
	};

	// 11 x vec4 = 176 bytes (std430)
	struct Instance {
		float inv_model[16];
		float model[16];
		float aabb_min[3]; int32_t blas_root;
		float aabb_max[3]; int32_t mat_offset;
		int32_t flags[4]; // x = bitmask (1=visible_in_trace, 2=cast_shadows, 4=double_sided_shadow)
	};

	struct MeshBLAS {
		uint32_t mesh_id = 0;
		uint64_t cached_mesh_instance_id = 0;
		AABB local_aabb;
		AABB cached_mesh_aabb;
		uint64_t simpl_signature = 0;

		std::vector<Triangle> triangles;
		std::vector<BVHNode> nodes;
		std::vector<float> gbuf_verts; // 9 floats/vert: pos(3), norm(3), uv(2), surf(1)

		bool has_skin_weights = false;
		int bones_per_vert = 4;
		std::vector<Vector3> bind_pos;
		std::vector<Vector3> bind_norm;
		std::vector<Vector2> bind_uv;
		std::vector<int32_t> bind_bones;
		std::vector<float> bind_weights;
		std::vector<uint32_t> ordered_tri_verts;
		std::vector<int32_t> ordered_tri_surf;

		uint32_t gl_vao = 0;
		uint32_t gl_vbo = 0;
		int32_t vertex_count = 0;
		int32_t blas_node_offset = 0;
		bool gpu_uploaded = false;
	};

	struct SkinnedSnapshot {
		uint64_t obj_id = 0;
		std::vector<Triangle> posed_tris;
		std::vector<BVHNode> refit_nodes;
		std::vector<float> posed_gbuf_verts;
		int32_t vertex_count = 0;
	};

	struct InstanceEntry {
		uint32_t mesh_id = 0;
		bool is_skinned = false;
		int32_t skinned_snap_idx = -1;
		Transform world_xform;
		Transform inv_xform;
		float model_mat[16] = {};
		AABB world_aabb;
		int32_t mat_offset = 0;
		uint32_t layer_mask = 0xFFFFFFFF;
		bool draw_in_gbuf = true;
		bool visible_in_trace = true;
		bool cast_shadows = true;
		bool double_sided_shadow = false;
		bool flip_facing = false;
		uint32_t user_flags = 0;
	};

private:
	struct SkinnedMeshGL {
		uint32_t gl_vao = 0;
		uint32_t gl_vbo = 0;
		int32_t vertex_count = 0;
		uint64_t last_frame_seen = 0;
	};

	struct PendingGLDelete {
		uint32_t vao = 0;
		uint32_t vbo = 0;
	};

	struct TlasCandidate {
		int32_t snap_idx = -1;
		float sort_dist = 0.0f;
		AABB world_aabb;
	};

	std::unordered_map<uint32_t, MeshBLAS> blas_cache;
	std::vector<uint32_t> blas_order;
	std::vector<BVHNode> cached_static_nodes;
	std::vector<Triangle> cached_static_tris;
	std::unordered_map<uint64_t, SkinnedMeshGL> skinned_gl_cache;
	std::vector<PendingGLDelete> gl_buffers_to_delete;

	std::vector<InstanceEntry> instances;
	std::vector<SkinnedSnapshot> skinned_instances;
	std::vector<TlasCandidate> tlas_candidates;
	std::vector<BVHNode> tlas_nodes;
	std::vector<int32_t> tlas_indices;

	bool blas_buffers_dirty = true;
	bool last_frame_had_skinned = false;
	uint64_t frame_counter = 0;

	uint32_t ssbo_blas_nodes = 0;
	uint32_t ssbo_triangles = 0;
	uint32_t ssbo_tlas_nodes = 0;
	uint32_t ssbo_tlas_indices = 0;
	uint32_t ssbo_instances = 0;

	void _simplify_mesh_data(
			const SimplificationConfig &p_cfg,
			std::vector<Vector3> &io_pos,
			std::vector<Vector3> &io_norm,
			std::vector<Vector2> &io_uv,
			std::vector<int32_t> &io_bones,
			std::vector<float> &io_weights,
			bool p_has_skin,
			int p_bpv,
			std::vector<uint32_t> &io_tri_verts,
			std::vector<int32_t> &io_tri_surf) const;

	bool _build_skinned_snapshot(MeshInstance *p_mi, const MeshBLAS &p_blas, SkinnedSnapshot &r_snap, AABB &r_local_aabb) const;

public:
	static const char *get_glsl_common();

	bool init_gl();
	void cleanup_gl();

	void invalidate_blas_cache();
	void rebuild_blas_cache(
			const std::vector<MeshInstance *> &p_meshes,
			const std::vector<Ref<Mesh>> &p_extra_meshes,
			const SimplificationConfig &p_simpl,
			int p_blas_builds_per_frame = 0);

	const MeshBLAS *get_blas(uint32_t p_mesh_id) const;

	static bool passes_distance_cull(
			const AABB &p_world_aabb,
			const Vector3 &p_cam_pos,
			float p_cull_distance,
			float p_cull_min_angle,
			float p_res_scale);

	void begin_frame_instances();

	bool add_mesh_instance(
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
			int32_t *r_out_index = nullptr);

	bool add_multimesh_instance(
			MultiMeshInstance *p_mmi,
			int32_t p_shared_mat_offset,
			const Vector3 &p_cam_pos,
			float p_cull_distance,
			float p_cull_min_angle,
			float p_res_scale,
			bool p_add_to_tlas,
			bool p_draw_in_gbuf,
			uint32_t p_user_flags);

	bool add_custom_mesh_instance(
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
			int32_t *r_out_index = nullptr);

	void build_tlas(int p_max_tlas_instances = 0);

	void sync_gpu_buffers();
	void bind_ssbos(
			uint32_t p_blas_node_binding = 0,
			uint32_t p_tri_binding = 1,
			uint32_t p_tlas_node_binding = 2,
			uint32_t p_tlas_idx_binding = 3,
			uint32_t p_inst_binding = 4) const;
	void unbind_ssbos(
			uint32_t p_blas_node_binding = 0,
			uint32_t p_tri_binding = 1,
			uint32_t p_tlas_node_binding = 2,
			uint32_t p_tlas_idx_binding = 3,
			uint32_t p_inst_binding = 4) const;

	bool get_instance_draw_data(size_t p_index, uint32_t &r_vao, int32_t &r_vertex_count) const;

	bool has_tlas() const { return !tlas_nodes.empty() && !tlas_indices.empty(); }
	bool is_empty() const { return instances.empty(); }
	const std::vector<InstanceEntry> &get_instances() const { return instances; }
	InstanceEntry &get_instance_mutable(size_t p_index) { return instances[p_index]; }
};

#endif // GL43_RAYTRACER_H
