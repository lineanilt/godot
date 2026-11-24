#ifndef GL43_EFFECT_HELPERS_H
#define GL43_EFFECT_HELPERS_H

#include "core/math/aabb.h"
#include "core/math/camera_matrix.h"
#include "core/math/transform.h"
#include "core/math/vector2.h"
#include "core/rid.h"
#include "scene/resources/material.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

class Camera;
class MeshInstance;
class Node;
class Viewport;

namespace GL43EffectHelpers {

// Math & Matrix helpers
void camera_matrix_to_float16(const CameraMatrix &p_cm, float r_out[16]);
void transform_to_float16(const Transform &p_t, float r_out[16]);
void mat4_mul(const float p_a[16], const float p_b[16], float r_out[16]);
void push_affine_row(const Transform &p_inv, const Vector3 &p_scale, const Vector3 &p_offset, int p_row, float r_out[4]);

void extract_frustum_planes(const CameraMatrix &p_vp, float r_planes[6][4]);
void extract_frustum_planes(const float p_vp[16], float r_planes[6][4]);
bool aabb_in_frustum(const AABB &p_box, const float p_planes[6][4]);
bool sphere_in_frustum(const Vector3 &p_center, float p_radius, const float p_planes[6][4]);
bool aabb_same(const AABB &p_a, const AABB &p_b);

int32_t float_to_int_bits(float p_f);
uint64_t hash_combine_u64(uint64_t p_h, uint64_t p_v);
uint64_t hash_float_u64(uint64_t p_h, float p_f);
bool is_srgb_gl_format(int32_t p_fmt);

// OpenGL 4.3 Shader / SSBO / Texture helpers
uint32_t compile_shader(uint32_t p_type, const char *p_src, const char *p_tag);
uint32_t link_vf_program(const char *p_vs_src, const char *p_fs_src, const char *p_tag, const char *p_label);
uint32_t link_compute_program(const char *p_cs_src, const char *p_tag, const char *p_label);
void upload_ssbo(uint32_t p_buf, const void *p_data, size_t p_bytes, size_t p_dummy_size, uint32_t p_usage = 0x88E8);
void ensure_rgba16f_texture(uint32_t &r_tex, int p_w, int p_h, int p_filter);
void ensure_depth24_texture(uint32_t &r_tex, int p_w, int p_h);

// Scene / Viewport / Camera discovery helpers
bool viewport_renders_effects(const Viewport *p_vp);
void find_active_viewports(const Node *p_context_node, Viewport *p_my_vp, std::vector<std::pair<Viewport *, Camera *>> &r_out);
bool find_primary_camera_and_viewport(const Node *p_context_node, Viewport *p_my_vp, Viewport *&r_vp, Camera *&r_cam);
void compute_camera_matrices(Camera *p_cam, const Size2 &p_vp_size, CameraMatrix &r_view, CameraMatrix &r_proj, CameraMatrix &r_vp, CameraMatrix &r_inv_proj, CameraMatrix &r_inv_vp);

// 3D Scene Render Hooks (executed inside RasterizerSceneGLES3::render_scene)
enum SceneHookStage {
	HOOK_PRE_SCREEN_TEXTURE = 0,
	HOOK_PRE_POST_PROCESS = 1,
};

struct SceneHookContext {
	RID color_tex_rid;
	uint32_t color_tex_id = 0;
	uint32_t depth_tex_id = 0;
	int width = 0;
	int height = 0;
	bool hdr = true;
	bool keep_3d_linear = false;
	float inv_proj_mat[16] = {};
	uint32_t gl43_fbo = 0;
	uint32_t gl43_vao = 0;
};

const char *get_fullscreen_vert_glsl();
void register_scene_hook(void *p_owner, SceneHookStage p_stage, int p_priority, std::function<void(const SceneHookContext &)> p_callback);
void unregister_scene_hooks(void *p_owner);
bool has_scene_hooks(SceneHookStage p_stage);
bool invoke_scene_hooks(SceneHookStage p_stage, const SceneHookContext &p_ctx);
uint32_t copy_scene_color_to_scratch(const SceneHookContext &p_ctx);
void shutdown_scene_hooks();

} // namespace GL43EffectHelpers

struct GL43CanvasOverlay {
	RID vp_rid;
	Ref<ShaderMaterial> material;
	RID canvas;
	RID canvas_item;

	void ensure_attached(Viewport *p_vp, const Ref<Shader> &p_shader, int p_layer, int p_sublayer, bool p_attach);
	void detach();
	void destroy();
};

namespace GL43SkinningHelper {

bool compute_skin_transforms(MeshInstance *p_mi, std::vector<Transform> &r_xforms);

bool pose_mesh_positions(
		MeshInstance *p_mi,
		const std::vector<Vector3> &p_bind_pos,
		const std::vector<int32_t> &p_bind_bones,
		const std::vector<float> &p_bind_weights,
		int p_bones_per_vert,
		std::vector<Vector3> &r_posed_pos,
		AABB *r_local_aabb = nullptr);

bool pose_mesh_positions_and_normals(
		MeshInstance *p_mi,
		const std::vector<Vector3> &p_bind_pos,
		const std::vector<Vector3> &p_bind_norm,
		const std::vector<int32_t> &p_bind_bones,
		const std::vector<float> &p_bind_weights,
		int p_bones_per_vert,
		std::vector<Vector3> &r_posed_pos,
		std::vector<Vector3> &r_posed_norm);

void pack_pos_uv_triangles(
		const std::vector<Vector3> &p_posed_pos,
		const std::vector<Vector2> &p_bind_uv,
		const std::vector<uint32_t> &p_tri_verts,
		std::vector<float> &r_curr_verts,
		std::vector<float> *r_curr_xyz = nullptr);

} // namespace GL43SkinningHelper

#endif // GL43_EFFECT_HELPERS_H
