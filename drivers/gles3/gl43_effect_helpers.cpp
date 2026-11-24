#include "gl43_effect_helpers.h"

#include "core/engine.h"
#include "core/math/math_funcs.h"
#include "core/os/mutex.h"
#include "core/print_string.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"
#include "scene/3d/camera.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/gui/viewport_container.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/resources/skin.h"
#include "scene/resources/world.h"
#include "servers/visual_server.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/plugins/spatial_editor_plugin.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
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
#ifndef GL_FRAMEBUFFER_SRGB
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#endif
#ifndef GL_READ_FRAMEBUFFER
#define GL_READ_FRAMEBUFFER 0x8CA8
#endif
#ifndef GL_DRAW_FRAMEBUFFER
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#endif

namespace GL43EffectHelpers {

namespace {

struct SceneHookEntry {
	void *owner = nullptr;
	SceneHookStage stage = HOOK_PRE_SCREEN_TEXTURE;
	int priority = 0;
	std::function<void(const SceneHookContext &)> callback;
};

static Mutex s_scene_hook_mutex;
static std::vector<SceneHookEntry> s_scene_hooks;

static uint32_t s_scene_fbo = 0;
static uint32_t s_scene_vao = 0;
static uint32_t s_copy_read_fbo = 0;
static uint32_t s_copy_draw_fbo = 0;
static uint32_t s_scratch_tex = 0;
static int s_scratch_w = 0;
static int s_scratch_h = 0;

static const char *GL43_FULLSCREEN_VERT_GLSL = R"GLSL(
#version 430 core
out vec2 vUv;
void main() {
	vec2 pos = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
	vUv = pos * 0.5 + 0.5;
	gl_Position = vec4(pos, 0.0, 1.0);
}
)GLSL";

} // namespace

void camera_matrix_to_float16(const CameraMatrix &p_cm, float r_out[16]) {
	for (int c = 0; c < 4; c++) {
		for (int r = 0; r < 4; r++) {
			r_out[c * 4 + r] = (float)p_cm.matrix[c][r];
		}
	}
}

void transform_to_float16(const Transform &p_t, float r_out[16]) {
	r_out[0] = (float)p_t.basis.elements[0][0];  r_out[1] = (float)p_t.basis.elements[1][0];  r_out[2] = (float)p_t.basis.elements[2][0];  r_out[3] = 0.0f;
	r_out[4] = (float)p_t.basis.elements[0][1];  r_out[5] = (float)p_t.basis.elements[1][1];  r_out[6] = (float)p_t.basis.elements[2][1];  r_out[7] = 0.0f;
	r_out[8] = (float)p_t.basis.elements[0][2];  r_out[9] = (float)p_t.basis.elements[1][2];  r_out[10] = (float)p_t.basis.elements[2][2]; r_out[11] = 0.0f;
	r_out[12] = (float)p_t.origin.x;             r_out[13] = (float)p_t.origin.y;             r_out[14] = (float)p_t.origin.z;             r_out[15] = 1.0f;
}

void mat4_mul(const float p_a[16], const float p_b[16], float r_out[16]) {
	for (int c = 0; c < 4; c++) {
		for (int r = 0; r < 4; r++) {
			r_out[c * 4 + r] =
					p_a[0 * 4 + r] * p_b[c * 4 + 0] +
					p_a[1 * 4 + r] * p_b[c * 4 + 1] +
					p_a[2 * 4 + r] * p_b[c * 4 + 2] +
					p_a[3 * 4 + r] * p_b[c * 4 + 3];
		}
	}
}

void push_affine_row(const Transform &p_inv, const Vector3 &p_scale, const Vector3 &p_offset, int p_row, float r_out[4]) {
	float s = p_scale[p_row];
	r_out[0] = (float)p_inv.basis.elements[p_row][0] * s;
	r_out[1] = (float)p_inv.basis.elements[p_row][1] * s;
	r_out[2] = (float)p_inv.basis.elements[p_row][2] * s;
	r_out[3] = (float)p_inv.origin[p_row] * s + p_offset[p_row];
}

void extract_frustum_planes(const CameraMatrix &p_vp, float r_planes[6][4]) {
	for (int i = 0; i < 4; i++) {
		r_planes[0][i] = p_vp.matrix[i][3] + p_vp.matrix[i][0];
		r_planes[1][i] = p_vp.matrix[i][3] - p_vp.matrix[i][0];
		r_planes[2][i] = p_vp.matrix[i][3] + p_vp.matrix[i][1];
		r_planes[3][i] = p_vp.matrix[i][3] - p_vp.matrix[i][1];
		r_planes[4][i] = p_vp.matrix[i][3] + p_vp.matrix[i][2];
		r_planes[5][i] = p_vp.matrix[i][3] - p_vp.matrix[i][2];
	}
	for (int p = 0; p < 6; p++) {
		float len = Math::sqrt(r_planes[p][0] * r_planes[p][0] + r_planes[p][1] * r_planes[p][1] + r_planes[p][2] * r_planes[p][2]);
		if (len > 1e-6f) {
			float inv = 1.0f / len;
			r_planes[p][0] *= inv;
			r_planes[p][1] *= inv;
			r_planes[p][2] *= inv;
			r_planes[p][3] *= inv;
		}
	}
}

void extract_frustum_planes(const float p_vp[16], float r_planes[6][4]) {
	for (int i = 0; i < 4; i++) {
		r_planes[0][i] = p_vp[i * 4 + 3] + p_vp[i * 4 + 0];
		r_planes[1][i] = p_vp[i * 4 + 3] - p_vp[i * 4 + 0];
		r_planes[2][i] = p_vp[i * 4 + 3] + p_vp[i * 4 + 1];
		r_planes[3][i] = p_vp[i * 4 + 3] - p_vp[i * 4 + 1];
		r_planes[4][i] = p_vp[i * 4 + 3] + p_vp[i * 4 + 2];
		r_planes[5][i] = p_vp[i * 4 + 3] - p_vp[i * 4 + 2];
	}
	for (int p = 0; p < 6; p++) {
		float len = Math::sqrt(r_planes[p][0] * r_planes[p][0] + r_planes[p][1] * r_planes[p][1] + r_planes[p][2] * r_planes[p][2]);
		if (len > 1e-6f) {
			float inv = 1.0f / len;
			r_planes[p][0] *= inv;
			r_planes[p][1] *= inv;
			r_planes[p][2] *= inv;
			r_planes[p][3] *= inv;
		}
	}
}

bool aabb_in_frustum(const AABB &p_box, const float p_planes[6][4]) {
	Vector3 bmin = p_box.position - Vector3(0.15f, 0.15f, 0.15f);
	Vector3 bmax = p_box.position + p_box.size + Vector3(0.15f, 0.15f, 0.15f);
	for (int p = 0; p < 6; p++) {
		float px = (p_planes[p][0] >= 0.0f) ? bmax.x : bmin.x;
		float py = (p_planes[p][1] >= 0.0f) ? bmax.y : bmin.y;
		float pz = (p_planes[p][2] >= 0.0f) ? bmax.z : bmin.z;
		if (p_planes[p][0] * px + p_planes[p][1] * py + p_planes[p][2] * pz + p_planes[p][3] < 0.0f) {
			return false;
		}
	}
	return true;
}

bool sphere_in_frustum(const Vector3 &p_center, float p_radius, const float p_planes[6][4]) {
	for (int p = 0; p < 6; p++) {
		float d = p_planes[p][0] * p_center.x + p_planes[p][1] * p_center.y + p_planes[p][2] * p_center.z + p_planes[p][3];
		if (d < -p_radius) {
			return false;
		}
	}
	return true;
}

bool aabb_same(const AABB &p_a, const AABB &p_b) {
	auto eqf = [](real_t x, real_t y) {
		if (Math::is_nan(x) && Math::is_nan(y)) return true;
		return x == y;
	};
	return eqf(p_a.position.x, p_b.position.x) && eqf(p_a.position.y, p_b.position.y) && eqf(p_a.position.z, p_b.position.z) &&
		   eqf(p_a.size.x, p_b.size.x) && eqf(p_a.size.y, p_b.size.y) && eqf(p_a.size.z, p_b.size.z);
}

int32_t float_to_int_bits(float p_f) {
	int32_t i = 0;
	std::memcpy(&i, &p_f, sizeof(float));
	return i;
}

uint64_t hash_combine_u64(uint64_t p_h, uint64_t p_v) {
	p_h ^= p_v + 0x9e3779b97f4a7c15ULL + (p_h << 6) + (p_h >> 2);
	return p_h;
}

uint64_t hash_float_u64(uint64_t p_h, float p_f) {
	uint32_t bits = 0;
	std::memcpy(&bits, &p_f, sizeof(float));
	return hash_combine_u64(p_h, (uint64_t)bits);
}

bool is_srgb_gl_format(int32_t p_fmt) {
	if (p_fmt >= 0x8C40 && p_fmt <= 0x8C4F) return true;
	if (p_fmt == 0x8E8D) return true;
	return false;
}

uint32_t compile_shader(uint32_t p_type, const char *p_src, const char *p_tag) {
	GLuint s = glCreateShader((GLenum)p_type);
	glShaderSource(s, 1, &p_src, nullptr);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[2048];
		glGetShaderInfoLog(s, sizeof(log), nullptr, log);
		ERR_PRINT(String(p_tag) + " shader compile error:\n" + log);
		glDeleteShader(s);
		return 0;
	}
	return (uint32_t)s;
}

uint32_t link_vf_program(const char *p_vs_src, const char *p_fs_src, const char *p_tag, const char *p_label) {
	GLuint vs = compile_shader(GL_VERTEX_SHADER, p_vs_src, p_tag);
	GLuint fs = compile_shader(GL_FRAGMENT_SHADER, p_fs_src, p_tag);
	if (!vs || !fs) {
		if (vs) glDeleteShader(vs);
		if (fs) glDeleteShader(fs);
		return 0;
	}
	GLuint p = glCreateProgram();
	glAttachShader(p, vs);
	glAttachShader(p, fs);
	glLinkProgram(p);
	glDeleteShader(vs);
	glDeleteShader(fs);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[2048];
		glGetProgramInfoLog(p, sizeof(log), nullptr, log);
		ERR_PRINT(String(p_tag) + " " + p_label + " link error:\n" + log);
		glDeleteProgram(p);
		return 0;
	}
	return (uint32_t)p;
}

uint32_t link_compute_program(const char *p_cs_src, const char *p_tag, const char *p_label) {
#ifndef GL_COMPUTE_SHADER
#define GL_COMPUTE_SHADER 0x91B9
#endif
	GLuint cs = compile_shader(GL_COMPUTE_SHADER, p_cs_src, p_tag);
	if (!cs) {
		return 0;
	}
	GLuint p = glCreateProgram();
	glAttachShader(p, cs);
	glLinkProgram(p);
	glDeleteShader(cs);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok) {
		char log[2048];
		glGetProgramInfoLog(p, sizeof(log), nullptr, log);
		ERR_PRINT(String(p_tag) + " " + p_label + " link error:\n" + log);
		glDeleteProgram(p);
		return 0;
	}
	return (uint32_t)p;
}

void upload_ssbo(uint32_t p_buf, const void *p_data, size_t p_bytes, size_t p_dummy_size, uint32_t p_usage) {
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, (GLuint)p_buf);
	if (p_bytes > 0) {
		glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)p_bytes, p_data, (GLenum)p_usage);
	} else {
		uint8_t zero[256] = {};
		size_t sz = p_dummy_size < sizeof(zero) ? p_dummy_size : sizeof(zero);
		glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)sz, zero, (GLenum)p_usage);
	}
}

void ensure_rgba16f_texture(uint32_t &r_tex, int p_w, int p_h, int p_filter) {
	if (r_tex == 0) {
		GLuint t = 0;
		glGenTextures(1, &t);
		r_tex = (uint32_t)t;
	}
	glBindTexture(GL_TEXTURE_2D, (GLuint)r_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, p_w, p_h, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, p_filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, p_filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void ensure_depth24_texture(uint32_t &r_tex, int p_w, int p_h) {
	if (r_tex == 0) {
		GLuint t = 0;
		glGenTextures(1, &t);
		r_tex = (uint32_t)t;
	}
	glBindTexture(GL_TEXTURE_2D, (GLuint)r_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, p_w, p_h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
}

bool viewport_renders_effects(const Viewport *p_vp) {
	if (!p_vp || !p_vp->is_inside_tree() || p_vp->is_3d_disabled() || p_vp->get_render_mode() == Viewport::RENDER_MODE_DEPTH_TEXTURE_ONLY) {
		return false;
	}
	if (p_vp->has_meta("render_effects")) {
		return bool(p_vp->get_meta("render_effects"));
	}
	return true;
}

void find_active_viewports(const Node *p_context_node, Viewport *p_my_vp, std::vector<std::pair<Viewport *, Camera *>> &r_out) {
	r_out.clear();
	if (!p_context_node || !p_my_vp) return;
	SceneTree *tree = p_context_node->get_tree();
	if (!tree || !tree->get_root()) return;
	Ref<World> target_world = p_my_vp->find_world();
	if (target_world.is_null()) return;
	bool in_editor = Engine::get_singleton()->is_editor_hint();

	Node *edited_scene = nullptr;
	Viewport *scene_root_vp = nullptr;
#ifdef TOOLS_ENABLED
	if (in_editor && EditorNode::get_singleton()) {
		edited_scene = EditorNode::get_singleton()->get_edited_scene();
		scene_root_vp = EditorNode::get_singleton()->get_scene_root();
	}
#endif

	struct VpFinder {
		static void visit(Node *p_node, Viewport *scene_root, Node *edited_root, const Ref<World> &p_world, bool editor, std::vector<std::pair<Viewport *, Camera *>> &out) {
			if (Viewport *v = Object::cast_to<Viewport>(p_node)) {
				if (v->find_world() == p_world && v->get_camera() != nullptr && viewport_renders_effects(v)) {
					Size2 sz = v->get_size();
					if (editor) {
						ViewportContainer *vc = Object::cast_to<ViewportContainer>(v->get_parent());
						bool is_editor_vp = false;
						bool is_scene_sub_vp = false;
#ifdef TOOLS_ENABLED
						for (Node *p = vc ? vc->get_parent() : nullptr; p; p = p->get_parent()) {
							if (Object::cast_to<SpatialEditorViewport>(p)) {
								is_editor_vp = true;
								break;
							}
						}
						if (!is_editor_vp && v != scene_root && edited_root && (v == edited_root || edited_root->is_a_parent_of(v))) {
							is_scene_sub_vp = true;
						}
#endif
						if ((is_editor_vp && vc && vc->is_visible_in_tree() && sz.x >= 16 && sz.y >= 16) ||
								(is_scene_sub_vp && v->get_update_mode() != Viewport::UPDATE_DISABLED && sz.x >= 4 && sz.y >= 4)) {
							out.push_back({ v, v->get_camera() });
						}
					} else {
						if (v->get_update_mode() != Viewport::UPDATE_DISABLED && sz.x >= 4 && sz.y >= 4) {
							out.push_back({ v, v->get_camera() });
						}
					}
				}
			}
			for (int i = 0; i < p_node->get_child_count(); i++) {
				visit(p_node->get_child(i), scene_root, edited_root, p_world, editor, out);
			}
		}
	};

	VpFinder::visit(tree->get_root(), scene_root_vp, edited_scene, target_world, in_editor, r_out);
}

bool find_primary_camera_and_viewport(const Node *p_context_node, Viewport *p_my_vp, Viewport *&r_vp, Camera *&r_cam) {
	r_vp = nullptr;
	r_cam = nullptr;
	if (!p_context_node || !p_my_vp) return false;

	if (!Engine::get_singleton()->is_editor_hint()) {
		if (!viewport_renders_effects(p_my_vp)) {
			return false;
		}
		r_vp = p_my_vp;
		r_cam = p_my_vp->get_camera();
		return r_cam != nullptr;
	}

	SceneTree *tree = p_context_node->get_tree();
	if (!tree || !tree->get_root()) return false;

	Ref<World> target_world = p_my_vp->find_world();
	if (target_world.is_null()) return false;

	Viewport *best_vp = nullptr;
	float best_score = -1.0f;

	struct EdFinder {
		static void visit(Node *n, Viewport *skip_vp, const Ref<World> &w, Viewport *&best, float &best_score) {
			if (Viewport *v = Object::cast_to<Viewport>(n)) {
				if (v != skip_vp && v->find_world() == w && v->get_camera() != nullptr && viewport_renders_effects(v)) {
					ViewportContainer *vc = Object::cast_to<ViewportContainer>(v->get_parent());
					bool is_editor_vp = false;
#ifdef TOOLS_ENABLED
					for (Node *p = vc ? vc->get_parent() : nullptr; p; p = p->get_parent()) {
						if (Object::cast_to<SpatialEditorViewport>(p)) {
							is_editor_vp = true;
							break;
						}
					}
#endif
					if (is_editor_vp && vc && vc->is_visible_in_tree()) {
						Size2 sz = v->get_size();
						if (sz.x >= 16 && sz.y >= 16) {
							float score = sz.x * sz.y + (vc->has_focus() ? 1e9f : 0.0f);
							if (score > best_score) {
								best_score = score;
								best = v;
							}
						}
					}
				}
			}
			for (int i = 0; i < n->get_child_count(); i++) {
				visit(n->get_child(i), skip_vp, w, best, best_score);
			}
		}
	};

	EdFinder::visit(tree->get_root(), p_my_vp, target_world, best_vp, best_score);
	if (best_vp) {
		r_vp = best_vp;
		r_cam = best_vp->get_camera();
		return r_cam != nullptr;
	}
	return false;
}

void compute_camera_matrices(Camera *p_cam, const Size2 &p_vp_size, CameraMatrix &r_view, CameraMatrix &r_proj, CameraMatrix &r_vp, CameraMatrix &r_inv_proj, CameraMatrix &r_inv_vp) {
	Transform xf = p_cam->get_camera_transform();
	bool is_ortho = (p_cam->get_projection() == Camera::PROJECTION_ORTHOGONAL);
	float aspect = (float)(p_vp_size.x / p_vp_size.y);
	bool flip_fov = (p_cam->get_keep_aspect_mode() == Camera::KEEP_WIDTH);
	if (is_ortho) {
		r_proj.set_orthogonal(p_cam->get_size(), aspect, p_cam->get_znear(), p_cam->get_zfar(), flip_fov);
	} else {
		r_proj.set_perspective(p_cam->get_fov(), aspect, p_cam->get_znear(), p_cam->get_zfar(), flip_fov);
	}
	r_view = CameraMatrix(xf.affine_inverse());
	r_vp = r_proj * r_view;
	r_inv_proj = r_proj.inverse();
	r_inv_vp = r_vp.inverse();
}

const char *get_fullscreen_vert_glsl() {
	return GL43_FULLSCREEN_VERT_GLSL;
}

void register_scene_hook(void *p_owner, SceneHookStage p_stage, int p_priority, std::function<void(const SceneHookContext &)> p_callback) {
	if (!p_owner || !p_callback) {
		return;
	}
	MutexLock lock(s_scene_hook_mutex);
	for (size_t i = 0; i < s_scene_hooks.size(); i++) {
		if (s_scene_hooks[i].owner == p_owner && s_scene_hooks[i].stage == p_stage) {
			s_scene_hooks[i].priority = p_priority;
			s_scene_hooks[i].callback = std::move(p_callback);
			return;
		}
	}
	s_scene_hooks.push_back({ p_owner, p_stage, p_priority, std::move(p_callback) });
}

void unregister_scene_hooks(void *p_owner) {
	if (!p_owner) {
		return;
	}
	MutexLock lock(s_scene_hook_mutex);
	for (size_t i = 0; i < s_scene_hooks.size();) {
		if (s_scene_hooks[i].owner == p_owner) {
			s_scene_hooks.erase(s_scene_hooks.begin() + (ptrdiff_t)i);
		} else {
			i++;
		}
	}
}

bool has_scene_hooks(SceneHookStage p_stage) {
	MutexLock lock(s_scene_hook_mutex);
	for (size_t i = 0; i < s_scene_hooks.size(); i++) {
		if (s_scene_hooks[i].stage == p_stage) {
			return true;
		}
	}
	return false;
}

bool invoke_scene_hooks(SceneHookStage p_stage, const SceneHookContext &p_ctx) {
	if (!p_ctx.color_tex_id || p_ctx.width <= 0 || p_ctx.height <= 0) {
		return false;
	}

	std::vector<SceneHookEntry> to_run;
	{
		MutexLock lock(s_scene_hook_mutex);
		for (size_t i = 0; i < s_scene_hooks.size(); i++) {
			if (s_scene_hooks[i].stage == p_stage) {
				to_run.push_back(s_scene_hooks[i]);
			}
		}
	}
	if (to_run.empty()) {
		return false;
	}

	std::stable_sort(to_run.begin(), to_run.end(), [](const SceneHookEntry &a, const SceneHookEntry &b) {
		return a.priority < b.priority;
	});

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return false;
	}

	if (s_scene_fbo == 0) {
		GLuint f = 0;
		glGenFramebuffers(1, &f);
		s_scene_fbo = (uint32_t)f;
	}
	if (s_scene_vao == 0) {
		GLuint v = 0;
		glGenVertexArrays(1, &v);
		s_scene_vao = (uint32_t)v;
	}

	SceneHookContext ctx = p_ctx;
	ctx.gl43_fbo = s_scene_fbo;
	ctx.gl43_vao = s_scene_vao;

	glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)s_scene_fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, (GLuint)ctx.color_tex_id, 0);
	GLenum db = GL_COLOR_ATTACHMENT0;
	glDrawBuffers(1, &db);

	glViewport(0, 0, ctx.width, ctx.height);
	glDisable(GL_DEPTH_TEST);
	glDepthMask(GL_FALSE);
	glDisable(GL_CULL_FACE);
	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_FRAMEBUFFER_SRGB);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glBindVertexArray((GLuint)s_scene_vao);

	for (size_t i = 0; i < to_run.size(); i++) {
		glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)s_scene_fbo);
		glViewport(0, 0, ctx.width, ctx.height);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		glDisable(GL_SCISSOR_TEST);
		glDisable(GL_FRAMEBUFFER_SRGB);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glBindVertexArray((GLuint)s_scene_vao);

		to_run[i].callback(ctx);
	}

	for (int u = 0; u < 6; u++) {
		glActiveTexture(GL_TEXTURE0 + u);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	glActiveTexture(GL_TEXTURE0);
	glDisable(GL_BLEND);
	glBindVertexArray(0);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	return true;
}

uint32_t copy_scene_color_to_scratch(const SceneHookContext &p_ctx) {
	if (!p_ctx.color_tex_id || p_ctx.width <= 0 || p_ctx.height <= 0) {
		return 0;
	}
	if (s_copy_read_fbo == 0) {
		GLuint f = 0;
		glGenFramebuffers(1, &f);
		s_copy_read_fbo = (uint32_t)f;
	}
	if (s_copy_draw_fbo == 0) {
		GLuint f = 0;
		glGenFramebuffers(1, &f);
		s_copy_draw_fbo = (uint32_t)f;
	}
	if (s_scratch_tex == 0 || s_scratch_w != p_ctx.width || s_scratch_h != p_ctx.height) {
		s_scratch_w = p_ctx.width;
		s_scratch_h = p_ctx.height;
		glActiveTexture(GL_TEXTURE0);
		ensure_rgba16f_texture(s_scratch_tex, p_ctx.width, p_ctx.height, GL_LINEAR);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glDisable(GL_SCISSOR_TEST);
	glDisable(GL_FRAMEBUFFER_SRGB);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)s_copy_read_fbo);
	glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, (GLuint)p_ctx.color_tex_id, 0);
	glReadBuffer(GL_COLOR_ATTACHMENT0);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)s_copy_draw_fbo);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, (GLuint)s_scratch_tex, 0);
	GLenum db = GL_COLOR_ATTACHMENT0;
	glDrawBuffers(1, &db);

	glBlitFramebuffer(0, 0, p_ctx.width, p_ctx.height, 0, 0, p_ctx.width, p_ctx.height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)p_ctx.gl43_fbo);
	glViewport(0, 0, p_ctx.width, p_ctx.height);
	return s_scratch_tex;
}

void shutdown_scene_hooks() {
	{
		MutexLock lock(s_scene_hook_mutex);
		s_scene_hooks.clear();
	}
	SharedGL43Context *ctx = SharedGL43Context::get_singleton();
	if (!ctx || !ctx->is_available()) {
		return;
	}
	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}
	if (s_scene_fbo) { GLuint f = (GLuint)s_scene_fbo; glDeleteFramebuffers(1, &f); s_scene_fbo = 0; }
	if (s_scene_vao) { GLuint v = (GLuint)s_scene_vao; glDeleteVertexArrays(1, &v); s_scene_vao = 0; }
	if (s_copy_read_fbo) { GLuint f = (GLuint)s_copy_read_fbo; glDeleteFramebuffers(1, &f); s_copy_read_fbo = 0; }
	if (s_copy_draw_fbo) { GLuint f = (GLuint)s_copy_draw_fbo; glDeleteFramebuffers(1, &f); s_copy_draw_fbo = 0; }
	if (s_scratch_tex) { GLuint t = (GLuint)s_scratch_tex; glDeleteTextures(1, &t); s_scratch_tex = 0; }
	s_scratch_w = 0;
	s_scratch_h = 0;
}

} // namespace GL43EffectHelpers

void GL43CanvasOverlay::ensure_attached(Viewport *p_vp, const Ref<Shader> &p_shader, int p_layer, int p_sublayer, bool p_attach) {
	if (!p_vp) {
		return;
	}
	RID new_vp_rid = p_vp->get_viewport_rid();
	VisualServer *vs = VisualServer::get_singleton();

	if (material.is_null()) {
		material.instance();
		material->set_shader(p_shader);
	}

	if (vs && !canvas.is_valid()) {
		canvas = vs->canvas_create();
		canvas_item = vs->canvas_item_create();
		vs->canvas_item_set_parent(canvas_item, canvas);
		vs->canvas_item_set_material(canvas_item, material->get_rid());
		vs->canvas_item_add_rect(canvas_item, Rect2(-100000.0f, -100000.0f, 200000.0f, 200000.0f), Color(1.0f, 1.0f, 1.0f, 1.0f));
	}

	if (!vs || !canvas.is_valid()) {
		return;
	}

	if (p_attach) {
		if (vp_rid != new_vp_rid) {
			if (vp_rid.is_valid()) {
				vs->viewport_remove_canvas(vp_rid, canvas);
			}
			vp_rid = new_vp_rid;
			vs->viewport_attach_canvas(vp_rid, canvas);
			vs->viewport_set_canvas_stacking(vp_rid, canvas, p_layer, p_sublayer);
		}
	} else if (vp_rid.is_valid()) {
		vs->viewport_remove_canvas(vp_rid, canvas);
		vp_rid = RID();
	}
}

void GL43CanvasOverlay::detach() {
	VisualServer *vs = VisualServer::get_singleton();
	if (vs && vp_rid.is_valid() && canvas.is_valid()) {
		vs->viewport_remove_canvas(vp_rid, canvas);
	}
	vp_rid = RID();
}

void GL43CanvasOverlay::destroy() {
	detach();
	VisualServer *vs = VisualServer::get_singleton();
	if (vs) {
		if (canvas_item.is_valid()) {
			vs->free(canvas_item);
			canvas_item = RID();
		}
		if (canvas.is_valid()) {
			vs->free(canvas);
			canvas = RID();
		}
	}
	material.unref();
}

namespace GL43SkinningHelper {

bool compute_skin_transforms(MeshInstance *p_mi, std::vector<Transform> &r_xforms) {
	if (!p_mi || p_mi->get_skeleton_path().is_empty() || !p_mi->has_node(p_mi->get_skeleton_path())) {
		return false;
	}
	Skeleton *skel = Object::cast_to<Skeleton>(p_mi->get_node(p_mi->get_skeleton_path()));
	if (!skel || skel->get_bone_count() <= 0) {
		return false;
	}

	int bone_count = skel->get_bone_count();
	Ref<Skin> skin = p_mi->get_skin();

	if (skin.is_valid() && skin->get_bind_count() > 0) {
		int bc = skin->get_bind_count();
		r_xforms.assign(bc, Transform());
		for (int i = 0; i < bc; i++) {
			int bidx = skin->get_bind_bone(i);
			if (bidx < 0) {
				StringName bname = skin->get_bind_name(i);
				if (bname != StringName()) bidx = skel->find_bone(bname);
			}
			if (bidx >= 0 && bidx < bone_count) {
				r_xforms[i] = skel->get_bone_global_pose(bidx) * skin->get_bind_pose(i);
			}
		}
	} else {
		std::vector<Transform> rest_global(bone_count);
		std::vector<uint8_t> computed(bone_count, 0);
		struct RestHelper {
			static Transform get_rest(Skeleton *s, int b, std::vector<Transform> &rg, std::vector<uint8_t> &done) {
				if (done[b]) return rg[b];
				int p = s->get_bone_parent(b);
				Transform r = s->get_bone_rest(b);
				rg[b] = (p >= 0 && p < (int)rg.size()) ? (get_rest(s, p, rg, done) * r) : r;
				done[b] = 1;
				return rg[b];
			}
		};
		r_xforms.resize(bone_count);
		for (int b = 0; b < bone_count; b++) {
			Transform inv_rest = RestHelper::get_rest(skel, b, rest_global, computed).affine_inverse();
			r_xforms[b] = skel->get_bone_global_pose(b) * inv_rest;
		}
	}
	return true;
}

bool pose_mesh_positions(
		MeshInstance *p_mi,
		const std::vector<Vector3> &p_bind_pos,
		const std::vector<int32_t> &p_bind_bones,
		const std::vector<float> &p_bind_weights,
		int p_bones_per_vert,
		std::vector<Vector3> &r_posed_pos,
		AABB *r_local_aabb) {

	std::vector<Transform> skin_xforms;
	if (!compute_skin_transforms(p_mi, skin_xforms)) {
		return false;
	}

	int num_verts = (int)p_bind_pos.size();
	int num_skins = (int)skin_xforms.size();
	r_posed_pos.resize(num_verts);
	bool first = true;
	AABB bounds;

	for (int v = 0; v < num_verts; v++) {
		const Vector3 &bp = p_bind_pos[v];
		float w_sum = 0.0f;
		float m[3][4] = {};
		int base = v * p_bones_per_vert;

		for (int k = 0; k < p_bones_per_vert; k++) {
			float w = p_bind_weights[base + k];
			int bi = p_bind_bones[base + k];
			if (w <= 1e-6f || bi < 0 || bi >= num_skins) continue;
			const Transform &xf = skin_xforms[bi];
			for (int r = 0; r < 3; r++) {
				m[r][0] += (float)xf.basis.elements[r][0] * w;
				m[r][1] += (float)xf.basis.elements[r][1] * w;
				m[r][2] += (float)xf.basis.elements[r][2] * w;
				m[r][3] += (float)xf.origin[r] * w;
			}
			w_sum += w;
		}

		Vector3 p;
		if (w_sum <= 1e-5f) {
			p = bp;
		} else {
			float inv_w = (std::abs(w_sum - 1.0f) > 1e-4f) ? (1.0f / w_sum) : 1.0f;
			p = Vector3(
					(m[0][0] * bp.x + m[0][1] * bp.y + m[0][2] * bp.z + m[0][3]) * inv_w,
					(m[1][0] * bp.x + m[1][1] * bp.y + m[1][2] * bp.z + m[1][3]) * inv_w,
					(m[2][0] * bp.x + m[2][1] * bp.y + m[2][2] * bp.z + m[2][3]) * inv_w);
		}
		r_posed_pos[v] = p;
		if (r_local_aabb) {
			if (first) {
				bounds = AABB(p, Vector3());
				first = false;
			} else {
				bounds.expand_to(p);
			}
		}
	}

	if (r_local_aabb) {
		bounds.grow_by(0.002f);
		*r_local_aabb = bounds;
	}
	return true;
}

bool pose_mesh_positions_and_normals(
		MeshInstance *p_mi,
		const std::vector<Vector3> &p_bind_pos,
		const std::vector<Vector3> &p_bind_norm,
		const std::vector<int32_t> &p_bind_bones,
		const std::vector<float> &p_bind_weights,
		int p_bones_per_vert,
		std::vector<Vector3> &r_posed_pos,
		std::vector<Vector3> &r_posed_norm) {

	std::vector<Transform> skin_xforms;
	if (!compute_skin_transforms(p_mi, skin_xforms)) {
		return false;
	}

	int num_verts = (int)p_bind_pos.size();
	int num_skins = (int)skin_xforms.size();
	r_posed_pos.resize(num_verts);
	r_posed_norm.resize(num_verts);

	for (int v = 0; v < num_verts; v++) {
		const Vector3 &bp = p_bind_pos[v];
		const Vector3 &bn = p_bind_norm[v];
		float w_sum = 0.0f;
		float m[3][4] = {};
		int base = v * p_bones_per_vert;

		for (int k = 0; k < p_bones_per_vert; k++) {
			float w = p_bind_weights[base + k];
			int bi = p_bind_bones[base + k];
			if (w <= 1e-6f || bi < 0 || bi >= num_skins) continue;
			const Transform &xf = skin_xforms[bi];
			for (int r = 0; r < 3; r++) {
				m[r][0] += (float)xf.basis.elements[r][0] * w;
				m[r][1] += (float)xf.basis.elements[r][1] * w;
				m[r][2] += (float)xf.basis.elements[r][2] * w;
				m[r][3] += (float)xf.origin[r] * w;
			}
			w_sum += w;
		}

		if (w_sum <= 1e-5f) {
			r_posed_pos[v] = bp;
			r_posed_norm[v] = bn;
		} else {
			float inv_w = (std::abs(w_sum - 1.0f) > 1e-4f) ? (1.0f / w_sum) : 1.0f;
			Vector3 p(
					(m[0][0] * bp.x + m[0][1] * bp.y + m[0][2] * bp.z + m[0][3]) * inv_w,
					(m[1][0] * bp.x + m[1][1] * bp.y + m[1][2] * bp.z + m[1][3]) * inv_w,
					(m[2][0] * bp.x + m[2][1] * bp.y + m[2][2] * bp.z + m[2][3]) * inv_w);
			Vector3 n(
					m[0][0] * bn.x + m[0][1] * bn.y + m[0][2] * bn.z,
					m[1][0] * bn.x + m[1][1] * bn.y + m[1][2] * bn.z,
					m[2][0] * bn.x + m[2][1] * bn.y + m[2][2] * bn.z);
			float nlen = n.length();
			n = (nlen > 1e-6f) ? (n / nlen) : bn;
			r_posed_pos[v] = p;
			r_posed_norm[v] = n;
		}
	}
	return true;
}

void pack_pos_uv_triangles(
		const std::vector<Vector3> &p_posed_pos,
		const std::vector<Vector2> &p_bind_uv,
		const std::vector<uint32_t> &p_tri_verts,
		std::vector<float> &r_curr_verts,
		std::vector<float> *r_curr_xyz) {

	int num_tri_verts = (int)p_tri_verts.size();
	r_curr_verts.resize((size_t)num_tri_verts * 5);
	if (r_curr_xyz) {
		r_curr_xyz->resize((size_t)num_tri_verts * 3);
	}

	for (int i = 0; i < num_tri_verts; i++) {
		uint32_t vi = p_tri_verts[i];
		const Vector3 &p = p_posed_pos[vi];
		const Vector2 &uv = p_bind_uv[vi];
		r_curr_verts[i * 5 + 0] = (float)p.x;
		r_curr_verts[i * 5 + 1] = (float)p.y;
		r_curr_verts[i * 5 + 2] = (float)p.z;
		r_curr_verts[i * 5 + 3] = (float)uv.x;
		r_curr_verts[i * 5 + 4] = (float)uv.y;

		if (r_curr_xyz) {
			(*r_curr_xyz)[i * 3 + 0] = (float)p.x;
			(*r_curr_xyz)[i * 3 + 1] = (float)p.y;
			(*r_curr_xyz)[i * 3 + 2] = (float)p.z;
		}
	}
}

} // namespace GL43SkinningHelper
