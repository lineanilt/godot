#include "motion_blur.h"
#include "motion_blur_shaders.h"

#include "core/engine.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"
#include "scene/3d/camera.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual_server.h"

#include <cmath>

#ifndef GL_SHADER_IMAGE_ACCESS_BARRIER_BIT
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020
#endif
#ifndef GL_TEXTURE_FETCH_BARRIER_BIT
#define GL_TEXTURE_FETCH_BARRIER_BIT 0x00000008
#endif
#ifndef GL_WRITE_ONLY
#define GL_WRITE_ONLY 0x88B9
#endif

namespace {

static const char *MOTION_BLUR_COMPOSITE_FRAG_GLSL = R"GLSL(
#version 430 core
in vec2 vUv;
layout (location = 0) out vec4 outColor;

uniform sampler2D uSceneTex;
uniform sampler2D uVelocityTex;
uniform sampler2D uNeighborMaxTex;
uniform vec2 uViewportSize;
uniform int uSamples;
uniform float uJitter;
uniform float uShutterOffset;
uniform float uDepthSoftness;
uniform float uMinVelocityPx;
uniform float uMaxVelocityPx;
uniform int uDebugMode;

float ign(vec2 p) {
	return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

float isCloser(float zCandidate, float zRef, float extent) {
	return clamp(1.0 - (zCandidate - zRef) / max(extent, 0.001), 0.0, 1.0);
}

float cone(float distPx, float speedPx) {
	return clamp(1.0 - distPx / max(speedPx, 0.5), 0.0, 1.0);
}

float cylinder(float distPx, float speedPx) {
	return 1.0 - smoothstep(0.95 * speedPx, 1.05 * speedPx, distPx);
}

void main() {
	vec4 centerVel = textureLod(uVelocityTex, vUv, 0.0);
	vec4 nmaxVel = textureLod(uNeighborMaxTex, vUv, 0.0);

	vec2 vC = centerVel.rg;
	vec2 vN = nmaxVel.rg;

	if (uDebugMode == 1) {
		outColor = vec4(abs(vC) * 20.0, 0.0, 1.0);
		return;
	}
	if (uDebugMode == 2) {
		outColor = vec4(abs(vN) * 20.0, 0.0, 1.0);
		return;
	}
	if (uDebugMode == 3) {
		float sp = clamp(centerVel.a / max(uMaxVelocityPx, 1.0), 0.0, 1.0);
		outColor = vec4(vec3(sp), 1.0);
		return;
	}

	vec4 centerCol = textureLod(uSceneTex, vUv, 0.0);
	float speedN = length(vN * uViewportSize);
	if (speedN < uMinVelocityPx) {
		outColor = centerCol;
		return;
	}

	float zC = centerVel.b;
	float speedC = length(vC * uViewportSize);
	float effExtent = uDepthSoftness * max(0.5, zC * 0.05);

	vec3 centerLin = max(centerCol.rgb, vec3(0.0));
	float totalWeight = 1.0 / max(speedC, 0.5);
	vec3 accum = centerLin * totalWeight;

	float jitterVal = (ign(gl_FragCoord.xy) - 0.5) * uJitter;
	int sampleCount = clamp(uSamples, 3, 32);

	for (int i = 0; i < 32; i++) {
		if (i >= sampleCount) {
			break;
		}

		float t = (float(i) + 0.5 + jitterVal) / float(sampleCount) - 0.5 + uShutterOffset;
		vec2 dir = ((i & 1) == 1 && speedC > 0.5) ? vC : vN;
		float distPx = abs(t) * length(dir * uViewportSize);

		vec2 sUV = clamp(vUv + dir * t, vec2(0.0005), vec2(0.9995));
		vec3 sLin = max(textureLod(uSceneTex, sUV, 0.0).rgb, vec3(0.0));

		vec4 sVel = textureLod(uVelocityTex, sUV, 0.0);
		float zY = sVel.b;
		float speedY = length(sVel.rg * uViewportSize);

		float f = isCloser(zY, zC, effExtent);
		float b = isCloser(zC, zY, effExtent);

		float wY = f * cone(distPx, speedY)
				 + b * cone(distPx, speedC)
				 + cylinder(distPx, speedY) * cylinder(distPx, speedC) * 2.0;

		accum += sLin * wY;
		totalWeight += wY;
	}

	outColor = vec4(accum / max(totalWeight, 1e-5), centerCol.a);
}
)GLSL";

} // namespace

MotionBlur::MotionBlur() {
	velocity_texture.instance();
	neighbor_max_texture.instance();
}

MotionBlur::~MotionBlur() {
	_unregister_scene_hook();
	_cleanup_gl_resources();
}

void MotionBlur::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &MotionBlur::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &MotionBlur::is_enabled);

	ClassDB::bind_method(D_METHOD("set_editor_preview", "preview"), &MotionBlur::set_editor_preview);
	ClassDB::bind_method(D_METHOD("get_editor_preview"), &MotionBlur::get_editor_preview);

	ClassDB::bind_method(D_METHOD("set_auto_composite", "auto_composite"), &MotionBlur::set_auto_composite);
	ClassDB::bind_method(D_METHOD("get_auto_composite"), &MotionBlur::get_auto_composite);

	ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &MotionBlur::set_intensity);
	ClassDB::bind_method(D_METHOD("get_intensity"), &MotionBlur::get_intensity);

	ClassDB::bind_method(D_METHOD("set_samples", "samples"), &MotionBlur::set_samples);
	ClassDB::bind_method(D_METHOD("get_samples"), &MotionBlur::get_samples);

	ClassDB::bind_method(D_METHOD("set_max_velocity_px", "max_velocity_px"), &MotionBlur::set_max_velocity_px);
	ClassDB::bind_method(D_METHOD("get_max_velocity_px"), &MotionBlur::get_max_velocity_px);

	ClassDB::bind_method(D_METHOD("set_min_velocity_px", "min_velocity_px"), &MotionBlur::set_min_velocity_px);
	ClassDB::bind_method(D_METHOD("get_min_velocity_px"), &MotionBlur::get_min_velocity_px);

	ClassDB::bind_method(D_METHOD("set_tile_size", "tile_size"), &MotionBlur::set_tile_size);
	ClassDB::bind_method(D_METHOD("get_tile_size"), &MotionBlur::get_tile_size);

	ClassDB::bind_method(D_METHOD("set_velocity_scale", "scale"), &MotionBlur::set_velocity_scale);
	ClassDB::bind_method(D_METHOD("get_velocity_scale"), &MotionBlur::get_velocity_scale);

	ClassDB::bind_method(D_METHOD("set_jitter", "jitter"), &MotionBlur::set_jitter);
	ClassDB::bind_method(D_METHOD("get_jitter"), &MotionBlur::get_jitter);

	ClassDB::bind_method(D_METHOD("set_shutter_offset", "offset"), &MotionBlur::set_shutter_offset);
	ClassDB::bind_method(D_METHOD("get_shutter_offset"), &MotionBlur::get_shutter_offset);

	ClassDB::bind_method(D_METHOD("set_depth_softness", "softness"), &MotionBlur::set_depth_softness);
	ClassDB::bind_method(D_METHOD("get_depth_softness"), &MotionBlur::get_depth_softness);

	ClassDB::bind_method(D_METHOD("set_camera_motion", "enable"), &MotionBlur::set_camera_motion);
	ClassDB::bind_method(D_METHOD("get_camera_motion"), &MotionBlur::get_camera_motion);

	ClassDB::bind_method(D_METHOD("set_object_motion", "enable"), &MotionBlur::set_object_motion);
	ClassDB::bind_method(D_METHOD("get_object_motion"), &MotionBlur::get_object_motion);

	ClassDB::bind_method(D_METHOD("set_skinned_meshes", "enable"), &MotionBlur::set_skinned_meshes);
	ClassDB::bind_method(D_METHOD("get_skinned_meshes"), &MotionBlur::get_skinned_meshes);

	ClassDB::bind_method(D_METHOD("set_alpha_scissor_support", "enable"), &MotionBlur::set_alpha_scissor_support);
	ClassDB::bind_method(D_METHOD("get_alpha_scissor_support"), &MotionBlur::get_alpha_scissor_support);

	ClassDB::bind_method(D_METHOD("set_fps_compensation", "enable"), &MotionBlur::set_fps_compensation);
	ClassDB::bind_method(D_METHOD("get_fps_compensation"), &MotionBlur::get_fps_compensation);

	ClassDB::bind_method(D_METHOD("set_target_fps", "fps"), &MotionBlur::set_target_fps);
	ClassDB::bind_method(D_METHOD("get_target_fps"), &MotionBlur::get_target_fps);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &MotionBlur::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &MotionBlur::get_cull_mask);

	ClassDB::bind_method(D_METHOD("set_debug_mode", "mode"), &MotionBlur::set_debug_mode);
	ClassDB::bind_method(D_METHOD("get_debug_mode"), &MotionBlur::get_debug_mode);

	ClassDB::bind_method(D_METHOD("reset_history"), &MotionBlur::reset_history);
	ClassDB::bind_method(D_METHOD("get_velocity_texture"), &MotionBlur::get_velocity_texture);
	ClassDB::bind_method(D_METHOD("get_neighbor_max_texture"), &MotionBlur::get_neighbor_max_texture);
	ClassDB::bind_method(D_METHOD("_on_frame_pre_draw"), &MotionBlur::_on_frame_pre_draw);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_preview"), "set_editor_preview", "get_editor_preview");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_composite"), "set_auto_composite", "get_auto_composite");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "intensity", PROPERTY_HINT_RANGE, "0.0,4.0,0.05"), "set_intensity", "get_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "samples", PROPERTY_HINT_RANGE, "3,32,1"), "set_samples", "get_samples");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_velocity_px", PROPERTY_HINT_RANGE, "4.0,256.0,1.0"), "set_max_velocity_px", "get_max_velocity_px");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "min_velocity_px", PROPERTY_HINT_RANGE, "0.0,5.0,0.05"), "set_min_velocity_px", "get_min_velocity_px");

	ADD_GROUP("Filtering & Quality", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "tile_size", PROPERTY_HINT_ENUM, "8px:8,16px:16,24px:24,32px:32"), "set_tile_size", "get_tile_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "velocity_scale", PROPERTY_HINT_RANGE, "0.25,1.0,0.05"), "set_velocity_scale", "get_velocity_scale");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "jitter", PROPERTY_HINT_RANGE, "0.0,1.0,0.05"), "set_jitter", "get_jitter");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "shutter_offset", PROPERTY_HINT_RANGE, "-0.5,0.5,0.05"), "set_shutter_offset", "get_shutter_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "depth_softness", PROPERTY_HINT_RANGE, "0.01,2.0,0.01"), "set_depth_softness", "get_depth_softness");

	ADD_GROUP("Motion Sources", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "camera_motion"), "set_camera_motion", "get_camera_motion");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "object_motion"), "set_object_motion", "get_object_motion");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "skinned_meshes"), "set_skinned_meshes", "get_skinned_meshes");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_scissor_support"), "set_alpha_scissor_support", "get_alpha_scissor_support");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "fps_compensation"), "set_fps_compensation", "get_fps_compensation");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "target_fps", PROPERTY_HINT_RANGE, "15.0,240.0,1.0"), "set_target_fps", "get_target_fps");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");

	ADD_GROUP("Debug", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_mode", PROPERTY_HINT_ENUM, "None,Velocity,NeighborMax,Speed"), "set_debug_mode", "get_debug_mode");

	BIND_ENUM_CONSTANT(TILE_8);
	BIND_ENUM_CONSTANT(TILE_16);
	BIND_ENUM_CONSTANT(TILE_24);
	BIND_ENUM_CONSTANT(TILE_32);

	BIND_ENUM_CONSTANT(DEBUG_NONE);
	BIND_ENUM_CONSTANT(DEBUG_VELOCITY);
	BIND_ENUM_CONSTANT(DEBUG_NEIGHBOR_MAX);
	BIND_ENUM_CONSTANT(DEBUG_SPEED);
}

void MotionBlur::_validate_property(PropertyInfo &property) const {
	if (!fps_compensation && property.name == "target_fps") {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
}

void MotionBlur::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_register_scene_hook();
			set_process(true);
			set_process_priority(10000); // Run after game scripts & animations update transforms
			reset_history();
			if (VisualServer::get_singleton() && !VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
				VisualServer::get_singleton()->connect("frame_pre_draw", this, "_on_frame_pre_draw");
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (VisualServer::get_singleton() && VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
				VisualServer::get_singleton()->disconnect("frame_pre_draw", this, "_on_frame_pre_draw");
			}
			_unregister_scene_hook();
			reset_history();
		} break;

		case NOTIFICATION_PROCESS: {
			if (!enabled || !is_inside_tree()) {
				_detach_all_viewports();
				reset_history();
				return;
			}
			if (!Engine::get_singleton()->is_editor_hint()) {
				_collect_scene_snapshot();
			}
		} break;
	}
}

void MotionBlur::_register_scene_hook() {
	if (hook_registered) {
		return;
	}
	GL43EffectHelpers::register_scene_hook(this, GL43EffectHelpers::HOOK_PRE_POST_PROCESS, 20, [this](const GL43EffectHelpers::SceneHookContext &p_ctx) {
		_on_scene_hook(p_ctx);
	});
	hook_registered = true;
}

void MotionBlur::_unregister_scene_hook() {
	if (!hook_registered) {
		return;
	}
	GL43EffectHelpers::unregister_scene_hooks(this);
	hook_registered = false;
	_detach_all_viewports();
}

void MotionBlur::_detach_all_viewports() {
	MutexLock lock(scene_mutex);
	for (auto &kv : viewport_states) {
		kv.second.active_composite = false;
	}
}

MotionBlur::ViewportRenderState &MotionBlur::_get_or_create_vp_state(Viewport *p_vp) {
	uint32_t key = p_vp->get_viewport_rid().get_id();
	auto it = viewport_states.find(key);
	if (it == viewport_states.end()) {
		ViewportRenderState st;
		st.velocity_texture.instance();
		st.neighbor_max_texture.instance();
		it = viewport_states.emplace(key, std::move(st)).first;
	}
	if (VisualServer::get_singleton()) {
		it->second.color_tex_rid = VisualServer::get_singleton()->viewport_get_texture(p_vp->get_viewport_rid(), VS::VIEWPORT_TEXTURE_BUFFER_COLOR);
	}
	it->second.active_composite = auto_composite;
	it->second.last_seen = vp_collect_counter;
	return it->second;
}

void MotionBlur::_prune_stale_viewports() {
	bool gl_on_thread = gl_ready && gl_thread_id == Thread::get_caller_id();
	velocity_handler.prune_stale_viewports(vp_collect_counter, gl_on_thread);

	for (auto it = viewport_states.begin(); it != viewport_states.end(); ) {
		if (it->second.last_seen != vp_collect_counter) {
			it->second.active_composite = false;
			if (vp_collect_counter - it->second.last_seen > 60ULL) {
				if (gl_on_thread) {
					SharedGL43Context::ScopedScope gl43;
					if (gl43.is_active()) {
						if (it->second.tile_max_tex) {
							GLuint t = (GLuint)it->second.tile_max_tex;
							glDeleteTextures(1, &t);
						}
					}
				}
				it = viewport_states.erase(it);
				continue;
			}
		}
		++it;
	}
}

void MotionBlur::_collect_scene_snapshot() {
	SceneTree *tree = get_tree();
	Viewport *my_vp = get_viewport();
	if (!tree || !my_vp) return;

	vp_collect_counter++;
	std::vector<std::pair<Viewport *, Camera *>> active_vps;
	GL43EffectHelpers::find_active_viewports(this, my_vp, active_vps);
	if (active_vps.empty()) {
		MutexLock lock(scene_mutex);
		_prune_stale_viewports();
		velocity_handler.reset_history();
		return;
	}

	uint64_t now_usec = OS::get_singleton()->get_ticks_usec();
	float dt = (last_tick_usec > 0 && now_usec > last_tick_usec)
			? (float)((now_usec - last_tick_usec) / 1000000.0)
			: (1.0f / MAX(target_fps, 1.0f));
	last_tick_usec = now_usec;
	dt = CLAMP(dt, 0.0005f, 0.25f);

	float fps_scale = fps_compensation ? CLAMP((1.0f / MAX(target_fps, 1.0f)) / dt, 0.1f, 4.0f) : 1.0f;
	float eff_vel_scale = intensity * fps_scale;
	Node *search_root = my_vp ? (Node *)my_vp : (Node *)tree->get_root();

	MutexLock lock(scene_mutex);
	velocity_handler.begin_collection();

	for (size_t vi = 0; vi < active_vps.size(); vi++) {
		Viewport *vp = active_vps[vi].first;
		Camera *vcam = active_vps[vi].second;
		Size2 vsz = vp->get_size();

		int vel_w = MAX(16, (int)Math::round(vsz.x * velocity_scale));
		int vel_h = MAX(16, (int)Math::round(vsz.y * velocity_scale));
		int eff_tile = MAX(2, (int)Math::round((float)tile_size * velocity_scale));
		int tile_w = (vel_w + eff_tile - 1) / eff_tile;
		int tile_h = (vel_h + eff_tile - 1) / eff_tile;

		ViewportRenderState &vst = _get_or_create_vp_state(vp);
		if (vst.tex_vel_w != vel_w || vst.tex_vel_h != vel_h) {
			vst.tex_vel_w = vel_w;
			vst.tex_vel_h = vel_h;
			Ref<Image> vimg;
			vimg.instance();
			vimg->create(vel_w, vel_h, false, Image::FORMAT_RGBAH);
			vimg->fill(Color(0.0f, 0.0f, 0.0f, 0.0f));
			vst.velocity_texture->create_from_image(vimg, 0);
		}
		if (vst.tex_tile_w != tile_w || vst.tex_tile_h != tile_h) {
			vst.tex_tile_w = tile_w;
			vst.tex_tile_h = tile_h;
			Ref<Image> nimg;
			nimg.instance();
			nimg->create(tile_w, tile_h, false, Image::FORMAT_RGBAH);
			nimg->fill(Color(0.0f, 0.0f, 0.0f, 0.0f));
			vst.neighbor_max_texture->create_from_image(nimg, 0);
		}

		if (vi == 0) {
			velocity_texture = vst.velocity_texture;
			neighbor_max_texture = vst.neighbor_max_texture;
		}

		vst.viewport_w = (int)vsz.x;
		vst.viewport_h = (int)vsz.y;

		velocity_handler.add_viewport_target(
				vp->get_viewport_rid().get_id(),
				vcam,
				vsz,
				vel_w,
				vel_h,
				camera_motion,
				eff_vel_scale,
				max_velocity_px,
				vst.velocity_texture->get_rid(),
				vp_collect_counter);
	}
	_prune_stale_viewports();

	velocity_handler.collect_scene_meshes(search_root, cull_mask, alpha_scissor_support, object_motion, skinned_meshes);
	velocity_handler.end_collection(true);
}

bool MotionBlur::_init_gl_resources() {
	if (gl_attempted) return gl_ready;
	gl_attempted = true;
	gl_thread_id = Thread::get_caller_id();

	if (!SharedGL43Context::get_singleton()->is_available()) {
		ERR_PRINT("MotionBlur: Shared OpenGL 4.3 context not available.");
		return false;
	}

	if (!velocity_handler.init_gl("MotionBlur")) {
		return false;
	}
	tile_max_program = GL43EffectHelpers::link_compute_program(MotionBlurShaders::TILE_MAX_COMPUTE_GLSL, "MotionBlur", "TileMax");
	neighbor_max_program = GL43EffectHelpers::link_compute_program(MotionBlurShaders::NEIGHBOR_MAX_COMPUTE_GLSL, "MotionBlur", "NeighborMax");
	composite_program = GL43EffectHelpers::link_vf_program(GL43EffectHelpers::get_fullscreen_vert_glsl(), MOTION_BLUR_COMPOSITE_FRAG_GLSL, "MotionBlur", "Composite");

	if (!tile_max_program || !neighbor_max_program || !composite_program) {
		return false;
	}

	gl_ready = true;
	return true;
}

void MotionBlur::_cleanup_gl_resources() {
	if (!gl_ready || gl_thread_id != Thread::get_caller_id()) {
		return;
	}

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	velocity_handler.cleanup_gl();

	if (tile_max_program) {
		glDeleteProgram((GLuint)tile_max_program);
		tile_max_program = 0;
	}
	if (neighbor_max_program) {
		glDeleteProgram((GLuint)neighbor_max_program);
		neighbor_max_program = 0;
	}
	if (composite_program) {
		glDeleteProgram((GLuint)composite_program);
		composite_program = 0;
	}

	for (auto &kv : viewport_states) {
		if (kv.second.tile_max_tex) {
			GLuint t = (GLuint)kv.second.tile_max_tex;
			glDeleteTextures(1, &t);
			kv.second.tile_max_tex = 0;
		}
	}

	gl_ready = false;
	gl_attempted = false;
}

void MotionBlur::_ensure_tile_tex_size(ViewportRenderState &r_state, int p_w, int p_h) {
	if (r_state.tile_max_tex != 0 && r_state.alloc_tile_w == p_w && r_state.alloc_tile_h == p_h) return;
	r_state.alloc_tile_w = p_w;
	r_state.alloc_tile_h = p_h;

	glActiveTexture(GL_TEXTURE0);
	GL43EffectHelpers::ensure_rgba16f_texture(r_state.tile_max_tex, p_w, p_h, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void MotionBlur::_on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx) {
	if (!enabled || !auto_composite || !is_inside_tree() || !gl_ready || composite_program == 0) {
		return;
	}
	if (Engine::get_singleton()->is_editor_hint() && !editor_preview) {
		return;
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	RID vel_rid;
	RID nmax_rid;
	int vp_w = p_ctx.width;
	int vp_h = p_ctx.height;
	{
		MutexLock lock(scene_mutex);
		if (!velocity_handler.is_valid()) return;
		for (const auto &kv : viewport_states) {
			if (kv.second.color_tex_rid == p_ctx.color_tex_rid && kv.second.active_composite && kv.second.velocity_texture.is_valid() && kv.second.neighbor_max_texture.is_valid()) {
				vel_rid = kv.second.velocity_texture->get_rid();
				nmax_rid = kv.second.neighbor_max_texture->get_rid();
				if (kv.second.viewport_w > 0 && kv.second.viewport_h > 0) {
					vp_w = kv.second.viewport_w;
					vp_h = kv.second.viewport_h;
				}
				break;
			}
		}
	}
	if (!vel_rid.is_valid() || !nmax_rid.is_valid()) return;

	RasterizerStorageGLES3::Texture *vel_tex = rs->texture_owner.getornull(vel_rid);
	RasterizerStorageGLES3::Texture *nmax_tex = rs->texture_owner.getornull(nmax_rid);
	if (!vel_tex || vel_tex->tex_id == 0 || !nmax_tex || nmax_tex->tex_id == 0) return;

	uint32_t scene_copy_tex = 0;
	if (debug_mode == DEBUG_NONE) {
		scene_copy_tex = GL43EffectHelpers::copy_scene_color_to_scratch(p_ctx);
		if (scene_copy_tex == 0) return;
	}

	glUseProgram((GLuint)composite_program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, (GLuint)scene_copy_tex);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uSceneTex"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, vel_tex->tex_id);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uVelocityTex"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, nmax_tex->tex_id);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uNeighborMaxTex"), 2);

	glUniform2f(glGetUniformLocation((GLuint)composite_program, "uViewportSize"), (float)vp_w, (float)vp_h);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uSamples"), samples);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uJitter"), jitter);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uShutterOffset"), shutter_offset);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uDepthSoftness"), depth_softness);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uMinVelocityPx"), min_velocity_px);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uMaxVelocityPx"), max_velocity_px);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uDebugMode"), (int)debug_mode);

	glDisable(GL_BLEND);
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void MotionBlur::_on_frame_pre_draw() {
	if (!hook_registered || !is_inside_tree()) return;

	bool in_editor = Engine::get_singleton()->is_editor_hint();
	if (!enabled || (in_editor && !editor_preview) || !get_viewport()) {
		_detach_all_viewports();
		return;
	}

	if (in_editor) {
		_collect_scene_snapshot();
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	MutexLock lock(scene_mutex);
	if (!velocity_handler.is_valid() || velocity_handler.get_viewport_targets().empty()) return;

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	if (!_init_gl_resources()) {
		return;
	}

	if (!velocity_handler.render_velocities(rs)) {
		return;
	}

	SharedGL43Context *gl43_ctx = SharedGL43Context::get_singleton();
	const int eff_tile = MAX(2, (int)Math::round((float)tile_size * velocity_scale));
	const float max_reach_px = (0.5f + Math::abs(shutter_offset)) * max_velocity_px;
	const int tile_radius = CLAMP((int)Math::ceil(max_reach_px / (float)tile_size), 1, 4);

	for (const GL43VelocityHandler::ViewportTarget &vt : velocity_handler.get_viewport_targets()) {
		auto st_it = viewport_states.find(vt.vp_id);
		if (st_it == viewport_states.end()) continue;
		ViewportRenderState &vst = st_it->second;

		int tile_w = (vt.vel_w + eff_tile - 1) / eff_tile;
		int tile_h = (vt.vel_h + eff_tile - 1) / eff_tile;

		RasterizerStorageGLES3::Texture *godot_vel_tex = rs->texture_owner.getornull(vst.velocity_texture->get_rid());
		RasterizerStorageGLES3::Texture *godot_nmax_tex = rs->texture_owner.getornull(vst.neighbor_max_texture->get_rid());
		if (!godot_vel_tex || godot_vel_tex->tex_id == 0 || !godot_nmax_tex || godot_nmax_tex->tex_id == 0) continue;
		if (godot_vel_tex->width != vt.vel_w || godot_vel_tex->height != vt.vel_h) continue;
		if (godot_nmax_tex->width != tile_w || godot_nmax_tex->height != tile_h) continue;

		_ensure_tile_tex_size(vst, tile_w, tile_h);
		GLuint tgx = (GLuint)((tile_w + 7) / 8);
		GLuint tgy = (GLuint)((tile_h + 7) / 8);

		glUseProgram((GLuint)tile_max_program);
		gl43_ctx->bind_image_texture(0, vst.tile_max_tex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, godot_vel_tex->tex_id);
		glUniform1i(glGetUniformLocation((GLuint)tile_max_program, "uVelocityTex"), 0);
		glUniform1i(glGetUniformLocation((GLuint)tile_max_program, "uTileSize"), eff_tile);

		gl43_ctx->dispatch_compute(tgx, tgy, 1);
		gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

		glUseProgram((GLuint)neighbor_max_program);
		gl43_ctx->bind_image_texture(0, godot_nmax_tex->tex_id, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, vst.tile_max_tex);
		glUniform1i(glGetUniformLocation((GLuint)neighbor_max_program, "uTileMaxTex"), 0);
		glUniform1i(glGetUniformLocation((GLuint)neighbor_max_program, "uTileRadius"), tile_radius);

		gl43_ctx->dispatch_compute(tgx, tgy, 1);
		gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, godot_nmax_tex->tex_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void MotionBlur::reset_history() {
	MutexLock lock(scene_mutex);
	last_tick_usec = 0;
	velocity_handler.reset_history();
}

void MotionBlur::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	if (!enabled) {
		_detach_all_viewports();
		reset_history();
	}
}

void MotionBlur::set_editor_preview(bool p_preview) {
	editor_preview = p_preview;
	if (!editor_preview && Engine::get_singleton()->is_editor_hint()) {
		_detach_all_viewports();
		reset_history();
	}
}

void MotionBlur::set_auto_composite(bool p_auto) {
	auto_composite = p_auto;
	if (!auto_composite) {
		_detach_all_viewports();
	}
}

void MotionBlur::set_intensity(float p_intensity) {
	intensity = CLAMP(p_intensity, 0.0f, 4.0f);
}

void MotionBlur::set_samples(int p_samples) {
	samples = CLAMP(p_samples, 3, 32);
}

void MotionBlur::set_max_velocity_px(float p_max_px) {
	max_velocity_px = CLAMP(p_max_px, 4.0f, 256.0f);
}

void MotionBlur::set_min_velocity_px(float p_min_px) {
	min_velocity_px = CLAMP(p_min_px, 0.0f, 5.0f);
}

void MotionBlur::set_tile_size(TileSize p_size) {
	tile_size = p_size;
}

void MotionBlur::set_velocity_scale(float p_scale) {
	velocity_scale = CLAMP(p_scale, 0.25f, 1.0f);
}

void MotionBlur::set_jitter(float p_jitter) {
	jitter = CLAMP(p_jitter, 0.0f, 1.0f);
}

void MotionBlur::set_shutter_offset(float p_offset) {
	shutter_offset = CLAMP(p_offset, -0.5f, 0.5f);
}

void MotionBlur::set_depth_softness(float p_softness) {
	depth_softness = CLAMP(p_softness, 0.01f, 2.0f);
}

void MotionBlur::set_fps_compensation(bool p_enable) {
	fps_compensation = p_enable;
	_change_notify();
}

void MotionBlur::set_target_fps(float p_fps) {
	target_fps = CLAMP(p_fps, 15.0f, 240.0f);
}
