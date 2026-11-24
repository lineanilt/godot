#include "taa.h"
#include "taa_shaders.h"

#include "core/engine.h"
#include "core/math/math_funcs.h"
#include "core/os/os.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"
#include "scene/3d/camera.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual/visual_server_viewport.h"
#include "servers/visual_server.h"

#ifndef GL_FRAMEBUFFER_SRGB
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#endif
#ifndef GL_READ_FRAMEBUFFER
#define GL_READ_FRAMEBUFFER 0x8CA8
#endif
#ifndef GL_DRAW_FRAMEBUFFER
#define GL_DRAW_FRAMEBUFFER 0x8CA9
#endif

namespace {

static const char *TAA_COMPOSITE_FRAG_GLSL = R"GLSL(
#version 430 core
in vec2 vUv;
layout (location = 0) out vec4 outColor;

uniform sampler2D uSceneTex;
uniform sampler2D uHistoryTex;
uniform sampler2D uVelocityTex;
uniform float uHistoryValid;
uniform float uHistoryBlend;
uniform float uMotionRejection;
uniform float uVarianceGamma;
uniform float uSharpenIntensity;
uniform int uCatmullRom;
uniform int uVarianceClipping;
uniform vec2 uJitterPx;
uniform vec2 uPixelSize;
uniform int uDebugMode;

vec3 rgbToYCoCg(vec3 c) {
	return vec3(
		 0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
		 0.5  * c.r - 0.5 * c.b,
		-0.25 * c.r + 0.5 * c.g - 0.25 * c.b
	);
}

vec3 ycocgToRgb(vec3 c) {
	float tmp = c.x - c.z;
	return max(vec3(tmp + c.y, c.x + c.z, tmp - c.y), vec3(0.0));
}

float karisWeight(vec3 lin) {
	float luma = dot(lin, vec3(0.2126, 0.7152, 0.0722));
	return 1.0 / (1.0 + max(luma, 0.0));
}

vec3 clipAABB(vec3 aabbMin, vec3 aabbMax, vec3 history) {
	vec3 pClip = 0.5 * (aabbMax + aabbMin);
	vec3 eClip = 0.5 * (aabbMax - aabbMin) + 1e-5;
	vec3 vClip = history - pClip;
	vec3 aUnit = abs(vClip / eClip);
	float maUnit = max(aUnit.x, max(aUnit.y, aUnit.z));
	if (maUnit > 1.0) {
		return pClip + vClip / maUnit;
	}
	return history;
}

vec3 sampleHistoryCatmullRom(sampler2D tex, vec2 uv) {
	vec2 texSize = vec2(textureSize(tex, 0));
	vec2 invTexSize = 1.0 / max(texSize, vec2(1.0));
	vec2 samplePos = uv * texSize;
	vec2 tc1 = floor(samplePos - 0.5) + 0.5;
	vec2 f = samplePos - tc1;
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;

	vec2 w0 = f2 - 0.5 * (f3 + f);
	vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
	vec2 w3 = 0.5 * (f3 - f2);
	vec2 w2 = 1.0 - w0 - w1 - w3;

	vec2 w12 = w1 + w2;
	vec2 tc0 = (tc1 - 1.0) * invTexSize;
	vec2 tc3 = (tc1 + 2.0) * invTexSize;
	vec2 tc12 = (tc1 + w2 / max(w12, vec2(1e-5))) * invTexSize;

	vec3 res =
		textureLod(tex, vec2(tc12.x, tc0.y), 0.0).rgb * (w12.x * w0.y) +
		textureLod(tex, vec2(tc0.x, tc12.y), 0.0).rgb * (w0.x * w12.y) +
		textureLod(tex, vec2(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y) +
		textureLod(tex, vec2(tc3.x, tc12.y), 0.0).rgb * (w3.x * w12.y) +
		textureLod(tex, vec2(tc12.x, tc3.y), 0.0).rgb * (w12.x * w3.y);

	float wSum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
	return max(res / max(wSum, 1e-5), vec3(0.0));
}

vec2 findClosestVelocity(vec2 uv, out float outSpeedPx) {
	vec2 velTexel = 1.0 / max(vec2(textureSize(uVelocityTex, 0)), vec2(1.0));
	float bestZ = 1e20;
	vec2 bestVel = vec2(0.0);
	outSpeedPx = 0.0;

	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			vec2 suv = clamp(uv + vec2(float(dx), float(dy)) * velTexel, vec2(0.0), vec2(1.0));
			vec4 v = textureLod(uVelocityTex, suv, 0.0);
			if (v.b < bestZ) {
				bestZ = v.b;
				bestVel = v.rg;
				outSpeedPx = v.a;
			}
		}
	}
	return bestVel;
}

void main() {
	float speedPx = 0.0;
	vec2 velUV = findClosestVelocity(vUv, speedPx);

	if (uDebugMode == 1) {
		outColor = vec4(abs(velUV) * 25.0, 0.0, 1.0);
		return;
	}
	if (uDebugMode == 2) {
		outColor = vec4(abs(uJitterPx) + 0.25, 0.0, 1.0);
		return;
	}

	vec4 curSample = textureLod(uSceneTex, vUv, 0.0);
	vec3 curLin = max(curSample.rgb, vec3(0.0));

	if (uHistoryValid < 0.5) {
		outColor = curSample;
		return;
	}

	vec3 m1 = vec3(0.0);
	vec3 m2 = vec3(0.0);
	vec3 nMin = vec3(1e10);
	vec3 nMax = vec3(-1e10);

	vec3 cN = curLin;
	vec3 cS = curLin;
	vec3 cE = curLin;
	vec3 cW = curLin;

	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			vec2 suv = clamp(vUv + vec2(float(dx), float(dy)) * uPixelSize, vec2(0.0005), vec2(0.9995));
			vec3 sLin = max(textureLod(uSceneTex, suv, 0.0).rgb, vec3(0.0));
			vec3 sYC = rgbToYCoCg(sLin);

			m1 += sYC;
			m2 += sYC * sYC;
			nMin = min(nMin, sYC);
			nMax = max(nMax, sYC);

			if (dx == 0 && dy == -1) cN = sLin;
			else if (dx == 0 && dy == 1) cS = sLin;
			else if (dx == 1 && dy == 0) cE = sLin;
			else if (dx == -1 && dy == 0) cW = sLin;
		}
	}

	vec2 prevUV = vUv - velUV;
	bool onScreen = (prevUV.x >= 0.0 && prevUV.x <= 1.0 && prevUV.y >= 0.0 && prevUV.y <= 1.0);

	vec3 resolvedLin = curLin;
	float wasClipped = 0.0;

	if (onScreen) {
		vec3 histLin = (uCatmullRom != 0)
			? sampleHistoryCatmullRom(uHistoryTex, prevUV)
			: max(textureLod(uHistoryTex, prevUV, 0.0).rgb, vec3(0.0));
		vec3 histYC = rgbToYCoCg(histLin);

		vec3 boxMin = nMin;
		vec3 boxMax = nMax;
		if (uVarianceClipping != 0) {
			vec3 mean = m1 / 9.0;
			vec3 stddev = sqrt(max(m2 / 9.0 - mean * mean, vec3(0.0)));
			boxMin = max(nMin, mean - stddev * uVarianceGamma);
			boxMax = min(nMax, mean + stddev * uVarianceGamma);
		}

		vec3 clippedYC = clipAABB(boxMin, boxMax, histYC);
		wasClipped = clamp(length(clippedYC - histYC) * 4.0, 0.0, 1.0);

		float wHist = clamp(uHistoryBlend, 0.0, 0.97);
		wHist *= 1.0 - clamp((speedPx / 24.0) * uMotionRejection, 0.0, 0.65);

		vec3 clippedLin = ycocgToRgb(clippedYC);
		float kwCur = karisWeight(curLin);
		float kwHist = karisWeight(clippedLin);
		float aCur = (1.0 - wHist) * kwCur;
		float aHist = wHist * kwHist;
		resolvedLin = (curLin * aCur + clippedLin * aHist) / max(aCur + aHist, 1e-5);
	}

	if (uSharpenIntensity > 0.001) {
		vec3 minRing = min(min(cN, cS), min(cE, cW));
		vec3 maxRing = max(max(cN, cS), max(cE, cW));
		vec3 avgRing = (cN + cS + cE + cW) * 0.25;
		vec3 sharp = resolvedLin + (resolvedLin - avgRing) * (uSharpenIntensity * 1.25);
		resolvedLin = clamp(sharp, min(resolvedLin, minRing), max(resolvedLin, maxRing));
	}

	if (uDebugMode == 3) {
		vec3 diff = abs(resolvedLin - curLin) * 6.0;
		outColor = vec4(clamp(diff + vec3(wasClipped * 0.35, 0.0, 0.0), 0.0, 1.0), 1.0);
	} else {
		outColor = vec4(max(resolvedLin, vec3(0.0)), curSample.a);
	}
}
)GLSL";

static float halton_seq(int p_index, int p_base) {
	float f = 1.0f;
	float r = 0.0f;
	int i = p_index;
	while (i > 0) {
		f /= (float)p_base;
		r += f * (float)(i % p_base);
		i /= p_base;
	}
	return r;
}

} // namespace

TAA::TAA() {
	velocity_texture.instance();
	history_texture.instance();
}

TAA::~TAA() {
	_restore_all_camera_jitters();
	_unregister_scene_hook();
	_cleanup_gl_resources();
}

void TAA::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &TAA::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &TAA::is_enabled);

	ClassDB::bind_method(D_METHOD("set_editor_preview", "preview"), &TAA::set_editor_preview);
	ClassDB::bind_method(D_METHOD("get_editor_preview"), &TAA::get_editor_preview);

	ClassDB::bind_method(D_METHOD("set_jitter_pattern", "pattern"), &TAA::set_jitter_pattern);
	ClassDB::bind_method(D_METHOD("get_jitter_pattern"), &TAA::get_jitter_pattern);

	ClassDB::bind_method(D_METHOD("set_jitter_scale", "scale"), &TAA::set_jitter_scale);
	ClassDB::bind_method(D_METHOD("get_jitter_scale"), &TAA::get_jitter_scale);

	ClassDB::bind_method(D_METHOD("set_history_blend", "blend"), &TAA::set_history_blend);
	ClassDB::bind_method(D_METHOD("get_history_blend"), &TAA::get_history_blend);

	ClassDB::bind_method(D_METHOD("set_motion_rejection", "rejection"), &TAA::set_motion_rejection);
	ClassDB::bind_method(D_METHOD("get_motion_rejection"), &TAA::get_motion_rejection);

	ClassDB::bind_method(D_METHOD("set_variance_clipping", "enable"), &TAA::set_variance_clipping);
	ClassDB::bind_method(D_METHOD("get_variance_clipping"), &TAA::get_variance_clipping);

	ClassDB::bind_method(D_METHOD("set_variance_gamma", "gamma"), &TAA::set_variance_gamma);
	ClassDB::bind_method(D_METHOD("get_variance_gamma"), &TAA::get_variance_gamma);

	ClassDB::bind_method(D_METHOD("set_catmull_rom_history", "enable"), &TAA::set_catmull_rom_history);
	ClassDB::bind_method(D_METHOD("get_catmull_rom_history"), &TAA::get_catmull_rom_history);

	ClassDB::bind_method(D_METHOD("set_sharpen_intensity", "intensity"), &TAA::set_sharpen_intensity);
	ClassDB::bind_method(D_METHOD("get_sharpen_intensity"), &TAA::get_sharpen_intensity);

	ClassDB::bind_method(D_METHOD("set_alpha_scissor_support", "enable"), &TAA::set_alpha_scissor_support);
	ClassDB::bind_method(D_METHOD("get_alpha_scissor_support"), &TAA::get_alpha_scissor_support);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &TAA::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &TAA::get_cull_mask);

	ClassDB::bind_method(D_METHOD("set_debug_mode", "mode"), &TAA::set_debug_mode);
	ClassDB::bind_method(D_METHOD("get_debug_mode"), &TAA::get_debug_mode);

	ClassDB::bind_method(D_METHOD("reset_history"), &TAA::reset_history);
	ClassDB::bind_method(D_METHOD("get_velocity_texture"), &TAA::get_velocity_texture);
	ClassDB::bind_method(D_METHOD("get_history_texture"), &TAA::get_history_texture);
	ClassDB::bind_method(D_METHOD("_on_frame_pre_draw"), &TAA::_on_frame_pre_draw);
	ClassDB::bind_method(D_METHOD("_on_frame_post_draw"), &TAA::_on_frame_post_draw);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_preview"), "set_editor_preview", "get_editor_preview");

	ADD_GROUP("Jitter & Sampling", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "jitter_pattern", PROPERTY_HINT_ENUM, "Halton 8,Halton 16,R2 Sequence"), "set_jitter_pattern", "get_jitter_pattern");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "jitter_scale", PROPERTY_HINT_RANGE, "0.0,2.0,0.05"), "set_jitter_scale", "get_jitter_scale");

	ADD_GROUP("Resolve & Anti-Ghosting", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "history_blend", PROPERTY_HINT_RANGE, "0.0,0.97,0.01"), "set_history_blend", "get_history_blend");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "motion_rejection", PROPERTY_HINT_RANGE, "0.0,1.0,0.05"), "set_motion_rejection", "get_motion_rejection");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "variance_clipping"), "set_variance_clipping", "get_variance_clipping");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "variance_gamma", PROPERTY_HINT_RANGE, "0.5,2.5,0.05"), "set_variance_gamma", "get_variance_gamma");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "catmull_rom_history"), "set_catmull_rom_history", "get_catmull_rom_history");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "sharpen_intensity", PROPERTY_HINT_RANGE, "0.0,1.0,0.05"), "set_sharpen_intensity", "get_sharpen_intensity");

	ADD_GROUP("Velocity Pass", "");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_scissor_support"), "set_alpha_scissor_support", "get_alpha_scissor_support");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");

	ADD_GROUP("Debug", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_mode", PROPERTY_HINT_ENUM, "None,Velocity,Jitter,Clip & Diff"), "set_debug_mode", "get_debug_mode");

	BIND_ENUM_CONSTANT(JITTER_HALTON_8);
	BIND_ENUM_CONSTANT(JITTER_HALTON_16);
	BIND_ENUM_CONSTANT(JITTER_R2);

	BIND_ENUM_CONSTANT(DEBUG_NONE);
	BIND_ENUM_CONSTANT(DEBUG_VELOCITY);
	BIND_ENUM_CONSTANT(DEBUG_JITTER);
	BIND_ENUM_CONSTANT(DEBUG_CLIP_DIFF);
}

void TAA::_validate_property(PropertyInfo &property) const {
	if (!variance_clipping && property.name == "variance_gamma") {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
}

void TAA::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_register_scene_hook();
			set_process(true);
			set_process_priority(10000);
			reset_history();
			if (VisualServer::get_singleton()) {
				if (!VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
					VisualServer::get_singleton()->connect("frame_pre_draw", this, "_on_frame_pre_draw");
				}
				if (!VisualServer::get_singleton()->is_connected("frame_post_draw", this, "_on_frame_post_draw")) {
					VisualServer::get_singleton()->connect("frame_post_draw", this, "_on_frame_post_draw");
				}
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_restore_all_camera_jitters();
			if (VisualServer::get_singleton()) {
				if (VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
					VisualServer::get_singleton()->disconnect("frame_pre_draw", this, "_on_frame_pre_draw");
				}
				if (VisualServer::get_singleton()->is_connected("frame_post_draw", this, "_on_frame_post_draw")) {
					VisualServer::get_singleton()->disconnect("frame_post_draw", this, "_on_frame_post_draw");
				}
			}
			_unregister_scene_hook();
			reset_history();
		} break;

		case NOTIFICATION_PROCESS: {
			if (!enabled || !is_inside_tree()) {
				_restore_all_camera_jitters();
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

void TAA::_register_scene_hook() {
	if (hook_registered) {
		return;
	}
	GL43EffectHelpers::register_scene_hook(this, GL43EffectHelpers::HOOK_PRE_POST_PROCESS, 10, [this](const GL43EffectHelpers::SceneHookContext &p_ctx) {
		_on_scene_hook(p_ctx);
	});
	hook_registered = true;
}

void TAA::_unregister_scene_hook() {
	if (!hook_registered) {
		return;
	}
	GL43EffectHelpers::unregister_scene_hooks(this);
	hook_registered = false;
	_detach_all_viewports();
}

void TAA::_detach_all_viewports() {
	MutexLock lock(scene_mutex);
	for (auto &kv : viewport_states) {
		kv.second.active_composite = false;
		kv.second.history_valid = false;
	}
}

TAA::ViewportRenderState &TAA::_get_or_create_vp_state(Viewport *p_vp) {
	uint32_t key = p_vp->get_viewport_rid().get_id();
	auto it = viewport_states.find(key);
	if (it == viewport_states.end()) {
		ViewportRenderState st;
		st.velocity_texture.instance();
		st.history_texture.instance();
		it = viewport_states.emplace(key, std::move(st)).first;
	}
	if (VisualServer::get_singleton()) {
		it->second.color_tex_rid = VisualServer::get_singleton()->viewport_get_texture(p_vp->get_viewport_rid(), VS::VIEWPORT_TEXTURE_BUFFER_COLOR);
	}
	it->second.active_composite = true;
	it->second.last_seen = vp_collect_counter;
	return it->second;
}

void TAA::_prune_stale_viewports() {
	bool gl_on_thread = gl_ready && gl_thread_id == Thread::get_caller_id();
	velocity_handler.prune_stale_viewports(vp_collect_counter, gl_on_thread);

	for (auto it = viewport_states.begin(); it != viewport_states.end(); ) {
		if (it->second.last_seen != vp_collect_counter) {
			_restore_camera_jitter(it->second);
			it->second.active_composite = false;
			it->second.history_valid = false;
			if (vp_collect_counter - it->second.last_seen > 60ULL) {
				it = viewport_states.erase(it);
				continue;
			}
		}
		++it;
	}
}

Vector2 TAA::_compute_jitter_px(uint64_t p_frame_idx) const {
	if (jitter_scale <= 0.0001f) {
		return Vector2(0.0f, 0.0f);
	}
	float jx = 0.0f;
	float jy = 0.0f;
	if (jitter_pattern == JITTER_R2) {
		float n = (float)((p_frame_idx % 64ULL) + 1ULL);
		jx = Math::fmod(n * 0.7548776662466927f, 1.0f) - 0.5f;
		jy = Math::fmod(n * 0.5698402909980532f, 1.0f) - 0.5f;
	} else {
		int period = (jitter_pattern == JITTER_HALTON_8) ? 8 : 16;
		int idx = (int)(p_frame_idx % (uint64_t)period) + 1;
		jx = halton_seq(idx, 2) - 0.5f;
		jy = halton_seq(idx, 3) - 0.5f;
	}
	return Vector2(jx * jitter_scale, jy * jitter_scale);
}

void TAA::_apply_camera_jitter(ViewportRenderState &r_state, Camera *p_cam, const Size2 &p_vp_size, const Vector2 &p_jitter_px) {
	if (!p_cam || !VisualServer::get_singleton() || p_vp_size.x < 4 || p_vp_size.y < 4) {
		return;
	}
	RID cam_rid = p_cam->get_camera();
	if (!cam_rid.is_valid()) {
		return;
	}

	if (r_state.jittered_cam_rid.is_valid() && r_state.jittered_cam_rid != cam_rid) {
		_restore_camera_jitter(r_state);
	}

	bool is_ortho = (p_cam->get_projection() == Camera::PROJECTION_ORTHOGONAL);
	float aspect = (float)(p_vp_size.x / p_vp_size.y);
	bool flip_fov = (p_cam->get_keep_aspect_mode() == Camera::KEEP_WIDTH);
	float znear = MAX(p_cam->get_znear(), 0.001f);
	float zfar = MAX(p_cam->get_zfar(), znear + 0.1f);

	r_state.jittered_cam_rid = cam_rid;
	r_state.jittered_was_ortho = is_ortho;
	r_state.jittered_fov = p_cam->get_fov();
	r_state.jittered_size = p_cam->get_size();
	r_state.jittered_near = znear;
	r_state.jittered_far = zfar;
	r_state.jittered_base_xf = p_cam->get_camera_transform();

	if (is_ortho) {
		float world_w = flip_fov ? r_state.jittered_size : (r_state.jittered_size * aspect);
		float world_h = flip_fov ? (r_state.jittered_size / MAX(aspect, 0.001f)) : r_state.jittered_size;
		Transform jxf = r_state.jittered_base_xf;
		jxf.origin += jxf.basis.get_axis(0) * (p_jitter_px.x * world_w / (float)p_vp_size.x);
		jxf.origin += jxf.basis.get_axis(1) * (p_jitter_px.y * world_h / (float)p_vp_size.y);
		VisualServer::get_singleton()->camera_set_transform(cam_rid, jxf);
	} else if (p_cam->get_projection() == Camera::PROJECTION_PERSPECTIVE) {
		float frustum_size = 2.0f * znear * Math::tan(Math::deg2rad(r_state.jittered_fov) * 0.5f);
		float near_w = flip_fov ? frustum_size : (frustum_size * aspect);
		float near_h = flip_fov ? (frustum_size / MAX(aspect, 0.001f)) : frustum_size;
		Vector2 offset(
				p_jitter_px.x * (near_w / (float)p_vp_size.x),
				p_jitter_px.y * (near_h / (float)p_vp_size.y));
		VisualServer::get_singleton()->camera_set_frustum(cam_rid, frustum_size, offset, znear, zfar);
	}
}

void TAA::_restore_camera_jitter(ViewportRenderState &r_state) {
	if (!r_state.jittered_cam_rid.is_valid() || !VisualServer::get_singleton()) {
		r_state.jittered_cam_rid = RID();
		return;
	}
	if (r_state.jittered_was_ortho) {
		VisualServer::get_singleton()->camera_set_transform(r_state.jittered_cam_rid, r_state.jittered_base_xf);
	} else {
		VisualServer::get_singleton()->camera_set_perspective(r_state.jittered_cam_rid, r_state.jittered_fov, r_state.jittered_near, r_state.jittered_far);
	}
	r_state.jittered_cam_rid = RID();
}

void TAA::_restore_all_camera_jitters() {
	for (auto &kv : viewport_states) {
		_restore_camera_jitter(kv.second);
	}
}

void TAA::_collect_scene_snapshot() {
	SceneTree *tree = get_tree();
	Viewport *my_vp = get_viewport();
	if (!tree || !my_vp) return;

	vp_collect_counter++;
	std::vector<std::pair<Viewport *, Camera *>> active_vps;
	GL43EffectHelpers::find_active_viewports(this, my_vp, active_vps);
	if (active_vps.empty()) {
		MutexLock lock(scene_mutex);
		_restore_all_camera_jitters();
		_prune_stale_viewports();
		velocity_handler.reset_history();
		return;
	}

	Vector2 jitter_px = _compute_jitter_px(jitter_index++);
	Node *search_root = my_vp ? (Node *)my_vp : (Node *)tree->get_root();

	MutexLock lock(scene_mutex);
	velocity_handler.begin_collection();

	for (size_t vi = 0; vi < active_vps.size(); vi++) {
		Viewport *vp = active_vps[vi].first;
		Camera *vcam = active_vps[vi].second;
		Size2 vsz = vp->get_size();
		int w = (int)vsz.x;
		int h = (int)vsz.y;

		ViewportRenderState &vst = _get_or_create_vp_state(vp);
		if (vst.tex_w != w || vst.tex_h != h) {
			vst.tex_w = w;
			vst.tex_h = h;
			vst.history_valid = false;

			Ref<Image> vimg;
			vimg.instance();
			vimg->create(w, h, false, Image::FORMAT_RGBAH);
			vimg->fill(Color(0.0f, 0.0f, 0.0f, 0.0f));
			vst.velocity_texture->create_from_image(vimg, 0);

			Ref<Image> himg;
			himg.instance();
			himg->create(w, h, false, Image::FORMAT_RGBAH);
			himg->fill(Color(0.0f, 0.0f, 0.0f, 1.0f));
			vst.history_texture->create_from_image(himg, Texture::FLAG_FILTER);
		}

		if (vi == 0) {
			velocity_texture = vst.velocity_texture;
			history_texture = vst.history_texture;
		}

		_apply_camera_jitter(vst, vcam, vsz, jitter_px);
		vst.jitter_px = jitter_px;

		velocity_handler.add_viewport_target(
				vp->get_viewport_rid().get_id(),
				vcam,
				vsz,
				w,
				h,
				true,
				1.0f,
				0.0f,
				vst.velocity_texture->get_rid(),
				vp_collect_counter);
	}
	_prune_stale_viewports();

	velocity_handler.collect_scene_meshes(search_root, cull_mask, alpha_scissor_support, true, true);
	velocity_handler.end_collection(true);
}

bool TAA::_init_gl_resources() {
	if (gl_attempted) return gl_ready;
	gl_attempted = true;
	gl_thread_id = Thread::get_caller_id();

	if (!SharedGL43Context::get_singleton()->is_available()) {
		ERR_PRINT("TAA: Shared OpenGL 4.3 context not available.");
		return false;
	}

	if (!velocity_handler.init_gl("TAA")) {
		return false;
	}

	composite_program = GL43EffectHelpers::link_vf_program(GL43EffectHelpers::get_fullscreen_vert_glsl(), TAA_COMPOSITE_FRAG_GLSL, "TAA", "Composite");
	if (!composite_program) {
		return false;
	}

	gl_ready = true;
	return true;
}

void TAA::_cleanup_gl_resources() {
	if (!gl_ready || gl_thread_id != Thread::get_caller_id()) {
		return;
	}

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	velocity_handler.cleanup_gl();

	if (composite_program) { glDeleteProgram((GLuint)composite_program); composite_program = 0; }

	gl_ready = false;
	gl_attempted = false;
}

void TAA::_on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx) {
	if (!enabled || !is_inside_tree() || !gl_ready || composite_program == 0) {
		return;
	}
	if (Engine::get_singleton()->is_editor_hint() && !editor_preview) {
		return;
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	uint32_t vp_key = 0;
	RID vel_rid;
	RID hist_rid;
	bool hist_valid = false;
	Vector2 jpx;
	{
		MutexLock lock(scene_mutex);
		if (!velocity_handler.is_valid()) return;
		for (const auto &kv : viewport_states) {
			if (kv.second.color_tex_rid == p_ctx.color_tex_rid && kv.second.active_composite && kv.second.velocity_texture.is_valid() && kv.second.history_texture.is_valid()) {
				vp_key = kv.first;
				vel_rid = kv.second.velocity_texture->get_rid();
				hist_rid = kv.second.history_texture->get_rid();
				hist_valid = kv.second.history_valid;
				jpx = kv.second.jitter_px;
				break;
			}
		}
	}
	if (!vel_rid.is_valid() || !hist_rid.is_valid()) return;

	RasterizerStorageGLES3::Texture *vel_tex = rs->texture_owner.getornull(vel_rid);
	RasterizerStorageGLES3::Texture *hist_tex = rs->texture_owner.getornull(hist_rid);
	if (!vel_tex || vel_tex->tex_id == 0 || !hist_tex || hist_tex->tex_id == 0) return;

	uint32_t scene_copy_tex = GL43EffectHelpers::copy_scene_color_to_scratch(p_ctx);
	if (scene_copy_tex == 0) return;

	glUseProgram((GLuint)composite_program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, (GLuint)scene_copy_tex);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uSceneTex"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, hist_tex->tex_id);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uHistoryTex"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, vel_tex->tex_id);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uVelocityTex"), 2);

	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uHistoryValid"), hist_valid ? 1.0f : 0.0f);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uHistoryBlend"), history_blend);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uMotionRejection"), motion_rejection);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uVarianceGamma"), variance_gamma);
	glUniform1f(glGetUniformLocation((GLuint)composite_program, "uSharpenIntensity"), sharpen_intensity);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uCatmullRom"), catmull_rom_history ? 1 : 0);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uVarianceClipping"), variance_clipping ? 1 : 0);
	glUniform2f(glGetUniformLocation((GLuint)composite_program, "uJitterPx"), (float)jpx.x, (float)jpx.y);
	glUniform2f(glGetUniformLocation((GLuint)composite_program, "uPixelSize"), 1.0f / (float)p_ctx.width, 1.0f / (float)p_ctx.height);
	glUniform1i(glGetUniformLocation((GLuint)composite_program, "uDebugMode"), (int)debug_mode);

	glDisable(GL_BLEND);
	glDrawArrays(GL_TRIANGLES, 0, 3);

	if (hist_tex->width == p_ctx.width && hist_tex->height == p_ctx.height) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)p_ctx.gl43_fbo);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, hist_tex->tex_id);
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, p_ctx.width, p_ctx.height);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		MutexLock lock(scene_mutex);
		auto it = viewport_states.find(vp_key);
		if (it != viewport_states.end()) {
			it->second.history_valid = true;
		}
	}
}

void TAA::_on_frame_pre_draw() {
	if (!hook_registered || !is_inside_tree()) return;

	bool in_editor = Engine::get_singleton()->is_editor_hint();
	if (!enabled || (in_editor && !editor_preview) || !get_viewport()) {
		_restore_all_camera_jitters();
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

	velocity_handler.render_velocities(rs);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void TAA::_on_frame_post_draw() {
	if (!enabled || !is_inside_tree()) {
		return;
	}
	if (Engine::get_singleton()->is_editor_hint() && !editor_preview) {
		return;
	}
	if (OS::get_singleton()->get_render_thread_mode() != OS::RENDER_SEPARATE_THREAD) {
		MutexLock lock(scene_mutex);
		_restore_all_camera_jitters();
	}
}

void TAA::reset_history() {
	MutexLock lock(scene_mutex);
	for (auto &kv : viewport_states) {
		kv.second.history_valid = false;
	}
	jitter_index = 0;
	velocity_handler.reset_history();
}

void TAA::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	if (!enabled) {
		_restore_all_camera_jitters();
		_detach_all_viewports();
		reset_history();
	}
}

void TAA::set_editor_preview(bool p_preview) {
	editor_preview = p_preview;
	if (!editor_preview && Engine::get_singleton()->is_editor_hint()) {
		_restore_all_camera_jitters();
		_detach_all_viewports();
		reset_history();
	}
}

void TAA::set_jitter_scale(float p_scale) {
	jitter_scale = CLAMP(p_scale, 0.0f, 2.0f);
}

void TAA::set_history_blend(float p_blend) {
	history_blend = CLAMP(p_blend, 0.0f, 0.97f);
}

void TAA::set_motion_rejection(float p_rejection) {
	motion_rejection = CLAMP(p_rejection, 0.0f, 1.0f);
}

void TAA::set_variance_clipping(bool p_enable) {
	variance_clipping = p_enable;
	_change_notify();
}

void TAA::set_variance_gamma(float p_gamma) {
	variance_gamma = CLAMP(p_gamma, 0.5f, 2.5f);
}

void TAA::set_sharpen_intensity(float p_intensity) {
	sharpen_intensity = CLAMP(p_intensity, 0.0f, 1.0f);
}
