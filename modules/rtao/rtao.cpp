#include "rtao.h"

#include "baked_rtao.h"
#include "local_rtao.h"
#include "rtao_shaders.h"

#include "core/engine.h"
#include "core/math/math_funcs.h"
#include "core/project_settings.h"
#include "drivers/gles3/rasterizer_storage_gles3.h"
#include "drivers/gles3/shared_gl43_context.h"
#include "modules/modules_enabled.gen.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"

#ifdef MODULE_CSG_ENABLED
#include "modules/csg/csg_shape.h"
#endif
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/resources/environment.h"
#include "scene/resources/mesh.h"
#include "scene/resources/sky.h"
#include "scene/resources/world.h"
#include "servers/visual/visual_server_globals.h"
#include "servers/visual_server.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <array>

#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_UNIFORM_BUFFER
#define GL_UNIFORM_BUFFER 0x8A11
#endif
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
#ifndef GL_R32UI
#define GL_R32UI 0x8236
#endif
#ifndef GL_RED_INTEGER
#define GL_RED_INTEGER 0x8D94
#endif
#ifndef GL_TEXTURE_2D_ARRAY
#define GL_TEXTURE_2D_ARRAY 0x8C1A
#endif
#ifndef GL_TEXTURE_3D
#define GL_TEXTURE_3D 0x806F
#endif

namespace {

static const char *RTAO_COMPOSITE_FRAG_GLSL = R"GLSL(
#version 430 core
in vec2 vUv;
layout (location = 0) out vec4 outColor;

uniform sampler2D uAOTex;
uniform sampler2D uDepthTex;
uniform float uBilateralUpsample;
uniform int uDebugMode;
uniform mat4 uInvProj;

float linearizeDist(float rawD, vec2 uv) {
	vec4 vp = uInvProj * vec4(uv * 2.0 - 1.0, min(rawD, 0.99999) * 2.0 - 1.0, 1.0);
	return length(vp.xyz / vp.w);
}

void main() {
	vec3 aoRGB = vec3(1.0);
	bool valid = false;

	if (uBilateralUpsample > 0.5) {
		float fullD = textureLod(uDepthTex, vUv, 0.0).r;
		if (fullD < 0.99999) {
			float fullDist = linearizeDist(fullD, vUv);
			vec2 texSize = vec2(textureSize(uAOTex, 0));
			vec2 fCoord = vUv * texSize - 0.5;
			ivec2 base = ivec2(floor(fCoord));
			vec2 f = clamp(fCoord - vec2(base), vec2(0.0), vec2(1.0));
			ivec2 maxIdx = ivec2(texSize) - 1;

			vec4 s00 = texelFetch(uAOTex, clamp(base + ivec2(0, 0), ivec2(0), maxIdx), 0);
			vec4 s10 = texelFetch(uAOTex, clamp(base + ivec2(1, 0), ivec2(0), maxIdx), 0);
			vec4 s01 = texelFetch(uAOTex, clamp(base + ivec2(0, 1), ivec2(0), maxIdx), 0);
			vec4 s11 = texelFetch(uAOTex, clamp(base + ivec2(1, 1), ivec2(0), maxIdx), 0);

			float w00 = (s00.a >= 0.0) ? ((1.0 - f.x) * (1.0 - f.y) / (0.02 + abs(fullDist - s00.a) * 2.5)) : 0.0;
			float w10 = (s10.a >= 0.0) ? (      f.x  * (1.0 - f.y) / (0.02 + abs(fullDist - s10.a) * 2.5)) : 0.0;
			float w01 = (s01.a >= 0.0) ? ((1.0 - f.x) *       f.y  / (0.02 + abs(fullDist - s01.a) * 2.5)) : 0.0;
			float w11 = (s11.a >= 0.0) ? (      f.x  *       f.y  / (0.02 + abs(fullDist - s11.a) * 2.5)) : 0.0;

			float wSum = w00 + w10 + w01 + w11;
			if (wSum > 1e-5) {
				aoRGB = (s00.rgb * w00 + s10.rgb * w10 + s01.rgb * w01 + s11.rgb * w11) / wSum;
				valid = true;
			}
		}
	} else {
		vec4 sFast = textureLod(uAOTex, vUv, 0.0);
		if (sFast.a >= 0.0) {
			aoRGB = sFast.rgb;
			valid = true;
		}
	}

	if (uDebugMode != 0) {
		outColor = vec4(valid ? aoRGB : vec3(1.0), 1.0);
	} else {
		if (!valid) discard;
		outColor = vec4(aoRGB, 1.0);
	}
}
)GLSL";

} // namespace

RTAO::RTAO() {
	ao_texture.instance();
	ao_volume_texture.instance();
}

RTAO::~RTAO() {
	_unregister_scene_hook();
	_cleanup_gl_resources();
}

void RTAO::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &RTAO::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &RTAO::is_enabled);

	ClassDB::bind_method(D_METHOD("set_editor_preview", "preview"), &RTAO::set_editor_preview);
	ClassDB::bind_method(D_METHOD("get_editor_preview"), &RTAO::get_editor_preview);

	ClassDB::bind_method(D_METHOD("set_auto_composite", "auto_composite"), &RTAO::set_auto_composite);
	ClassDB::bind_method(D_METHOD("get_auto_composite"), &RTAO::get_auto_composite);

	ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &RTAO::set_intensity);
	ClassDB::bind_method(D_METHOD("get_intensity"), &RTAO::get_intensity);

	ClassDB::bind_method(D_METHOD("set_power", "power"), &RTAO::set_power);
	ClassDB::bind_method(D_METHOD("get_power"), &RTAO::get_power);

	ClassDB::bind_method(D_METHOD("set_ao_color", "color"), &RTAO::set_ao_color);
	ClassDB::bind_method(D_METHOD("get_ao_color"), &RTAO::get_ao_color);

	ClassDB::bind_method(D_METHOD("set_direct_light_preservation", "preservation"), &RTAO::set_direct_light_preservation);
	ClassDB::bind_method(D_METHOD("get_direct_light_preservation"), &RTAO::get_direct_light_preservation);

	ClassDB::bind_method(D_METHOD("set_samples_per_pixel", "samples"), &RTAO::set_samples_per_pixel);
	ClassDB::bind_method(D_METHOD("get_samples_per_pixel"), &RTAO::get_samples_per_pixel);

	ClassDB::bind_method(D_METHOD("set_ray_radius", "radius"), &RTAO::set_ray_radius);
	ClassDB::bind_method(D_METHOD("get_ray_radius"), &RTAO::get_ray_radius);

	ClassDB::bind_method(D_METHOD("set_falloff_power", "power"), &RTAO::set_falloff_power);
	ClassDB::bind_method(D_METHOD("get_falloff_power"), &RTAO::get_falloff_power);

	ClassDB::bind_method(D_METHOD("set_ray_origin_bias", "bias"), &RTAO::set_ray_origin_bias);
	ClassDB::bind_method(D_METHOD("get_ray_origin_bias"), &RTAO::get_ray_origin_bias);

	ClassDB::bind_method(D_METHOD("set_skinned_meshes", "enable"), &RTAO::set_skinned_meshes);
	ClassDB::bind_method(D_METHOD("get_skinned_meshes"), &RTAO::get_skinned_meshes);

	ClassDB::bind_method(D_METHOD("set_alpha_scissor_support", "enable"), &RTAO::set_alpha_scissor_support);
	ClassDB::bind_method(D_METHOD("get_alpha_scissor_support"), &RTAO::get_alpha_scissor_support);

	ClassDB::bind_method(D_METHOD("set_respect_material_cull_mode", "enable"), &RTAO::set_respect_material_cull_mode);
	ClassDB::bind_method(D_METHOD("get_respect_material_cull_mode"), &RTAO::get_respect_material_cull_mode);

	ClassDB::bind_method(D_METHOD("set_volume_enabled", "enable"), &RTAO::set_volume_enabled);
	ClassDB::bind_method(D_METHOD("is_volume_enabled"), &RTAO::is_volume_enabled);

	ClassDB::bind_method(D_METHOD("set_volume_extents", "extents"), &RTAO::set_volume_extents);
	ClassDB::bind_method(D_METHOD("get_volume_extents"), &RTAO::get_volume_extents);

	ClassDB::bind_method(D_METHOD("set_volume_resolution", "resolution"), &RTAO::set_volume_resolution);
	ClassDB::bind_method(D_METHOD("get_volume_resolution"), &RTAO::get_volume_resolution);

	ClassDB::bind_method(D_METHOD("set_volume_radius", "radius"), &RTAO::set_volume_radius);
	ClassDB::bind_method(D_METHOD("get_volume_radius"), &RTAO::get_volume_radius);

	ClassDB::bind_method(D_METHOD("set_volume_samples", "samples"), &RTAO::set_volume_samples);
	ClassDB::bind_method(D_METHOD("get_volume_samples"), &RTAO::get_volume_samples);

	ClassDB::bind_method(D_METHOD("set_volume_blend", "blend"), &RTAO::set_volume_blend);
	ClassDB::bind_method(D_METHOD("get_volume_blend"), &RTAO::get_volume_blend);

	ClassDB::bind_method(D_METHOD("set_volume_update_interval", "interval"), &RTAO::set_volume_update_interval);
	ClassDB::bind_method(D_METHOD("get_volume_update_interval"), &RTAO::get_volume_update_interval);

	ClassDB::bind_method(D_METHOD("set_resolution_scale", "scale"), &RTAO::set_resolution_scale);
	ClassDB::bind_method(D_METHOD("get_resolution_scale"), &RTAO::get_resolution_scale);

	ClassDB::bind_method(D_METHOD("set_checkerboard_tracing", "enable"), &RTAO::set_checkerboard_tracing);
	ClassDB::bind_method(D_METHOD("get_checkerboard_tracing"), &RTAO::get_checkerboard_tracing);

	ClassDB::bind_method(D_METHOD("set_temporal_accumulation", "enable"), &RTAO::set_temporal_accumulation);
	ClassDB::bind_method(D_METHOD("get_temporal_accumulation"), &RTAO::get_temporal_accumulation);

	ClassDB::bind_method(D_METHOD("set_temporal_blend", "blend"), &RTAO::set_temporal_blend);
	ClassDB::bind_method(D_METHOD("get_temporal_blend"), &RTAO::get_temporal_blend);

	ClassDB::bind_method(D_METHOD("set_spatial_denoise_passes", "passes"), &RTAO::set_spatial_denoise_passes);
	ClassDB::bind_method(D_METHOD("get_spatial_denoise_passes"), &RTAO::get_spatial_denoise_passes);

	ClassDB::bind_method(D_METHOD("set_bilateral_upsample", "enable"), &RTAO::set_bilateral_upsample);
	ClassDB::bind_method(D_METHOD("get_bilateral_upsample"), &RTAO::get_bilateral_upsample);

	ClassDB::bind_method(D_METHOD("set_update_interval", "interval"), &RTAO::set_update_interval);
	ClassDB::bind_method(D_METHOD("get_update_interval"), &RTAO::get_update_interval);

	ClassDB::bind_method(D_METHOD("set_blas_builds_per_frame", "budget"), &RTAO::set_blas_builds_per_frame);
	ClassDB::bind_method(D_METHOD("get_blas_builds_per_frame"), &RTAO::get_blas_builds_per_frame);

	ClassDB::bind_method(D_METHOD("set_max_tlas_instances", "max"), &RTAO::set_max_tlas_instances);
	ClassDB::bind_method(D_METHOD("get_max_tlas_instances"), &RTAO::get_max_tlas_instances);

	ClassDB::bind_method(D_METHOD("set_frustum_cull_gbuffer", "enable"), &RTAO::set_frustum_cull_gbuffer);
	ClassDB::bind_method(D_METHOD("get_frustum_cull_gbuffer"), &RTAO::get_frustum_cull_gbuffer);

	ClassDB::bind_method(D_METHOD("set_texture_array_resolution", "resolution"), &RTAO::set_texture_array_resolution);
	ClassDB::bind_method(D_METHOD("get_texture_array_resolution"), &RTAO::get_texture_array_resolution);

	ClassDB::bind_method(D_METHOD("set_max_texture_layers", "layers"), &RTAO::set_max_texture_layers);
	ClassDB::bind_method(D_METHOD("get_max_texture_layers"), &RTAO::get_max_texture_layers);

	ClassDB::bind_method(D_METHOD("set_mesh_cull_distance", "distance"), &RTAO::set_mesh_cull_distance);
	ClassDB::bind_method(D_METHOD("get_mesh_cull_distance"), &RTAO::get_mesh_cull_distance);

	ClassDB::bind_method(D_METHOD("set_mesh_cull_min_angle", "angle"), &RTAO::set_mesh_cull_min_angle);
	ClassDB::bind_method(D_METHOD("get_mesh_cull_min_angle"), &RTAO::get_mesh_cull_min_angle);

	ClassDB::bind_method(D_METHOD("set_ao_cull_mask", "mask"), &RTAO::set_ao_cull_mask);
	ClassDB::bind_method(D_METHOD("get_ao_cull_mask"), &RTAO::get_ao_cull_mask);

	ClassDB::bind_method(D_METHOD("set_receiver_cull_mask", "mask"), &RTAO::set_receiver_cull_mask);
	ClassDB::bind_method(D_METHOD("get_receiver_cull_mask"), &RTAO::get_receiver_cull_mask);

	ClassDB::bind_method(D_METHOD("set_debug_mode", "mode"), &RTAO::set_debug_mode);
	ClassDB::bind_method(D_METHOD("get_debug_mode"), &RTAO::get_debug_mode);

	ClassDB::bind_method(D_METHOD("apply_quality_preset", "preset"), &RTAO::apply_quality_preset);
	ClassDB::bind_method(D_METHOD("mark_geometry_dirty"), &RTAO::mark_geometry_dirty);
	ClassDB::bind_method(D_METHOD("reset_history"), &RTAO::reset_history);
	ClassDB::bind_method(D_METHOD("get_ao_texture"), &RTAO::get_ao_texture);
	ClassDB::bind_method(D_METHOD("get_ao_volume_texture"), &RTAO::get_ao_volume_texture);
	ClassDB::bind_method(D_METHOD("_on_frame_pre_draw"), &RTAO::_on_frame_pre_draw);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_preview"), "set_editor_preview", "get_editor_preview");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_composite"), "set_auto_composite", "get_auto_composite");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "intensity", PROPERTY_HINT_RANGE, "0,4,0.05,or_greater"), "set_intensity", "get_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "power", PROPERTY_HINT_RANGE, "0.25,4,0.05"), "set_power", "get_power");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ao_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ao_color", "get_ao_color");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "direct_light_preservation", PROPERTY_HINT_RANGE, "0,1,0.02"), "set_direct_light_preservation", "get_direct_light_preservation");

	ADD_GROUP("Ray Tracing", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "samples_per_pixel", PROPERTY_HINT_RANGE, "1,16,1"), "set_samples_per_pixel", "get_samples_per_pixel");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ray_radius", PROPERTY_HINT_RANGE, "0.1,50,0.05,or_greater"), "set_ray_radius", "get_ray_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "falloff_power", PROPERTY_HINT_RANGE, "0.1,4,0.05"), "set_falloff_power", "get_falloff_power");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "ray_origin_bias", PROPERTY_HINT_RANGE, "0.0005,0.05,0.0005"), "set_ray_origin_bias", "get_ray_origin_bias");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "skinned_meshes"), "set_skinned_meshes", "get_skinned_meshes");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "alpha_scissor_support"), "set_alpha_scissor_support", "get_alpha_scissor_support");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "respect_material_cull_mode"), "set_respect_material_cull_mode", "get_respect_material_cull_mode");

	ADD_GROUP("3D Texture Volume", "volume_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "volume_enabled"), "set_volume_enabled", "is_volume_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "volume_extents"), "set_volume_extents", "get_volume_extents");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "volume_resolution"), "set_volume_resolution", "get_volume_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_radius", PROPERTY_HINT_RANGE, "0.25,32,0.1,or_greater"), "set_volume_radius", "get_volume_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "volume_samples", PROPERTY_HINT_RANGE, "1,16,1"), "set_volume_samples", "get_volume_samples");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "volume_blend", PROPERTY_HINT_RANGE, "0,1,0.02"), "set_volume_blend", "get_volume_blend");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "volume_update_interval", PROPERTY_HINT_RANGE, "1,10,1"), "set_volume_update_interval", "get_volume_update_interval");

	ADD_GROUP("Resolution & Denoising", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "resolution_scale", PROPERTY_HINT_RANGE, "0.25,1,0.05"), "set_resolution_scale", "get_resolution_scale");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "checkerboard_tracing"), "set_checkerboard_tracing", "get_checkerboard_tracing");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "temporal_accumulation"), "set_temporal_accumulation", "get_temporal_accumulation");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "temporal_blend", PROPERTY_HINT_RANGE, "0.02,1,0.02"), "set_temporal_blend", "get_temporal_blend");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "spatial_denoise_passes", PROPERTY_HINT_RANGE, "0,5,1"), "set_spatial_denoise_passes", "get_spatial_denoise_passes");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "bilateral_upsample"), "set_bilateral_upsample", "get_bilateral_upsample");

	ADD_GROUP("Scalability & Culling", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_interval", PROPERTY_HINT_RANGE, "1,10,1"), "set_update_interval", "get_update_interval");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "blas_builds_per_frame", PROPERTY_HINT_RANGE, "0,64,1"), "set_blas_builds_per_frame", "get_blas_builds_per_frame");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_tlas_instances", PROPERTY_HINT_RANGE, "0,8192,16"), "set_max_tlas_instances", "get_max_tlas_instances");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "frustum_cull_gbuffer"), "set_frustum_cull_gbuffer", "get_frustum_cull_gbuffer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_array_resolution", PROPERTY_HINT_ENUM, "128px:128,256px:256,512px:512,1024px:1024"), "set_texture_array_resolution", "get_texture_array_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_texture_layers", PROPERTY_HINT_RANGE, "8,256,8"), "set_max_texture_layers", "get_max_texture_layers");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mesh_cull_distance", PROPERTY_HINT_RANGE, "5,500,5"), "set_mesh_cull_distance", "get_mesh_cull_distance");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mesh_cull_min_angle", PROPERTY_HINT_RANGE, "0,0.1,0.002"), "set_mesh_cull_min_angle", "get_mesh_cull_min_angle");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "ao_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_ao_cull_mask", "get_ao_cull_mask");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "receiver_cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_receiver_cull_mask", "get_receiver_cull_mask");

	ADD_GROUP("Debug", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_mode", PROPERTY_HINT_ENUM, "None,Ambient Occlusion,3D Volume AO,Bent Normal"), "set_debug_mode", "get_debug_mode");

	BIND_ENUM_CONSTANT(PRESET_LOW);
	BIND_ENUM_CONSTANT(PRESET_MEDIUM);
	BIND_ENUM_CONSTANT(PRESET_HIGH);
	BIND_ENUM_CONSTANT(PRESET_ULTRA);

	BIND_ENUM_CONSTANT(DEBUG_NONE);
	BIND_ENUM_CONSTANT(DEBUG_AO);
	BIND_ENUM_CONSTANT(DEBUG_VOLUME_3D);
	BIND_ENUM_CONSTANT(DEBUG_BENT_NORMAL);

	BIND_ENUM_CONSTANT(TEX_RES_128);
	BIND_ENUM_CONSTANT(TEX_RES_256);
	BIND_ENUM_CONSTANT(TEX_RES_512);
	BIND_ENUM_CONSTANT(TEX_RES_1024);
}

void RTAO::_validate_property(PropertyInfo &property) const {
	if (!temporal_accumulation && property.name == "temporal_blend") {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
	if (!volume_enabled &&
			(property.name == "volume_extents" ||
			 property.name == "volume_resolution" ||
			 property.name == "volume_radius" ||
			 property.name == "volume_samples" ||
			 property.name == "volume_blend" ||
			 property.name == "volume_update_interval")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
}

void RTAO::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_register_scene_hook();
			set_process(true);
			bvh_dirty = true;
			if (VisualServer::get_singleton() && !VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
				VisualServer::get_singleton()->connect("frame_pre_draw", this, "_on_frame_pre_draw");
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible_in_tree()) {
				_detach_all_viewports();
			}
		} break;

		case NOTIFICATION_EXIT_WORLD: {
			_detach_all_viewports();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (VisualServer::get_singleton() && VisualServer::get_singleton()->is_connected("frame_pre_draw", this, "_on_frame_pre_draw")) {
				VisualServer::get_singleton()->disconnect("frame_pre_draw", this, "_on_frame_pre_draw");
			}
			_unregister_scene_hook();
		} break;

		case NOTIFICATION_PROCESS: {
			noise_time += get_process_delta_time();
			if (!enabled || !is_inside_tree() || !is_visible_in_tree()) {
				_detach_all_viewports();
				MutexLock lock(scene_mutex);
				snapshot.valid = false;
				return;
			}
			if (!Engine::get_singleton()->is_editor_hint()) {
				process_frame_counter++;
				if (update_interval <= 1 || (process_frame_counter % (uint64_t)update_interval) == 0) {
					_collect_scene_snapshot();
				}
			}
		} break;
	}
}

void RTAO::_register_scene_hook() {
	if (hook_registered) {
		return;
	}
	GL43EffectHelpers::register_scene_hook(this, GL43EffectHelpers::HOOK_PRE_SCREEN_TEXTURE, 10, [this](const GL43EffectHelpers::SceneHookContext &p_ctx) {
		_on_scene_hook(p_ctx);
	});
	hook_registered = true;
}

void RTAO::_unregister_scene_hook() {
	if (!hook_registered) {
		return;
	}
	GL43EffectHelpers::unregister_scene_hooks(this);
	hook_registered = false;
	_detach_all_viewports();
}

void RTAO::_detach_all_viewports() {
	MutexLock lock(scene_mutex);
	for (auto &kv : viewport_states) {
		kv.second.active_composite = false;
		kv.second.history_valid = false;
	}
}

RTAO::ViewportRenderState &RTAO::_get_or_create_vp_state(Viewport *p_vp) {
	uint32_t key = p_vp->get_viewport_rid().get_id();
	auto it = viewport_states.find(key);
	if (it == viewport_states.end()) {
		ViewportRenderState st;
		st.ao_texture.instance();
		it = viewport_states.emplace(key, std::move(st)).first;
	}
	if (VisualServer::get_singleton()) {
		it->second.color_tex_rid = VisualServer::get_singleton()->viewport_get_texture(p_vp->get_viewport_rid(), VS::VIEWPORT_TEXTURE_BUFFER_COLOR);
	}
	it->second.active_composite = auto_composite;
	it->second.last_seen = vp_collect_counter;
	return it->second;
}

void RTAO::_prune_stale_viewports() {
	for (auto it = viewport_states.begin(); it != viewport_states.end(); ) {
		if (it->second.last_seen != vp_collect_counter) {
			it->second.active_composite = false;
			it->second.history_valid = false;
			if (vp_collect_counter - it->second.last_seen > 60ULL) {
				if (gl_ready && gl_thread_id == Thread::get_caller_id()) {
					SharedGL43Context::ScopedScope gl43;
					if (gl43.is_active()) {
						if (it->second.gbuf_fbo) glDeleteFramebuffers(1, &it->second.gbuf_fbo);
						if (it->second.gbuf_normal_mask) glDeleteTextures(1, &it->second.gbuf_normal_mask);
						if (it->second.gbuf_layer_mask) glDeleteTextures(1, &it->second.gbuf_layer_mask);
						if (it->second.gbuf_depth) glDeleteTextures(1, &it->second.gbuf_depth);
						if (it->second.ao_work_a) glDeleteTextures(1, &it->second.ao_work_a);
						if (it->second.ao_work_b) glDeleteTextures(1, &it->second.ao_work_b);
						if (it->second.ao_history_tex[0]) glDeleteTextures(1, &it->second.ao_history_tex[0]);
						if (it->second.ao_history_tex[1]) glDeleteTextures(1, &it->second.ao_history_tex[1]);
					}
				}
				it = viewport_states.erase(it);
				continue;
			}
		}
		++it;
	}
}



void RTAO::_collect_scene_snapshot() {
	SceneTree *tree = get_tree();
	Viewport *my_vp = get_viewport();
	if (!tree || !my_vp) return;

	vp_collect_counter++;
	std::vector<std::pair<Viewport *, Camera *>> active_vps;
	GL43EffectHelpers::find_active_viewports(this, my_vp, active_vps);
	if (active_vps.empty()) {
		MutexLock lock(scene_mutex);
		_prune_stale_viewports();
		snapshot.valid = false;
		return;
	}

	Viewport *active_vp = active_vps[0].first;
	Camera *cam = active_vps[0].second;
	Size2 vp_size = active_vp->get_size();
	Transform cam_xf = cam->get_camera_transform();
	Vector3 cam_pos = cam_xf.origin;
	uint32_t cam_cull_mask = cam->get_cull_mask();
	uint32_t recv_mask = receiver_cull_mask & cam_cull_mask;

	std::vector<std::array<float, 24>> all_frustums(active_vps.size());
	for (size_t vi = 0; vi < active_vps.size(); vi++) {
		Viewport *vp = active_vps[vi].first;
		Camera *vcam = active_vps[vi].second;
		CameraMatrix vview, vproj, vvp, vinv_proj, vinv_vp;
		GL43EffectHelpers::compute_camera_matrices(vcam, vp->get_size(), vview, vproj, vvp, vinv_proj, vinv_vp);
		float fplanes[6][4];
		GL43EffectHelpers::extract_frustum_planes(vvp, fplanes);
		std::memcpy(all_frustums[vi].data(), fplanes, sizeof(fplanes));
	}

	Node *search_root = my_vp ? (Node *)my_vp : (Node *)tree->get_root();
	std::vector<MeshInstance *> meshes;
	std::vector<MultiMeshInstance *> multimeshes;
	std::vector<Ref<Mesh>> extra_meshes;
	std::vector<LocalRTAO *> local_vols;
	std::vector<BakedRTAO *> baked_vols;
	std::vector<Light *> scene_lights;
#ifdef MODULE_CSG_ENABLED
	struct CsgEntry {
		CSGShape *csg;
		Transform local_xf;
		Ref<Mesh> mesh;
	};
	std::vector<CsgEntry> csg_entries;
#endif

	struct SceneCollector {
		static void visit(Node *n, uint32_t ao_m, uint32_t recv_m, uint32_t cam_m,
				std::vector<MeshInstance *> &out_m,
				std::vector<MultiMeshInstance *> &out_mm,
				std::vector<Ref<Mesh>> &out_em,
				std::vector<LocalRTAO *> &out_lv,
				std::vector<BakedRTAO *> &out_bv,
				std::vector<Light *> &out_l
#ifdef MODULE_CSG_ENABLED
				, std::vector<CsgEntry> &out_csg
#endif
				) {
			if (MeshInstance *mi = Object::cast_to<MeshInstance>(n)) {
				if (mi->is_visible_in_tree() && mi->get_mesh().is_valid()) {
					uint32_t lm = mi->get_layer_mask();
					if ((lm & ao_m) != 0 || (lm & recv_m) != 0) {
						out_m.push_back(mi);
					}
				}
			} else if (MultiMeshInstance *mmi = Object::cast_to<MultiMeshInstance>(n)) {
				if (mmi->is_visible_in_tree() && mmi->get_multimesh().is_valid() && mmi->get_multimesh()->get_mesh().is_valid()) {
					uint32_t lm = mmi->get_layer_mask();
					if ((lm & ao_m) != 0 || (lm & recv_m) != 0) {
						out_mm.push_back(mmi);
						out_em.push_back(mmi->get_multimesh()->get_mesh());
					}
				}
			}
#ifdef MODULE_CSG_ENABLED
			else if (CSGShape *csg = Object::cast_to<CSGShape>(n)) {
				if (csg->is_visible_in_tree() && csg->is_root_shape()) {
					uint32_t lm = csg->get_layer_mask();
					if ((lm & ao_m) != 0 || (lm & recv_m) != 0) {
						Array csg_meshes = csg->get_meshes();
						if (csg_meshes.size() >= 2) {
							Ref<Mesh> m = csg_meshes[1];
							if (m.is_valid()) {
								Transform lxf = csg_meshes[0];
								out_csg.push_back({ csg, lxf, m });
								out_em.push_back(m);
							}
						}
					}
				}
			}
#endif
			else if (LocalRTAO *lv = Object::cast_to<LocalRTAO>(n)) {
				if (lv->is_enabled() && lv->is_visible_in_tree() && (lv->get_cull_mask() & cam_m)) {
					out_lv.push_back(lv);
				}
			} else if (BakedRTAO *bv = Object::cast_to<BakedRTAO>(n)) {
				if (bv->is_enabled() && bv->is_visible_in_tree() && bv->get_baked_texture().is_valid() && (bv->get_cull_mask() & cam_m) && bv->get_viewing_cull_mask() != 0) {
					out_bv.push_back(bv);
				}
			} else if (Light *l = Object::cast_to<Light>(n)) {
				if (l->is_visible_in_tree() && !l->is_negative() && l->get_param(Light::PARAM_ENERGY) > 0.0001f) {
					out_l.push_back(l);
				}
			}
			for (int i = 0; i < n->get_child_count(); i++) {
				visit(n->get_child(i), ao_m, recv_m, cam_m, out_m, out_mm, out_em, out_lv, out_bv, out_l
#ifdef MODULE_CSG_ENABLED
						, out_csg
#endif
				);
			}
		}
	};

	if (search_root) {
		SceneCollector::visit(search_root, ao_cull_mask, recv_mask, cam_cull_mask, meshes, multimeshes, extra_meshes, local_vols, baked_vols, scene_lights
#ifdef MODULE_CSG_ENABLED
				, csg_entries
#endif
		);
	}

	auto any_aabb_visible = [&](const AABB &box) {
		for (size_t vi = 0; vi < all_frustums.size(); vi++) {
			float fp[6][4];
			std::memcpy(fp, all_frustums[vi].data(), sizeof(fp));
			if (GL43EffectHelpers::aabb_in_frustum(box, fp)) return true;
		}
		return false;
	};

	local_vols.erase(std::remove_if(local_vols.begin(), local_vols.end(), [&](LocalRTAO *lv) {
		return !any_aabb_visible(lv->get_world_aabb());
	}), local_vols.end());
	std::sort(local_vols.begin(), local_vols.end(), [&](LocalRTAO *a, LocalRTAO *b) {
		if (a->get_priority() != b->get_priority()) return a->get_priority() < b->get_priority();
		return a->get_global_transform().origin.distance_squared_to(cam_pos) < b->get_global_transform().origin.distance_squared_to(cam_pos);
	});

	baked_vols.erase(std::remove_if(baked_vols.begin(), baked_vols.end(), [&](BakedRTAO *bv) {
		return !any_aabb_visible(bv->get_world_aabb());
	}), baked_vols.end());
	std::sort(baked_vols.begin(), baked_vols.end(), [&](BakedRTAO *a, BakedRTAO *b) {
		return a->get_global_transform().origin.distance_squared_to(cam_pos) < b->get_global_transform().origin.distance_squared_to(cam_pos);
	});

	int vx = CLAMP((int)Math::round(volume_resolution.x), 4, 128);
	int vy = CLAMP((int)Math::round(volume_resolution.y), 4, 128);
	int vz = CLAMP((int)Math::round(volume_resolution.z), 4, 128);
	if (volume_enabled && (alloc_vol_x != vx || alloc_vol_y != vy || alloc_vol_z != vz)) {
		alloc_vol_x = vx;
		alloc_vol_y = vy;
		alloc_vol_z = vz;
		ao_volume_texture->create(vx, vy, vz, Image::FORMAT_RGBAH, Texture::FLAG_FILTER);
		for (int z = 0; z < vz; z++) {
			Ref<Image> slice;
			slice.instance();
			slice->create(vx, vy, false, Image::FORMAT_RGBAH);
			slice->fill(Color(0.0f, 1.0f, 0.0f, 1.0f));
			ao_volume_texture->set_layer_data(slice, z);
		}
	}

	MutexLock lock(scene_mutex);

	raytracer.rebuild_blas_cache(meshes, extra_meshes, GL43Raytracer::SimplificationConfig(), blas_builds_per_frame);
	pinned_3d_textures.clear();

	snapshot.viewports.clear();
	for (size_t vi = 0; vi < active_vps.size(); vi++) {
		Viewport *vp = active_vps[vi].first;
		Camera *vcam = active_vps[vi].second;
		Size2 vsz = vp->get_size();
		int tw = MAX(16, (int)Math::round(vsz.x * resolution_scale));
		int th = MAX(16, (int)Math::round(vsz.y * resolution_scale));

		ViewportRenderState &vst = _get_or_create_vp_state(vp);
		if (vst.ao_tex_w != tw || vst.ao_tex_h != th) {
			vst.ao_tex_w = tw;
			vst.ao_tex_h = th;
			vst.history_valid = false;
			Ref<Image> init_img;
			init_img.instance();
			init_img->create(tw, th, false, Image::FORMAT_RGBAH);
			init_img->fill(Color(1.0f, 1.0f, 1.0f, -1.0f));
			vst.ao_texture->create_from_image(init_img, Texture::FLAG_FILTER);
		}

		if (vi == 0) {
			ao_texture = vst.ao_texture;
		}

		Transform vxf = vcam->get_camera_transform();
		CameraMatrix v_view, v_proj, v_vp, v_inv_proj, v_inv_vp;
		GL43EffectHelpers::compute_camera_matrices(vcam, vsz, v_view, v_proj, v_vp, v_inv_proj, v_inv_vp);

		ViewportSnapshot vsnap;
		vsnap.vp_id = vp->get_viewport_rid().get_id();
		vsnap.vp_rid = vp->get_viewport_rid();
		vsnap.viewport_w = (int)vsz.x;
		vsnap.viewport_h = (int)vsz.y;
		vsnap.ao_w = tw;
		vsnap.ao_h = th;
		vsnap.cam_pos[0] = (float)vxf.origin.x;
		vsnap.cam_pos[1] = (float)vxf.origin.y;
		vsnap.cam_pos[2] = (float)vxf.origin.z;
		GL43EffectHelpers::camera_matrix_to_float16(v_vp, vsnap.view_proj_mat);
		GL43EffectHelpers::camera_matrix_to_float16(v_inv_vp, vsnap.inv_view_proj);

		std::memcpy(vsnap.cam_frustum_planes, all_frustums[vi].data(), sizeof(vsnap.cam_frustum_planes));
		snapshot.viewports.push_back(vsnap);
	}
	_prune_stale_viewports();
	Vector3 voxel_size(
			(2.0f * volume_extents.x) / (float)vx,
			(2.0f * volume_extents.y) / (float)vy,
			(2.0f * volume_extents.z) / (float)vz);
	Vector3 snapped_center(
			Math::floor(cam_pos.x / voxel_size.x) * voxel_size.x,
			Math::floor(cam_pos.y / voxel_size.y) * voxel_size.y,
			Math::floor(cam_pos.z / voxel_size.z) * voxel_size.z);
	Vector3 vol_min = snapped_center - volume_extents;
	snapshot.vol_res[0] = vx;
	snapshot.vol_res[1] = vy;
	snapshot.vol_res[2] = vz;
	snapshot.vol_min[0] = (float)vol_min.x;
	snapshot.vol_min[1] = (float)vol_min.y;
	snapshot.vol_min[2] = (float)vol_min.z;
	snapshot.vol_size[0] = (float)(volume_extents.x * 2.0f);
	snapshot.vol_size[1] = (float)(volume_extents.y * 2.0f);
	snapshot.vol_size[2] = (float)(volume_extents.z * 2.0f);
	snapshot.vol_voxel_size = MAX(voxel_size.x, MAX(voxel_size.y, voxel_size.z));

	raytracer.begin_frame_instances();
	snapshot.materials.clear();

	auto push_materials = [&](const Ref<Mesh> &mesh, auto get_mat_fn) {
		int surf_count = MAX(1, mesh->get_surface_count());
		for (int s = 0; s < surf_count; s++) {
			GpuMaterial gm = {};
			gm.uv_scale_offset[0] = 1.0f;
			gm.uv_scale_offset[1] = 1.0f;
			gm.params[0] = 1.0f;
			gm.params[1] = 0.5f;
			gm.flags[0] = -1;

			Ref<SpatialMaterial> smat = get_mat_fn(s);
			if (smat.is_valid()) {
				Vector3 uvs = smat->get_uv1_scale();
				Vector3 uvo = smat->get_uv1_offset();
				gm.uv_scale_offset[0] = (float)uvs.x;
				gm.uv_scale_offset[1] = (float)uvs.y;
				gm.uv_scale_offset[2] = (float)uvo.x;
				gm.uv_scale_offset[3] = (float)uvo.y;
				gm.params[0] = smat->get_albedo().a;
				gm.params[1] = smat->get_alpha_scissor_threshold();

				bool is_scissor = smat->get_flag(SpatialMaterial::FLAG_USE_ALPHA_SCISSOR);
				bool is_transp = smat->get_feature(SpatialMaterial::FEATURE_TRANSPARENT);
				gm.flags[1] = is_scissor ? 1 : (is_transp ? 2 : 0);
				gm.flags[2] = respect_material_cull_mode ? CLAMP((int)smat->get_cull_mode(), 0, 2) : 0;
				if (alpha_scissor_support && is_scissor) {
					gm.flags[0] = tex_pool.register_layer(smat->get_texture(SpatialMaterial::TEXTURE_ALBEDO), false, max_texture_layers);
				}
			}
			snapshot.materials.push_back(gm);
		}
	};

	for (MeshInstance *mi : meshes) {
		Ref<Mesh> mesh = mi->get_mesh();
		uint32_t mid = mesh->get_rid().get_id();
		const GL43Raytracer::MeshBLAS *blas = raytracer.get_blas(mid);
		if (!blas || (!skinned_meshes && blas->has_skin_weights)) continue;

		uint32_t lm = mi->get_layer_mask();
		bool in_ao_mask = (lm & ao_cull_mask) != 0 && mi->get_cast_shadows_setting() != GeometryInstance::SHADOW_CASTING_SETTING_OFF;
		bool in_recv_mask = (lm & recv_mask) != 0;
		bool cam_vis = (lm & cam_cull_mask) != 0;
		bool shadows_only = (mi->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		uint32_t uflags = (!in_recv_mask || shadows_only) ? INST_FLAG_SKIP_RECEIVER : 0;

		if (raytracer.add_mesh_instance(
				mi,
				(int32_t)snapshot.materials.size(),
				cam_pos,
				mesh_cull_distance,
				mesh_cull_min_angle,
				resolution_scale,
				skinned_meshes,
				in_ao_mask,
				cam_vis,
				uflags)) {
			push_materials(mesh, [&](int s) { return mi->get_active_material(s); });
		}
	}

	for (MultiMeshInstance *mmi : multimeshes) {
		Ref<MultiMesh> mm = mmi->get_multimesh();
		Ref<Mesh> mesh = mm->get_mesh();
		uint32_t mid = mesh->get_rid().get_id();
		const GL43Raytracer::MeshBLAS *blas = raytracer.get_blas(mid);
		if (!blas || (!skinned_meshes && blas->has_skin_weights)) continue;

		uint32_t lm = mmi->get_layer_mask();
		bool in_ao_mask = (lm & ao_cull_mask) != 0 && mmi->get_cast_shadows_setting() != GeometryInstance::SHADOW_CASTING_SETTING_OFF;
		bool in_recv_mask = (lm & recv_mask) != 0;
		bool cam_vis = (lm & cam_cull_mask) != 0;
		bool shadows_only = (mmi->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		uint32_t uflags = (!in_recv_mask || shadows_only) ? INST_FLAG_SKIP_RECEIVER : 0;

		if (raytracer.add_multimesh_instance(
				mmi,
				(int32_t)snapshot.materials.size(),
				cam_pos,
				mesh_cull_distance,
				mesh_cull_min_angle,
				resolution_scale,
				in_ao_mask,
				cam_vis,
				uflags)) {
			Ref<Material> override_mat = mmi->get_material_override();
			push_materials(mesh, [&](int s) { return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s); });
		}
	}

#ifdef MODULE_CSG_ENABLED
	for (const CsgEntry &entry : csg_entries) {
		CSGShape *csg = entry.csg;
		Ref<Mesh> mesh = entry.mesh;
		uint32_t mid = mesh->get_rid().get_id();
		const GL43Raytracer::MeshBLAS *blas = raytracer.get_blas(mid);
		if (!blas) continue;

		uint32_t lm = csg->get_layer_mask();
		bool in_ao_mask = (lm & ao_cull_mask) != 0 && csg->get_cast_shadows_setting() != GeometryInstance::SHADOW_CASTING_SETTING_OFF;
		bool in_recv_mask = (lm & recv_mask) != 0;
		bool cam_vis = (lm & cam_cull_mask) != 0;
		bool shadows_only = (csg->get_cast_shadows_setting() == GeometryInstance::SHADOW_CASTING_SETTING_SHADOWS_ONLY);
		uint32_t uflags = (!in_recv_mask || shadows_only) ? INST_FLAG_SKIP_RECEIVER : 0;

		if (raytracer.add_custom_mesh_instance(
				mesh,
				csg->get_global_transform() * entry.local_xf,
				lm,
				csg->get_cast_shadows_setting(),
				(int32_t)snapshot.materials.size(),
				cam_pos,
				mesh_cull_distance,
				mesh_cull_min_angle,
				resolution_scale,
				in_ao_mask,
				cam_vis,
				uflags)) {
			Ref<Material> override_mat = csg->get_material_override();
			push_materials(mesh, [&](int s) { return override_mat.is_valid() ? override_mat : mesh->surface_get_material(s); });
		}
	}
#endif

	raytracer.build_tlas(max_tlas_instances);

	// LocalRTAO Volumes
	snapshot.local_volumes.clear();
	int local_count = MIN((int)local_vols.size(), (int)MAX_LOCAL_VOLUMES);
	for (int i = 0; i < local_count; i++) {
		LocalRTAO *lv = local_vols[i];
		Transform inv = lv->get_global_transform().affine_inverse();
		Vector3 e = lv->get_extents();
		if (lv->get_shape() == LocalRTAO::SHAPE_SPHERE) {
			float r = MAX(e.x, MAX(e.y, e.z));
			e = Vector3(r, r, r);
		}
		Vector3 inv_e(1.0f / MAX(e.x, 0.0001f), 1.0f / MAX(e.y, 0.0001f), 1.0f / MAX(e.z, 0.0001f));

		GpuLocalVolume glv = {};
		GL43EffectHelpers::push_affine_row(inv, inv_e, Vector3(), 0, glv.inv_xf_row0);
		GL43EffectHelpers::push_affine_row(inv, inv_e, Vector3(), 1, glv.inv_xf_row1);
		GL43EffectHelpers::push_affine_row(inv, inv_e, Vector3(), 2, glv.inv_xf_row2);

		Color c = lv->get_ao_color().to_linear();
		glv.color_occ[0] = c.r;
		glv.color_occ[1] = c.g;
		glv.color_occ[2] = c.b;
		glv.color_occ[3] = lv->get_occlusion();
		glv.params[0] = (float)lv->get_shape();
		glv.params[1] = (float)lv->get_mode();
		glv.params[2] = (float)lv->get_blend_mode();
		glv.params[3] = lv->get_edge_falloff();
		glv.mod_params[0] = lv->get_radius_multiplier();
		glv.mod_params[1] = lv->get_intensity_multiplier();
		glv.mod_params[2] = lv->get_noise_strength();
		glv.mod_params[3] = lv->get_noise_scale();
		Vector3 noff = lv->get_noise_velocity() * noise_time;
		glv.noise_offset[0] = (float)noff.x;
		glv.noise_offset[1] = (float)noff.y;
		glv.noise_offset[2] = (float)noff.z;
		snapshot.local_volumes.push_back(glv);
	}

	// BakedRTAO 3D Texture Volumes
	snapshot.baked_volumes.clear();
	snapshot.baked_tex_rids.clear();
	int baked_count = MIN((int)baked_vols.size(), (int)MAX_BAKED_VOLUMES);
	for (int i = 0; i < baked_count; i++) {
		BakedRTAO *bv = baked_vols[i];
		Transform inv = bv->get_global_transform().affine_inverse();
		Vector3 e = bv->get_extents();
		Vector3 inv_2e(1.0f / MAX(2.0f * e.x, 0.0001f), 1.0f / MAX(2.0f * e.y, 0.0001f), 1.0f / MAX(2.0f * e.z, 0.0001f));

		GpuBakedVolume gbv = {};
		GL43EffectHelpers::push_affine_row(inv, inv_2e, Vector3(0.5f, 0.5f, 0.5f), 0, gbv.inv_xf_row0);
		GL43EffectHelpers::push_affine_row(inv, inv_2e, Vector3(0.5f, 0.5f, 0.5f), 1, gbv.inv_xf_row1);
		GL43EffectHelpers::push_affine_row(inv, inv_2e, Vector3(0.5f, 0.5f, 0.5f), 2, gbv.inv_xf_row2);

		Color bc = bv->get_ao_color().to_linear();
		gbv.color_int[0] = bc.r;
		gbv.color_int[1] = bc.g;
		gbv.color_int[2] = bc.b;
		gbv.color_int[3] = bv->get_intensity();
		gbv.params[0] = (float)bv->get_blend_mode();
		gbv.params[1] = bv->get_edge_falloff();
		gbv.masks[0] = bv->get_viewing_cull_mask();

		Ref<Texture3D> btex = bv->get_baked_texture();
		pinned_3d_textures.push_back(btex);
		snapshot.baked_tex_rids.push_back(btex->get_rid());
		snapshot.baked_volumes.push_back(gbv);
	}

	// Environment Ambient & Scene Lights for Direct Light Preservation
	Ref<World> world = active_vp->find_world();
	if (world.is_null()) world = my_vp->find_world();
	Ref<Environment> env = world.is_valid() ? world->get_environment() : Ref<Environment>();
	if (env.is_null() && world.is_valid()) env = world->get_fallback_environment();

	float amb_lum = 0.15f;
	if (env.is_valid()) {
		Color amb = env->get_ambient_light_color().to_linear();
		float amb_e = env->get_ambient_light_energy();
		float sky_c = env->get_ambient_light_sky_contribution();
		float bg_e = env->get_bg_energy();
		Color sky_avg(0.35f, 0.45f, 0.55f);
		bool has_sky_bg = false;

		if (env->get_background() == Environment::BG_SKY && env->get_sky().is_valid()) {
			has_sky_bg = true;
			Ref<ProceduralSky> psky = env->get_sky();
			if (psky.is_valid()) {
				Color st = psky->get_sky_top_color().to_linear() * psky->get_sky_energy();
				Color sh = psky->get_sky_horizon_color().to_linear() * psky->get_sky_energy();
				Color gh = psky->get_ground_horizon_color().to_linear() * psky->get_ground_energy();
				Color gb = psky->get_ground_bottom_color().to_linear() * psky->get_ground_energy();
				sky_avg = Color(
						(st.r + sh.r + gh.r + gb.r) * 0.25f * bg_e,
						(st.g + sh.g + gh.g + gb.g) * 0.25f * bg_e,
						(st.b + sh.b + gh.b + gb.b) * 0.25f * bg_e);
			} else {
				sky_avg = Color(0.5f * bg_e, 0.5f * bg_e, 0.5f * bg_e);
			}
		} else if (env->get_background() == Environment::BG_COLOR || env->get_background() == Environment::BG_CLEAR_COLOR) {
			has_sky_bg = true;
			Color bg = env->get_bg_color().to_linear();
			sky_avg = Color(bg.r * bg_e, bg.g * bg_e, bg.b * bg_e);
		} else {
			sky_c = 0.0f;
		}

		Color eff_amb = has_sky_bg ? amb.linear_interpolate(sky_avg, sky_c) : amb;
		eff_amb = Color(eff_amb.r * amb_e, eff_amb.g * amb_e, eff_amb.b * amb_e);
		amb_lum = MAX(eff_amb.r * 0.2126f + eff_amb.g * 0.7152f + eff_amb.b * 0.0722f, 0.05f);
	}
	snapshot.ambient_lum = amb_lum;
	snapshot.use_physical_atten = ProjectSettings::get_singleton()->has_setting("rendering/quality/shading/use_physical_light_attenuation")
			? bool(ProjectSettings::get_singleton()->get("rendering/quality/shading/use_physical_light_attenuation"))
			: false;

	struct LightCand {
		GpuLight gl;
		int type_order;
		float sort_dist;
	};
	std::vector<LightCand> light_cands;
	light_cands.reserve(scene_lights.size());

	for (Light *l : scene_lights) {
		if (!(l->get_cull_mask() & cam_cull_mask)) continue;
		Transform lx = l->get_global_transform();
		Color lc = l->get_color().to_linear();
		float energy = l->get_param(Light::PARAM_ENERGY);

		GpuLight gl = {};
		gl.color_atten[0] = lc.r * energy;
		gl.color_atten[1] = lc.g * energy;
		gl.color_atten[2] = lc.b * energy;
		gl.spot_params[2] = l->has_shadow() ? 1.0f : 0.0f;

		if (Object::cast_to<DirectionalLight>(l)) {
			Vector3 to_light = lx.basis.get_axis(2).normalized();
			gl.dir_type[0] = (float)to_light.x;
			gl.dir_type[1] = (float)to_light.y;
			gl.dir_type[2] = (float)to_light.z;
			gl.dir_type[3] = 0.0f;
			float max_d = l->get_param(Light::PARAM_SHADOW_MAX_DISTANCE);
			gl.spot_params[3] = (max_d > 0.1f) ? MIN(max_d, 250.0f) : 200.0f;
			light_cands.push_back({ gl, 0, 0.0f });
		} else {
			float range = MAX(l->get_param(Light::PARAM_RANGE), 0.01f);
			bool visible_in_any = false;
			for (size_t vi = 0; vi < all_frustums.size(); vi++) {
				float fp[6][4];
				std::memcpy(fp, all_frustums[vi].data(), sizeof(fp));
				if (GL43EffectHelpers::sphere_in_frustum(lx.origin, range, fp)) {
					visible_in_any = true;
					break;
				}
			}
			if (!visible_in_any) continue;

			float d = MAX(0.0f, (float)lx.origin.distance_to(cam_pos) - range);
			gl.pos_range[0] = (float)lx.origin.x;
			gl.pos_range[1] = (float)lx.origin.y;
			gl.pos_range[2] = (float)lx.origin.z;
			gl.pos_range[3] = range;
			gl.color_atten[3] = l->get_param(Light::PARAM_ATTENUATION);
			gl.spot_params[3] = range;

			if (Object::cast_to<SpotLight>(l)) {
				Vector3 beam_dir = (-lx.basis.get_axis(2)).normalized();
				gl.dir_type[0] = (float)beam_dir.x;
				gl.dir_type[1] = (float)beam_dir.y;
				gl.dir_type[2] = (float)beam_dir.z;
				gl.dir_type[3] = 2.0f;
				gl.spot_params[0] = std::cos(Math::deg2rad(l->get_param(Light::PARAM_SPOT_ANGLE)));
				gl.spot_params[1] = l->get_param(Light::PARAM_SPOT_ATTENUATION);
				light_cands.push_back({ gl, 1, d });
			} else {
				gl.dir_type[3] = 1.0f;
				light_cands.push_back({ gl, 1, d });
			}
		}
	}

	std::sort(light_cands.begin(), light_cands.end(), [](const LightCand &a, const LightCand &b) {
		if (a.type_order != b.type_order) return a.type_order < b.type_order;
		return a.sort_dist < b.sort_dist;
	});
	int light_count = MIN((int)light_cands.size(), (int)MAX_RTAO_LIGHTS);
	snapshot.lights.resize(light_count);
	for (int i = 0; i < light_count; i++) {
		snapshot.lights[i] = light_cands[i].gl;
	}

	snapshot.valid = !raytracer.is_empty();
}

bool RTAO::_init_gl_resources() {
	if (gl_attempted) return gl_ready;
	gl_attempted = true;
	gl_thread_id = Thread::get_caller_id();

	if (!SharedGL43Context::get_singleton()->is_available()) {
		ERR_PRINT("RTAO: Shared OpenGL 4.3 context not available.");
		return false;
	}

	if (!raytracer.init_gl()) {
		return false;
	}
	if (!tex_pool.init_gl("RTAO")) {
		return false;
	}

	gbuf_program = GL43EffectHelpers::link_vf_program(RTAOShaders::GBUF_VERT_GLSL, RTAOShaders::GBUF_FRAG_GLSL, "RTAO", "GBuffer");

	String vol_src = String("#version 430 core\n") + GL43Raytracer::get_glsl_common() + RTAOShaders::BVH_COMMON_GLSL + RTAOShaders::VOLUME_COMPUTE_GLSL_BODY;
	String rt_src = String("#version 430 core\n") + GL43Raytracer::get_glsl_common() + RTAOShaders::BVH_COMMON_GLSL + RTAOShaders::RT_COMPUTE_GLSL_BODY;

	volume_program = GL43EffectHelpers::link_compute_program(vol_src.utf8().get_data(), "RTAO", "Volume3D");
	volume_filter_program = GL43EffectHelpers::link_compute_program(RTAOShaders::VOLUME_FILTER_COMPUTE_GLSL, "RTAO", "Volume3DFilter");
	rt_program = GL43EffectHelpers::link_compute_program(rt_src.utf8().get_data(), "RTAO", "RaytraceAO");
	reconstruct_program = GL43EffectHelpers::link_compute_program(RTAOShaders::RECONSTRUCT_COMPUTE_GLSL, "RTAO", "Reconstruct");
	denoise_program = GL43EffectHelpers::link_compute_program(RTAOShaders::DENOISE_COMPUTE_GLSL, "RTAO", "Denoise");
	composite_program = GL43EffectHelpers::link_vf_program(GL43EffectHelpers::get_fullscreen_vert_glsl(), RTAO_COMPOSITE_FRAG_GLSL, "RTAO", "Composite");

	if (!gbuf_program || !volume_program || !volume_filter_program || !rt_program || !reconstruct_program || !denoise_program || !composite_program) {
		return false;
	}

	uint8_t white[4] = { 255, 255, 255, 255 };
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &dummy_gl_3d);
	glBindTexture(GL_TEXTURE_3D, dummy_gl_3d);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, 1, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_3D, 0);

	glGenBuffers(1, &ssbo_materials);
	glGenBuffers(1, &ssbo_local_vols);
	glGenBuffers(1, &ssbo_baked_vols);
	glGenBuffers(1, &ubo_lights);

	gl_ready = true;
	return true;
}

void RTAO::_cleanup_gl_resources() {
	if (!gl_ready || gl_thread_id != Thread::get_caller_id()) {
		return;
	}

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) {
		return;
	}

	raytracer.cleanup_gl();

	for (auto &kv : viewport_states) {
		ViewportRenderState &st = kv.second;
		if (st.gbuf_fbo) { glDeleteFramebuffers(1, &st.gbuf_fbo); st.gbuf_fbo = 0; }
		if (st.gbuf_normal_mask) { glDeleteTextures(1, &st.gbuf_normal_mask); st.gbuf_normal_mask = 0; }
		if (st.gbuf_layer_mask) { glDeleteTextures(1, &st.gbuf_layer_mask); st.gbuf_layer_mask = 0; }
		if (st.gbuf_depth) { glDeleteTextures(1, &st.gbuf_depth); st.gbuf_depth = 0; }
		if (st.ao_work_a) { glDeleteTextures(1, &st.ao_work_a); st.ao_work_a = 0; }
		if (st.ao_work_b) { glDeleteTextures(1, &st.ao_work_b); st.ao_work_b = 0; }
		if (st.ao_history_tex[0]) { glDeleteTextures(1, &st.ao_history_tex[0]); st.ao_history_tex[0] = 0; }
		if (st.ao_history_tex[1]) { glDeleteTextures(1, &st.ao_history_tex[1]); st.ao_history_tex[1] = 0; }
	}

	if (gbuf_program) { glDeleteProgram(gbuf_program); gbuf_program = 0; }
	if (volume_program) { glDeleteProgram(volume_program); volume_program = 0; }
	if (volume_filter_program) { glDeleteProgram(volume_filter_program); volume_filter_program = 0; }
	if (rt_program) { glDeleteProgram(rt_program); rt_program = 0; }
	if (reconstruct_program) { glDeleteProgram(reconstruct_program); reconstruct_program = 0; }
	if (denoise_program) { glDeleteProgram(denoise_program); denoise_program = 0; }
	if (composite_program) { glDeleteProgram(composite_program); composite_program = 0; }

	if (vol_hist_tex[0]) { glDeleteTextures(1, &vol_hist_tex[0]); vol_hist_tex[0] = 0; }
	if (vol_hist_tex[1]) { glDeleteTextures(1, &vol_hist_tex[1]); vol_hist_tex[1] = 0; }
	vol_hist_idx = 0;
	vol_hist_valid = false;
	alloc_gl_vol_x = 0;
	alloc_gl_vol_y = 0;
	alloc_gl_vol_z = 0;
	has_rendered_vol = false;

	tex_pool.cleanup_gl();
	if (dummy_gl_3d) { glDeleteTextures(1, &dummy_gl_3d); dummy_gl_3d = 0; }

	if (ssbo_materials) { glDeleteBuffers(1, &ssbo_materials); ssbo_materials = 0; }
	if (ssbo_local_vols) { glDeleteBuffers(1, &ssbo_local_vols); ssbo_local_vols = 0; }
	if (ssbo_baked_vols) { glDeleteBuffers(1, &ssbo_baked_vols); ssbo_baked_vols = 0; }
	if (ubo_lights) { glDeleteBuffers(1, &ubo_lights); ubo_lights = 0; }

	gl_ready = false;
	gl_attempted = false;
}

void RTAO::_ensure_gbuffer_size(ViewportRenderState &r_state, int p_w, int p_h) {
	if (r_state.gbuf_fbo != 0 && r_state.gbuf_w == p_w && r_state.gbuf_h == p_h) return;
	r_state.gbuf_w = p_w;
	r_state.gbuf_h = p_h;

	if (r_state.gbuf_fbo == 0) {
		glGenFramebuffers(1, &r_state.gbuf_fbo);
	}

	glActiveTexture(GL_TEXTURE0);
	GL43EffectHelpers::ensure_rgba16f_texture(r_state.gbuf_normal_mask, p_w, p_h, GL_NEAREST);
	if (r_state.gbuf_layer_mask == 0) {
		glGenTextures(1, &r_state.gbuf_layer_mask);
	}
	glBindTexture(GL_TEXTURE_2D, r_state.gbuf_layer_mask);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, p_w, p_h, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	GL43EffectHelpers::ensure_depth24_texture(r_state.gbuf_depth, p_w, p_h);
	glBindTexture(GL_TEXTURE_2D, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, r_state.gbuf_fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r_state.gbuf_normal_mask, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, r_state.gbuf_layer_mask, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, r_state.gbuf_depth, 0);
	GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, bufs);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RTAO::_ensure_vol_raw_size(int p_x, int p_y, int p_z) {
	if (vol_hist_tex[0] != 0 && vol_hist_tex[1] != 0 && alloc_gl_vol_x == p_x && alloc_gl_vol_y == p_y && alloc_gl_vol_z == p_z) {
		return;
	}
	alloc_gl_vol_x = p_x;
	alloc_gl_vol_y = p_y;
	alloc_gl_vol_z = p_z;
	vol_hist_valid = false;

	glActiveTexture(GL_TEXTURE0);
	for (int i = 0; i < 2; i++) {
		if (vol_hist_tex[i] == 0) {
			glGenTextures(1, &vol_hist_tex[i]);
		}
		glBindTexture(GL_TEXTURE_3D, vol_hist_tex[i]);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA16F, p_x, p_y, p_z, 0, GL_RGBA, GL_HALF_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	}
	glBindTexture(GL_TEXTURE_3D, 0);
}

void RTAO::_ensure_work_textures(ViewportRenderState &r_state, int p_w, int p_h, bool p_need_history) {
	bool size_changed = (r_state.work_w != p_w || r_state.work_h != p_h);
	if (size_changed) {
		r_state.work_w = p_w;
		r_state.work_h = p_h;
		r_state.history_valid = false;
	}

	glActiveTexture(GL_TEXTURE0);
	if (size_changed || r_state.ao_work_a == 0) GL43EffectHelpers::ensure_rgba16f_texture(r_state.ao_work_a, p_w, p_h, GL_LINEAR);
	if (size_changed || r_state.ao_work_b == 0) GL43EffectHelpers::ensure_rgba16f_texture(r_state.ao_work_b, p_w, p_h, GL_LINEAR);
	if (p_need_history && (size_changed || r_state.ao_history_tex[0] == 0 || r_state.ao_history_tex[1] == 0)) {
		GL43EffectHelpers::ensure_rgba16f_texture(r_state.ao_history_tex[0], p_w, p_h, GL_LINEAR);
		GL43EffectHelpers::ensure_rgba16f_texture(r_state.ao_history_tex[1], p_w, p_h, GL_LINEAR);
	}
	glBindTexture(GL_TEXTURE_2D, 0);
}

void RTAO::_on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx) {
	if (!enabled || !auto_composite || !is_inside_tree() || !is_visible_in_tree() || !gl_ready || composite_program == 0) {
		return;
	}
	if (Engine::get_singleton()->is_editor_hint() && !editor_preview) {
		return;
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	RID ao_tex_rid;
	{
		MutexLock lock(scene_mutex);
		if (!snapshot.valid) return;
		for (const auto &kv : viewport_states) {
			if (kv.second.color_tex_rid == p_ctx.color_tex_rid && kv.second.active_composite && kv.second.ao_texture.is_valid()) {
				ao_tex_rid = kv.second.ao_texture->get_rid();
				break;
			}
		}
	}
	if (!ao_tex_rid.is_valid()) return;

	RasterizerStorageGLES3::Texture *godot_ao_tex = rs->texture_owner.getornull(ao_tex_rid);
	if (!godot_ao_tex || godot_ao_tex->tex_id == 0) return;

	glUseProgram(composite_program);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, godot_ao_tex->tex_id);
	glUniform1i(glGetUniformLocation(composite_program, "uAOTex"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, p_ctx.depth_tex_id);
	glUniform1i(glGetUniformLocation(composite_program, "uDepthTex"), 1);

	glUniform1f(glGetUniformLocation(composite_program, "uBilateralUpsample"), (bilateral_upsample && resolution_scale < 0.99f) ? 1.0f : 0.0f);
	glUniform1i(glGetUniformLocation(composite_program, "uDebugMode"), (int)debug_mode);
	glUniformMatrix4fv(glGetUniformLocation(composite_program, "uInvProj"), 1, GL_FALSE, p_ctx.inv_proj_mat);

	if (debug_mode != DEBUG_NONE) {
		glDisable(GL_BLEND);
	} else {
		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);
		glBlendFunc(GL_DST_COLOR, GL_ZERO);
	}

	glDrawArrays(GL_TRIANGLES, 0, 3);
	glDisable(GL_BLEND);
}

void RTAO::_on_frame_pre_draw() {
	if (!hook_registered || !is_inside_tree()) return;

	bool in_editor = Engine::get_singleton()->is_editor_hint();
	if (!enabled || !is_visible_in_tree() || (in_editor && !editor_preview) || !get_world().is_valid()) {
		_detach_all_viewports();
		return;
	}

	if (in_editor) {
		process_frame_counter++;
		if (update_interval <= 1 || (process_frame_counter % (uint64_t)update_interval) == 0) {
			_collect_scene_snapshot();
		}
	}

	if (update_interval > 1 && (process_frame_counter % (uint64_t)update_interval) != 0) {
		return;
	}

	RasterizerStorageGLES3 *rs = dynamic_cast<RasterizerStorageGLES3 *>(VSG::storage);
	if (!rs) return;

	MutexLock lock(scene_mutex);
	if (!snapshot.valid || snapshot.viewports.empty()) return;

	RasterizerStorageGLES3::Texture *godot_vol_tex = nullptr;
	if (volume_enabled) {
		godot_vol_tex = rs->texture_owner.getornull(ao_volume_texture->get_rid());
		if (godot_vol_tex && (godot_vol_tex->tex_id == 0 || godot_vol_tex->width != snapshot.vol_res[0] || godot_vol_tex->height != snapshot.vol_res[1] || godot_vol_tex->depth != snapshot.vol_res[2])) {
			godot_vol_tex = nullptr;
		}
	}

	SharedGL43Context::ScopedScope gl43;
	if (!gl43.is_active()) return;

	if (!_init_gl_resources()) return;

	glDisable(GL_FRAMEBUFFER_SRGB);
	tex_pool.flush_blits(rs, (int)texture_array_resolution, max_texture_layers, false);

	frame_counter++;
	raytracer.sync_gpu_buffers();
	bool has_tlas = raytracer.has_tlas();

	GL43EffectHelpers::upload_ssbo(ssbo_materials, snapshot.materials.data(), snapshot.materials.size() * sizeof(GpuMaterial), sizeof(GpuMaterial));
	GL43EffectHelpers::upload_ssbo(ssbo_local_vols, snapshot.local_volumes.data(), snapshot.local_volumes.size() * sizeof(GpuLocalVolume), sizeof(GpuLocalVolume));
	GL43EffectHelpers::upload_ssbo(ssbo_baked_vols, snapshot.baked_volumes.data(), snapshot.baked_volumes.size() * sizeof(GpuBakedVolume), sizeof(GpuBakedVolume));

	raytracer.bind_ssbos(0, 1, 2, 3, 4);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, ssbo_materials);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, ssbo_local_vols);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, ssbo_baked_vols);

	std::vector<GpuLight> ubo_light_pad(MAX_RTAO_LIGHTS);
	for (size_t li = 0; li < snapshot.lights.size() && li < (size_t)MAX_RTAO_LIGHTS; li++) {
		ubo_light_pad[li] = snapshot.lights[li];
	}
	glBindBuffer(GL_UNIFORM_BUFFER, ubo_lights);
	glBufferData(GL_UNIFORM_BUFFER, ubo_light_pad.size() * sizeof(GpuLight), ubo_light_pad.data(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo_lights);

	// =========================================================
	// 1. REAL-TIME 3D AO VOLUME TEXTURE COMPUTE PASS (Shared across viewports)
	// =========================================================
	if (volume_enabled && godot_vol_tex) {
		bool vol_res_changed = (alloc_gl_vol_x != snapshot.vol_res[0] || alloc_gl_vol_y != snapshot.vol_res[1] || alloc_gl_vol_z != snapshot.vol_res[2]);
		_ensure_vol_raw_size(snapshot.vol_res[0], snapshot.vol_res[1], snapshot.vol_res[2]);
		volume_frame_counter++;
		if (!has_rendered_vol || vol_res_changed || volume_frame_counter == 1 || volume_update_interval <= 1 || (volume_frame_counter % (uint64_t)volume_update_interval) == 0) {
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_3D, godot_vol_tex->tex_id);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
			glBindTexture(GL_TEXTURE_3D, 0);

			int write_vol = vol_hist_idx;
			int read_vol = 1 - vol_hist_idx;

			SharedGL43Context *gl43_ctx = SharedGL43Context::get_singleton();
			glUseProgram(volume_program);
			gl43_ctx->bind_image_texture(0, vol_hist_tex[write_vol], 0, true, 0, GL_WRITE_ONLY, GL_RGBA16F);

			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D_ARRAY, tex_pool.get_tex_array());
			glUniform1i(glGetUniformLocation(volume_program, "uTexArray"), 0);

			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_3D, vol_hist_tex[read_vol]);
			glUniform1i(glGetUniformLocation(volume_program, "uPrevHistVolume"), 1);

			glUniform3f(glGetUniformLocation(volume_program, "uVolMin"), snapshot.vol_min[0], snapshot.vol_min[1], snapshot.vol_min[2]);
			glUniform3f(glGetUniformLocation(volume_program, "uPrevVolMin"), prev_vol_min[0], prev_vol_min[1], prev_vol_min[2]);
			glUniform3f(glGetUniformLocation(volume_program, "uVolSize"), snapshot.vol_size[0], snapshot.vol_size[1], snapshot.vol_size[2]);
			glUniform1f(glGetUniformLocation(volume_program, "uVolRadius"), volume_radius);
			glUniform1i(glGetUniformLocation(volume_program, "uVolSamples"), volume_samples);
			glUniform1f(glGetUniformLocation(volume_program, "uFalloffPower"), falloff_power);
			glUniform1f(glGetUniformLocation(volume_program, "uRayBias"), ray_origin_bias);
			glUniform1i(glGetUniformLocation(volume_program, "uHasTLAS"), has_tlas ? 1 : 0);
			glUniform1i(glGetUniformLocation(volume_program, "uRespectCullMode"), respect_material_cull_mode ? 1 : 0);
			glUniform1i(glGetUniformLocation(volume_program, "uAlphaScissorSupport"), alpha_scissor_support ? 1 : 0);
			glUniform1i(glGetUniformLocation(volume_program, "uVolHistValid"), vol_hist_valid ? 1 : 0);
			glUniform1i(glGetUniformLocation(volume_program, "uVolFrame"), (int)(vol_update_count++ & 0x7FFFFFFF));

			GLuint vgx = (GLuint)((snapshot.vol_res[0] + 3) / 4);
			GLuint vgy = (GLuint)((snapshot.vol_res[1] + 3) / 4);
			GLuint vgz = (GLuint)((snapshot.vol_res[2] + 3) / 4);
			gl43_ctx->dispatch_compute(vgx, vgy, vgz);
			gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

			glUseProgram(volume_filter_program);
			gl43_ctx->bind_image_texture(0, godot_vol_tex->tex_id, 0, true, 0, GL_WRITE_ONLY, GL_RGBA16F);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_3D, vol_hist_tex[write_vol]);
			glUniform1i(glGetUniformLocation(volume_filter_program, "uRawAOVolume"), 0);
			gl43_ctx->dispatch_compute(vgx, vgy, vgz);
			gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

			std::memcpy(prev_vol_min, snapshot.vol_min, sizeof(prev_vol_min));
			std::memcpy(rendered_vol_min, snapshot.vol_min, sizeof(rendered_vol_min));
			std::memcpy(rendered_vol_size, snapshot.vol_size, sizeof(rendered_vol_size));
			rendered_vol_voxel_size = snapshot.vol_voxel_size;
			vol_hist_idx = read_vol;
			vol_hist_valid = true;
			has_rendered_vol = true;
		}
	} else {
		vol_hist_valid = false;
		has_rendered_vol = false;
	}

	// =========================================================
	// 2. PER-VIEWPORT G-BUFFER, RAY-TRACE, RECONSTRUCT & DENOISE
	// =========================================================
	bool need_reconstruct = checkerboard_tracing || temporal_accumulation;
	bool need_denoise = (spatial_denoise_passes > 0);
	bool need_post = need_reconstruct || need_denoise;

	for (size_t vi = 0; vi < snapshot.viewports.size(); vi++) {
		const ViewportSnapshot &vsnap = snapshot.viewports[vi];
		auto st_it = viewport_states.find(vsnap.vp_id);
		if (st_it == viewport_states.end()) continue;
		ViewportRenderState &vst = st_it->second;

		RasterizerStorageGLES3::Texture *godot_ao_tex = rs->texture_owner.getornull(vst.ao_texture->get_rid());
		if (!godot_ao_tex || godot_ao_tex->tex_id == 0) continue;
		if (godot_ao_tex->width != vsnap.ao_w || godot_ao_tex->height != vsnap.ao_h) continue;

		// 2A. G-Buffer Pass for this viewport
		_ensure_gbuffer_size(vst, vsnap.ao_w, vsnap.ao_h);
		glBindFramebuffer(GL_FRAMEBUFFER, vst.gbuf_fbo);
		glViewport(0, 0, vsnap.ao_w, vsnap.ao_h);
		glFrontFace(GL_CW);
		glDisable(GL_SCISSOR_TEST);
		glDisable(GL_BLEND);
		glDisable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glDepthMask(GL_TRUE);
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
		glClearDepth(1.0);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		const GLuint zero_u[4] = { 0, 0, 0, 0 };
		glClearBufferuiv(GL_COLOR, 1, zero_u);

		glUseProgram(gbuf_program);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, tex_pool.get_tex_array());
		glUniform1i(glGetUniformLocation(gbuf_program, "uTexArray"), 0);
		glUniformMatrix4fv(glGetUniformLocation(gbuf_program, "uViewProj"), 1, GL_FALSE, vsnap.view_proj_mat);
		glUniform1i(glGetUniformLocation(gbuf_program, "uRespectCullMode"), respect_material_cull_mode ? 1 : 0);
		glUniform1i(glGetUniformLocation(gbuf_program, "uAlphaScissorSupport"), alpha_scissor_support ? 1 : 0);

		GLint loc_model = glGetUniformLocation(gbuf_program, "uModel");
		GLint loc_mat_off = glGetUniformLocation(gbuf_program, "uMatOffset");
		GLint loc_skip = glGetUniformLocation(gbuf_program, "uSkipReceiver");
		GLint loc_flip = glGetUniformLocation(gbuf_program, "uFlipFacing");
		GLint loc_layer_mask = glGetUniformLocation(gbuf_program, "uLayerMask");

		const std::vector<GL43Raytracer::InstanceEntry> &insts = raytracer.get_instances();
		for (size_t i = 0; i < insts.size(); i++) {
			const GL43Raytracer::InstanceEntry &inst = insts[i];
			if (!inst.draw_in_gbuf) continue;
			if (frustum_cull_gbuffer && !GL43EffectHelpers::aabb_in_frustum(inst.world_aabb, vsnap.cam_frustum_planes)) continue;
			uint32_t vao = 0;
			int32_t vcount = 0;
			if (!raytracer.get_instance_draw_data(i, vao, vcount)) continue;

			glUniformMatrix4fv(loc_model, 1, GL_FALSE, inst.model_mat);
			glUniform1i(loc_mat_off, inst.mat_offset);
			glUniform1i(loc_skip, (inst.user_flags & INST_FLAG_SKIP_RECEIVER) ? 1 : 0);
			glUniform1i(loc_flip, inst.flip_facing ? 1 : 0);
			glUniform1ui(loc_layer_mask, inst.layer_mask);
			glBindVertexArray((GLuint)vao);
			glDrawArrays(GL_TRIANGLES, 0, vcount);
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// 2B. Ray-Traced AO Compute Pass
		if (need_post) {
			_ensure_work_textures(vst, vsnap.ao_w, vsnap.ao_h, temporal_accumulation);
		}

		GLuint rt_target_tex = need_post ? vst.ao_work_a : godot_ao_tex->tex_id;

		SharedGL43Context *gl43_ctx = SharedGL43Context::get_singleton();
		glUseProgram(rt_program);
		gl43_ctx->bind_image_texture(0, rt_target_tex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, vst.gbuf_normal_mask);
		glUniform1i(glGetUniformLocation(rt_program, "uGNormalMask"), 0);

		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, vst.gbuf_depth);
		glUniform1i(glGetUniformLocation(rt_program, "uGDepth"), 1);

		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D_ARRAY, tex_pool.get_tex_array());
		glUniform1i(glGetUniformLocation(rt_program, "uTexArray"), 2);

		glActiveTexture(GL_TEXTURE3);
		glBindTexture(GL_TEXTURE_3D, (volume_enabled && godot_vol_tex) ? godot_vol_tex->tex_id : dummy_gl_3d);
		glUniform1i(glGetUniformLocation(rt_program, "uRealtimeVolTex"), 3);

		glActiveTexture(GL_TEXTURE8);
		glBindTexture(GL_TEXTURE_2D, vst.gbuf_layer_mask);
		glUniform1i(glGetUniformLocation(rt_program, "uGLayerMask"), 8);

		for (int b = 0; b < MAX_BAKED_VOLUMES; b++) {
			glActiveTexture(GL_TEXTURE4 + b);
			GLuint tex3d = dummy_gl_3d;
			if (b < (int)snapshot.baked_tex_rids.size() && snapshot.baked_tex_rids[b].is_valid()) {
				RasterizerStorageGLES3::Texture *bt = rs->texture_owner.getornull(snapshot.baked_tex_rids[b]);
				if (bt) bt = bt->get_ptr();
				if (bt && bt->tex_id != 0) {
					tex3d = bt->tex_id;
				}
			}
			glBindTexture(GL_TEXTURE_3D, tex3d);
			char uname[16];
			snprintf(uname, sizeof(uname), "uBakedTex%d", b);
			glUniform1i(glGetUniformLocation(rt_program, uname), 4 + b);
		}

		glUniformMatrix4fv(glGetUniformLocation(rt_program, "uInvViewProj"), 1, GL_FALSE, vsnap.inv_view_proj);
		glUniform3f(glGetUniformLocation(rt_program, "uCamPos"), vsnap.cam_pos[0], vsnap.cam_pos[1], vsnap.cam_pos[2]);
		glUniform1i(glGetUniformLocation(rt_program, "uHasTLAS"), has_tlas ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uRespectCullMode"), respect_material_cull_mode ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uAlphaScissorSupport"), alpha_scissor_support ? 1 : 0);

		glUniform1i(glGetUniformLocation(rt_program, "uSamplesPerPixel"), samples_per_pixel);
		glUniform1f(glGetUniformLocation(rt_program, "uRayRadius"), ray_radius);
		glUniform1f(glGetUniformLocation(rt_program, "uFalloffPower"), falloff_power);
		glUniform1f(glGetUniformLocation(rt_program, "uRayBias"), ray_origin_bias);
		glUniform1f(glGetUniformLocation(rt_program, "uIntensity"), intensity);
		glUniform1f(glGetUniformLocation(rt_program, "uPower"), power);
		Color ac_lin = ao_color.to_linear();
		glUniform3f(glGetUniformLocation(rt_program, "uAOColor"), ac_lin.r, ac_lin.g, ac_lin.b);

		const float *active_vol_min = has_rendered_vol ? rendered_vol_min : snapshot.vol_min;
		const float *active_vol_size = has_rendered_vol ? rendered_vol_size : snapshot.vol_size;
		float active_voxel_size = has_rendered_vol ? rendered_vol_voxel_size : snapshot.vol_voxel_size;
		glUniform1i(glGetUniformLocation(rt_program, "uVolumeEnabled"), (volume_enabled && godot_vol_tex) ? 1 : 0);
		glUniform3f(glGetUniformLocation(rt_program, "uVolMin"), active_vol_min[0], active_vol_min[1], active_vol_min[2]);
		glUniform3f(glGetUniformLocation(rt_program, "uVolSize"), active_vol_size[0], active_vol_size[1], active_vol_size[2]);
		glUniform1f(glGetUniformLocation(rt_program, "uVolVoxelSize"), active_voxel_size);
		glUniform1f(glGetUniformLocation(rt_program, "uVolBlend"), volume_blend);

		glUniform1i(glGetUniformLocation(rt_program, "uLocalCount"), (int)snapshot.local_volumes.size());
		glUniform1i(glGetUniformLocation(rt_program, "uBakedCount"), (int)snapshot.baked_volumes.size());
		glUniform1i(glGetUniformLocation(rt_program, "uNumLights"), (int)snapshot.lights.size());
		glUniform1f(glGetUniformLocation(rt_program, "uAmbientLum"), snapshot.ambient_lum);
		glUniform1f(glGetUniformLocation(rt_program, "uDirectPreservation"), direct_light_preservation);
		glUniform1i(glGetUniformLocation(rt_program, "uUsePhysicalAtten"), snapshot.use_physical_atten ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uCheckerboard"), checkerboard_tracing ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uTemporalAccum"), temporal_accumulation ? 1 : 0);
		glUniform1i(glGetUniformLocation(rt_program, "uFrameIdx"), (int)(frame_counter & 0x7FFFFFFF));
		glUniform1i(glGetUniformLocation(rt_program, "uDebugMode"), (int)debug_mode);

		GLuint gx = (GLuint)((vsnap.ao_w + 7) / 8);
		GLuint gy = (GLuint)((vsnap.ao_h + 7) / 8);
		gl43_ctx->dispatch_compute(gx, gy, 1);
		gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

		// 2C. Checkerboard / Temporal Reconstruction + Spatial Denoise
		if (need_post) {
			GLuint cur_tex = vst.ao_work_a;

			if (need_reconstruct) {
				int next_hist = 1 - vst.history_idx;
				GLuint out_hist = temporal_accumulation ? vst.ao_history_tex[next_hist] : vst.ao_work_b;
				GLuint in_hist = (temporal_accumulation && vst.history_valid) ? vst.ao_history_tex[vst.history_idx] : cur_tex;

				glUseProgram(reconstruct_program);
				gl43_ctx->bind_image_texture(0, out_hist, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
				gl43_ctx->bind_image_texture(1, need_denoise ? vst.ao_work_b : godot_ao_tex->tex_id, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

				glActiveTexture(GL_TEXTURE0);
				glBindTexture(GL_TEXTURE_2D, cur_tex);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uRawTex"), 0);

				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D, in_hist);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uHistoryTex"), 1);

				glActiveTexture(GL_TEXTURE2);
				glBindTexture(GL_TEXTURE_2D, vst.gbuf_normal_mask);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uGNormalMask"), 2);

				glActiveTexture(GL_TEXTURE3);
				glBindTexture(GL_TEXTURE_2D, vst.gbuf_depth);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uGDepth"), 3);

				glUniformMatrix4fv(glGetUniformLocation(reconstruct_program, "uInvViewProj"), 1, GL_FALSE, vsnap.inv_view_proj);
				glUniformMatrix4fv(glGetUniformLocation(reconstruct_program, "uPrevViewProj"), 1, GL_FALSE, vst.prev_view_proj);
				glUniform3f(glGetUniformLocation(reconstruct_program, "uCamPos"), vsnap.cam_pos[0], vsnap.cam_pos[1], vsnap.cam_pos[2]);
				glUniform3f(glGetUniformLocation(reconstruct_program, "uPrevCamPos"), vst.prev_cam_pos[0], vst.prev_cam_pos[1], vst.prev_cam_pos[2]);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uCheckerboard"), checkerboard_tracing ? 1 : 0);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uTemporalAccum"), temporal_accumulation ? 1 : 0);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uHistoryValid"), vst.history_valid ? 1 : 0);
				glUniform1f(glGetUniformLocation(reconstruct_program, "uTemporalBlend"), temporal_blend);
				glUniform1i(glGetUniformLocation(reconstruct_program, "uWriteFinal"), need_denoise ? 0 : 1);

				gl43_ctx->dispatch_compute(gx, gy, 1);
				gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

				if (temporal_accumulation) {
					vst.history_idx = next_hist;
					vst.history_valid = true;
					std::memcpy(vst.prev_view_proj, vsnap.view_proj_mat, sizeof(vst.prev_view_proj));
					std::memcpy(vst.prev_cam_pos, vsnap.cam_pos, sizeof(vst.prev_cam_pos));
				}
				cur_tex = out_hist;
			}

			if (need_denoise) {
				glUseProgram(denoise_program);
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_2D, vst.gbuf_normal_mask);
				glUniform1i(glGetUniformLocation(denoise_program, "uGNormalMask"), 1);

				for (int p = 0; p < spatial_denoise_passes; p++) {
					bool is_last = (p + 1 == spatial_denoise_passes);
					GLuint dst_tex = is_last ? godot_ao_tex->tex_id : ((cur_tex == vst.ao_work_a) ? vst.ao_work_b : vst.ao_work_a);
					gl43_ctx->bind_image_texture(0, dst_tex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

					glActiveTexture(GL_TEXTURE0);
					glBindTexture(GL_TEXTURE_2D, cur_tex);
					glUniform1i(glGetUniformLocation(denoise_program, "uSrcTex"), 0);
					glUniform1i(glGetUniformLocation(denoise_program, "uStepSize"), 1 << p);

					gl43_ctx->dispatch_compute(gx, gy, 1);
					gl43_ctx->memory_barrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
					cur_tex = dst_tex;
				}
			}
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, godot_ao_tex->tex_id);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	for (int i = 0; i < 8; i++) {
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (GLuint)i, 0);
	}
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	glBindBufferBase(GL_UNIFORM_BUFFER, 0, 0);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	for (int i = 0; i < 9; i++) {
		glActiveTexture(GL_TEXTURE0 + i);
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
		glBindTexture(GL_TEXTURE_3D, 0);
	}
	glActiveTexture(GL_TEXTURE0);
	glBindVertexArray(0);
	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RTAO::reset_history() {
	for (auto &kv : viewport_states) {
		kv.second.history_valid = false;
	}
}

void RTAO::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	if (!enabled) {
		_detach_all_viewports();
		reset_history();
	}
}

void RTAO::set_editor_preview(bool p_preview) {
	editor_preview = p_preview;
	if (!editor_preview && Engine::get_singleton()->is_editor_hint()) {
		_detach_all_viewports();
		reset_history();
	}
}

void RTAO::set_auto_composite(bool p_auto) {
	auto_composite = p_auto;
	if (!auto_composite) {
		_detach_all_viewports();
	}
}

void RTAO::set_intensity(float p_intensity) {
	intensity = MAX(p_intensity, 0.0f);
}

void RTAO::set_power(float p_power) {
	power = MAX(p_power, 0.1f);
}

void RTAO::set_direct_light_preservation(float p_val) {
	direct_light_preservation = CLAMP(p_val, 0.0f, 1.0f);
}

void RTAO::set_samples_per_pixel(int p_samples) {
	samples_per_pixel = CLAMP(p_samples, 1, 16);
}

void RTAO::set_ray_radius(float p_radius) {
	ray_radius = MAX(p_radius, 0.05f);
}

void RTAO::set_falloff_power(float p_power) {
	falloff_power = MAX(p_power, 0.05f);
}

void RTAO::set_ray_origin_bias(float p_bias) {
	ray_origin_bias = CLAMP(p_bias, 0.0005f, 0.05f);
}

void RTAO::set_volume_enabled(bool p_enable) {
	volume_enabled = p_enable;
	_change_notify();
}

void RTAO::set_volume_extents(const Vector3 &p_extents) {
	volume_extents = Vector3(MAX(p_extents.x, 1.0f), MAX(p_extents.y, 1.0f), MAX(p_extents.z, 1.0f));
}

void RTAO::set_volume_resolution(const Vector3 &p_res) {
	volume_resolution = Vector3(
			CLAMP((int)Math::round(p_res.x), 4, 128),
			CLAMP((int)Math::round(p_res.y), 4, 128),
			CLAMP((int)Math::round(p_res.z), 4, 128));
}

void RTAO::set_volume_radius(float p_radius) {
	volume_radius = MAX(p_radius, 0.1f);
}

void RTAO::set_volume_samples(int p_samples) {
	volume_samples = CLAMP(p_samples, 1, 16);
}

void RTAO::set_volume_blend(float p_blend) {
	volume_blend = CLAMP(p_blend, 0.0f, 1.0f);
}

void RTAO::set_volume_update_interval(int p_interval) {
	volume_update_interval = CLAMP(p_interval, 1, 10);
}

void RTAO::set_resolution_scale(float p_scale) {
	float clamped = CLAMP(p_scale, 0.25f, 1.0f);
	if (std::abs(clamped - resolution_scale) > 0.001f) {
		resolution_scale = clamped;
		reset_history();
	}
}

void RTAO::set_temporal_accumulation(bool p_enable) {
	if (temporal_accumulation != p_enable) {
		temporal_accumulation = p_enable;
		reset_history();
		_change_notify();
	}
}

void RTAO::set_temporal_blend(float p_blend) {
	temporal_blend = CLAMP(p_blend, 0.02f, 1.0f);
}

void RTAO::set_spatial_denoise_passes(int p_passes) {
	spatial_denoise_passes = CLAMP(p_passes, 0, 5);
}

void RTAO::set_update_interval(int p_interval) {
	update_interval = CLAMP(p_interval, 1, 10);
}

void RTAO::set_blas_builds_per_frame(int p_budget) {
	blas_builds_per_frame = CLAMP(p_budget, 0, 64);
}

void RTAO::set_max_tlas_instances(int p_max) {
	max_tlas_instances = CLAMP(p_max, 0, 8192);
}

void RTAO::set_texture_array_resolution(TextureResolution p_res) {
	if (texture_array_resolution != p_res) {
		texture_array_resolution = p_res;
		tex_pool.mark_reset_needed();
	}
}

void RTAO::set_max_texture_layers(int p_layers) {
	int clamped = CLAMP(p_layers, 8, 256);
	if (max_texture_layers != clamped) {
		max_texture_layers = clamped;
		tex_pool.mark_reset_needed();
	}
}

void RTAO::set_mesh_cull_distance(float p_dist) {
	mesh_cull_distance = MAX(p_dist, 1.0f);
}

void RTAO::set_mesh_cull_min_angle(float p_angle) {
	mesh_cull_min_angle = CLAMP(p_angle, 0.0f, 0.2f);
}

void RTAO::apply_quality_preset(QualityPreset p_preset) {
	switch (p_preset) {
		case PRESET_LOW: {
			resolution_scale = 0.25f;
			samples_per_pixel = 1;
			checkerboard_tracing = true;
			temporal_accumulation = true;
			temporal_blend = 0.25f;
			spatial_denoise_passes = 1;
			bilateral_upsample = false;
			volume_enabled = true;
			volume_resolution = Vector3(16, 8, 16);
			volume_samples = 2;
			volume_update_interval = 4;
			mesh_cull_distance = 25.0f;
			mesh_cull_min_angle = 0.025f;
		} break;
		case PRESET_MEDIUM: {
			resolution_scale = 0.5f;
			samples_per_pixel = 2;
			checkerboard_tracing = false;
			temporal_accumulation = true;
			temporal_blend = 0.2f;
			spatial_denoise_passes = 1;
			bilateral_upsample = true;
			volume_enabled = true;
			volume_resolution = Vector3(32, 16, 32);
			volume_samples = 4;
			volume_update_interval = 2;
			mesh_cull_distance = 45.0f;
			mesh_cull_min_angle = 0.012f;
		} break;
		case PRESET_HIGH: {
			resolution_scale = 0.75f;
			samples_per_pixel = 4;
			checkerboard_tracing = false;
			temporal_accumulation = true;
			temporal_blend = 0.15f;
			spatial_denoise_passes = 2;
			bilateral_upsample = true;
			volume_enabled = true;
			volume_resolution = Vector3(48, 24, 48);
			volume_samples = 6;
			volume_update_interval = 1;
			mesh_cull_distance = 80.0f;
			mesh_cull_min_angle = 0.008f;
		} break;
		case PRESET_ULTRA: {
			resolution_scale = 1.0f;
			samples_per_pixel = 8;
			checkerboard_tracing = false;
			temporal_accumulation = true;
			temporal_blend = 0.12f;
			spatial_denoise_passes = 2;
			bilateral_upsample = true;
			volume_enabled = true;
			volume_resolution = Vector3(64, 32, 64);
			volume_samples = 8;
			volume_update_interval = 1;
			mesh_cull_distance = 150.0f;
			mesh_cull_min_angle = 0.0f;
		} break;
	}
	reset_history();
	_change_notify();
}
