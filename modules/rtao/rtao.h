#ifndef RTAO_H
#define RTAO_H

#include "core/math/aabb.h"
#include "core/math/transform.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "drivers/gles3/gl43_bvh_helper.h"
#include "drivers/gles3/gl43_effect_helpers.h"
#include "drivers/gles3/gl43_raytracer.h"
#include "scene/3d/spatial.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/texture.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

class Camera;
class Viewport;
class MeshInstance;
class MultiMeshInstance;
class LocalRTAO;
class BakedRTAO;

class RTAO : public Spatial {
	GDCLASS(RTAO, Spatial);

public:
	enum QualityPreset {
		PRESET_LOW = 0,
		PRESET_MEDIUM = 1,
		PRESET_HIGH = 2,
		PRESET_ULTRA = 3,
	};

	enum DebugMode {
		DEBUG_NONE = 0,
		DEBUG_AO = 1,
		DEBUG_VOLUME_3D = 2,
		DEBUG_BENT_NORMAL = 3,
	};

	enum TextureResolution {
		TEX_RES_128 = 128,
		TEX_RES_256 = 256,
		TEX_RES_512 = 512,
		TEX_RES_1024 = 1024,
	};

	enum {
		MAX_LOCAL_VOLUMES = 16,
		MAX_BAKED_VOLUMES = 4,
		MAX_RTAO_LIGHTS = 32,
	};

	struct GpuMaterial {
		float uv_scale_offset[4];
		float params[4];    // x = alpha_base, y = alpha_scissor, z = 0, w = 0
		int32_t flags[4];   // x = albedo_layer, y = alpha_mode, z = cull_mode, w = 0
	};

	struct GpuLocalVolume {
		float inv_xf_row0[4];
		float inv_xf_row1[4];
		float inv_xf_row2[4];
		float color_occ[4];     // rgb = ao_color, w = occlusion
		float params[4];        // shape, mode, blend_mode, edge_falloff
		float mod_params[4];    // radius_mult, intensity_mult, noise_strength, noise_scale
		float noise_offset[4];  // xyz, 0
	};

	struct GpuBakedVolume {
		float inv_xf_row0[4];
		float inv_xf_row1[4];
		float inv_xf_row2[4];
		float color_int[4]; // rgb = ao_color, w = intensity
		float params[4];    // blend_mode, edge_falloff, 0, 0
		uint32_t masks[4];  // x = viewing_cull_mask
	};

	struct GpuLight {
		float pos_range[4];   // xyz = world pos, w = range
		float dir_type[4];    // xyz = dir (toward light for dir, forward beam for spot), w = type (0=dir, 1=omni, 2=spot)
		float color_atten[4]; // rgb = linear color * energy, w = distance attenuation
		float spot_params[4]; // x = spot_cutoff (cos), y = spot_atten, z = has_shadow (0/1), w = shadow_max_dist
	};

private:
	enum InstanceUserFlags {
		INST_FLAG_SKIP_RECEIVER = 1 << 0,
	};

	struct ViewportSnapshot {
		uint32_t vp_id = 0;
		RID vp_rid;
		int viewport_w = 0;
		int viewport_h = 0;
		int ao_w = 0;
		int ao_h = 0;
		float cam_pos[3] = { 0, 0, 0 };
		float view_proj_mat[16] = {};
		float inv_view_proj[16] = {};
		float cam_frustum_planes[6][4] = {};
	};

	struct ViewportRenderState {
		RID color_tex_rid;
		bool active_composite = false;
		Ref<ImageTexture> ao_texture;
		int ao_tex_w = 0;
		int ao_tex_h = 0;

		uint32_t gbuf_fbo = 0;
		uint32_t gbuf_normal_mask = 0;
		uint32_t gbuf_layer_mask = 0;
		uint32_t gbuf_depth = 0;
		int gbuf_w = 0;
		int gbuf_h = 0;

		uint32_t ao_work_a = 0;
		uint32_t ao_work_b = 0;
		uint32_t ao_history_tex[2] = { 0, 0 };
		int work_w = 0;
		int work_h = 0;

		bool history_valid = false;
		int history_idx = 0;
		float prev_view_proj[16] = {};
		float prev_cam_pos[3] = { 0.0f, 0.0f, 0.0f };
		uint64_t last_seen = 0;
	};

	struct SceneSnapshot {
		bool valid = false;
		std::vector<ViewportSnapshot> viewports;

		int vol_res[3] = { 32, 16, 32 };
		float vol_min[3] = { 0, 0, 0 };
		float vol_size[3] = { 48, 24, 48 };
		float vol_voxel_size = 1.5f;

		std::vector<GpuMaterial> materials;
		float ambient_lum = 0.15f;
		bool use_physical_atten = false;
		std::vector<GpuLight> lights;
		std::vector<GpuLocalVolume> local_volumes;
		std::vector<GpuBakedVolume> baked_volumes;
		std::vector<RID> baked_tex_rids;
	};

	// Core Properties
	bool enabled = true;
	bool editor_preview = true;
	bool auto_composite = true;
	float intensity = 1.0f;
	float power = 1.0f;
	Color ao_color = Color(0.0f, 0.0f, 0.0f, 1.0f);
	float direct_light_preservation = 0.5f;

	// Ray Tracing
	int samples_per_pixel = 2;
	float ray_radius = 2.5f;
	float falloff_power = 1.0f;
	float ray_origin_bias = 0.004f;
	bool skinned_meshes = true;
	bool alpha_scissor_support = true;
	bool respect_material_cull_mode = true;

	// 3D Texture Volume
	bool volume_enabled = true;
	Vector3 volume_extents = Vector3(24.0f, 12.0f, 24.0f);
	Vector3 volume_resolution = Vector3(32, 16, 32);
	float volume_radius = 4.0f;
	int volume_samples = 4;
	float volume_blend = 0.35f;
	int volume_update_interval = 2;

	// Resolution & Denoising
	float resolution_scale = 0.5f;
	bool checkerboard_tracing = false;
	bool temporal_accumulation = true;
	float temporal_blend = 0.2f;
	int spatial_denoise_passes = 1;
	bool bilateral_upsample = true;

	// Scalability & Culling
	int update_interval = 1;
	int blas_builds_per_frame = 0;
	int max_tlas_instances = 0;
	bool frustum_cull_gbuffer = true;
	TextureResolution texture_array_resolution = TEX_RES_256;
	int max_texture_layers = 64;
	float mesh_cull_distance = 45.0f;
	float mesh_cull_min_angle = 0.012f;
	uint32_t ao_cull_mask = 0xFFFFFFFF;
	uint32_t receiver_cull_mask = 0xFFFFFFFF;

	// Debug
	DebugMode debug_mode = DEBUG_NONE;

	// Exported Textures & Per-Viewport State
	Ref<ImageTexture> ao_texture;
	Ref<Texture3D> ao_volume_texture;
	std::unordered_map<uint32_t, ViewportRenderState> viewport_states;
	uint64_t vp_collect_counter = 0;
	bool hook_registered = false;

	// Runtime State & Caches
	Mutex scene_mutex;
	SceneSnapshot snapshot;
	GL43Raytracer raytracer;
	GL43TextureArrayPool tex_pool;
	std::vector<Ref<Texture3D>> pinned_3d_textures;

	uint64_t frame_counter = 0;
	uint64_t process_frame_counter = 0;
	uint64_t volume_frame_counter = 0;
	float noise_time = 0.0f;
	bool bvh_dirty = true;

	int alloc_vol_x = 0;
	int alloc_vol_y = 0;
	int alloc_vol_z = 0;

	// OpenGL 4.3 Resources
	bool gl_attempted = false;
	bool gl_ready = false;
	Thread::ID gl_thread_id = 0;

	uint32_t gbuf_program = 0;
	uint32_t volume_program = 0;
	uint32_t volume_filter_program = 0;
	uint32_t rt_program = 0;
	uint32_t reconstruct_program = 0;
	uint32_t denoise_program = 0;
	uint32_t composite_program = 0;

	uint32_t vol_hist_tex[2] = { 0, 0 };
	int vol_hist_idx = 0;
	bool vol_hist_valid = false;
	uint32_t vol_update_count = 0;
	int alloc_gl_vol_x = 0;
	int alloc_gl_vol_y = 0;
	int alloc_gl_vol_z = 0;
	bool has_rendered_vol = false;
	float prev_vol_min[3] = { 0.0f, 0.0f, 0.0f };
	float rendered_vol_min[3] = { 0.0f, 0.0f, 0.0f };
	float rendered_vol_size[3] = { 48.0f, 24.0f, 48.0f };
	float rendered_vol_voxel_size = 1.5f;


	uint32_t dummy_gl_3d = 0;

	uint32_t ssbo_materials = 0;
	uint32_t ssbo_local_vols = 0;
	uint32_t ssbo_baked_vols = 0;
	uint32_t ubo_lights = 0;

	void _register_scene_hook();
	void _unregister_scene_hook();
	void _detach_all_viewports();
	ViewportRenderState &_get_or_create_vp_state(Viewport *p_vp);
	void _prune_stale_viewports();
	void _on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx);

	bool _init_gl_resources();
	void _cleanup_gl_resources();
	void _ensure_gbuffer_size(ViewportRenderState &r_state, int p_w, int p_h);
	void _ensure_vol_raw_size(int p_x, int p_y, int p_z);
	void _ensure_work_textures(ViewportRenderState &r_state, int p_w, int p_h, bool p_need_history);

	void _collect_scene_snapshot();
	void _on_frame_pre_draw();

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const;

public:
	RTAO();
	~RTAO();

	// Core
	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_editor_preview(bool p_preview);
	bool get_editor_preview() const { return editor_preview; }

	void set_auto_composite(bool p_auto);
	bool get_auto_composite() const { return auto_composite; }

	void set_intensity(float p_intensity);
	float get_intensity() const { return intensity; }

	void set_power(float p_power);
	float get_power() const { return power; }

	void set_ao_color(const Color &p_color) { ao_color = p_color; }
	Color get_ao_color() const { return ao_color; }

	void set_direct_light_preservation(float p_val);
	float get_direct_light_preservation() const { return direct_light_preservation; }

	// Ray Tracing
	void set_samples_per_pixel(int p_samples);
	int get_samples_per_pixel() const { return samples_per_pixel; }

	void set_ray_radius(float p_radius);
	float get_ray_radius() const { return ray_radius; }

	void set_falloff_power(float p_power);
	float get_falloff_power() const { return falloff_power; }

	void set_ray_origin_bias(float p_bias);
	float get_ray_origin_bias() const { return ray_origin_bias; }

	void set_skinned_meshes(bool p_enable) { skinned_meshes = p_enable; }
	bool get_skinned_meshes() const { return skinned_meshes; }

	void set_alpha_scissor_support(bool p_enable) { alpha_scissor_support = p_enable; }
	bool get_alpha_scissor_support() const { return alpha_scissor_support; }

	void set_respect_material_cull_mode(bool p_enable) { respect_material_cull_mode = p_enable; }
	bool get_respect_material_cull_mode() const { return respect_material_cull_mode; }

	// 3D Texture Volume
	void set_volume_enabled(bool p_enable);
	bool is_volume_enabled() const { return volume_enabled; }

	void set_volume_extents(const Vector3 &p_extents);
	Vector3 get_volume_extents() const { return volume_extents; }

	void set_volume_resolution(const Vector3 &p_res);
	Vector3 get_volume_resolution() const { return volume_resolution; }

	void set_volume_radius(float p_radius);
	float get_volume_radius() const { return volume_radius; }

	void set_volume_samples(int p_samples);
	int get_volume_samples() const { return volume_samples; }

	void set_volume_blend(float p_blend);
	float get_volume_blend() const { return volume_blend; }

	void set_volume_update_interval(int p_interval);
	int get_volume_update_interval() const { return volume_update_interval; }

	// Resolution & Denoising
	void set_resolution_scale(float p_scale);
	float get_resolution_scale() const { return resolution_scale; }

	void set_checkerboard_tracing(bool p_enable) { checkerboard_tracing = p_enable; }
	bool get_checkerboard_tracing() const { return checkerboard_tracing; }

	void set_temporal_accumulation(bool p_enable);
	bool get_temporal_accumulation() const { return temporal_accumulation; }

	void set_temporal_blend(float p_blend);
	float get_temporal_blend() const { return temporal_blend; }

	void set_spatial_denoise_passes(int p_passes);
	int get_spatial_denoise_passes() const { return spatial_denoise_passes; }

	void set_bilateral_upsample(bool p_enable) { bilateral_upsample = p_enable; }
	bool get_bilateral_upsample() const { return bilateral_upsample; }

	// Scalability & Culling
	void set_update_interval(int p_interval);
	int get_update_interval() const { return update_interval; }

	void set_blas_builds_per_frame(int p_budget);
	int get_blas_builds_per_frame() const { return blas_builds_per_frame; }

	void set_max_tlas_instances(int p_max);
	int get_max_tlas_instances() const { return max_tlas_instances; }

	void set_frustum_cull_gbuffer(bool p_enable) { frustum_cull_gbuffer = p_enable; }
	bool get_frustum_cull_gbuffer() const { return frustum_cull_gbuffer; }

	void set_texture_array_resolution(TextureResolution p_res);
	TextureResolution get_texture_array_resolution() const { return texture_array_resolution; }

	void set_max_texture_layers(int p_layers);
	int get_max_texture_layers() const { return max_texture_layers; }

	void set_mesh_cull_distance(float p_dist);
	float get_mesh_cull_distance() const { return mesh_cull_distance; }

	void set_mesh_cull_min_angle(float p_angle);
	float get_mesh_cull_min_angle() const { return mesh_cull_min_angle; }

	void set_ao_cull_mask(uint32_t p_mask) { ao_cull_mask = p_mask; }
	uint32_t get_ao_cull_mask() const { return ao_cull_mask; }

	void set_receiver_cull_mask(uint32_t p_mask) { receiver_cull_mask = p_mask; }
	uint32_t get_receiver_cull_mask() const { return receiver_cull_mask; }

	void set_debug_mode(DebugMode p_mode) { debug_mode = p_mode; }
	DebugMode get_debug_mode() const { return debug_mode; }

	void apply_quality_preset(QualityPreset p_preset);
	void mark_geometry_dirty() { bvh_dirty = true; raytracer.invalidate_blas_cache(); }
	void reset_history();

	Ref<Texture> get_ao_texture() const { return ao_texture; }
	Ref<Texture3D> get_ao_volume_texture() const { return ao_volume_texture; }
};

VARIANT_ENUM_CAST(RTAO::QualityPreset);
VARIANT_ENUM_CAST(RTAO::DebugMode);
VARIANT_ENUM_CAST(RTAO::TextureResolution);

#endif // RTAO_H
