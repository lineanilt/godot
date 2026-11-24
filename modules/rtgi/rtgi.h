#ifndef RTGI_H
#define RTGI_H

#include "core/math/aabb.h"
#include "core/math/camera_matrix.h"
#include "core/math/transform.h"
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "drivers/gles3/gl43_bvh_helper.h"
#include "drivers/gles3/gl43_effect_helpers.h"
#include "drivers/gles3/gl43_raytracer.h"
#include "scene/main/node.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/sky.h"
#include "scene/resources/texture.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

class Camera;
class Viewport;
class MeshInstance;

class RTGI : public Node {
	GDCLASS(RTGI, Node);

public:
	enum QualityPreset {
		PRESET_LOW = 0,
		PRESET_MEDIUM = 1,
		PRESET_HIGH = 2,
		PRESET_ULTRA = 3,
	};

	enum CustomSkyMode {
		SKY_MODE_ENVIRONMENT = 0,
		SKY_MODE_CUSTOM_SKY = 1,
		SKY_MODE_COLOR = 2,
	};

	enum SimplificationMode {
		SIMPLIFY_QEM = 0,
		SIMPLIFY_VERTEX_CLUSTERING = 1,
	};

	enum TextureResolution {
		TEX_RES_128 = 128,
		TEX_RES_256 = 256,
		TEX_RES_512 = 512,
		TEX_RES_1024 = 1024,
	};

	enum DebugMode {
		DEBUG_NONE = 0,
		DEBUG_IRRADIANCE = 1,
		DEBUG_GI_ONLY = 2,
		DEBUG_ALBEDO = 3,
		DEBUG_VOLUME_3D = 4,
	};

	// 7 x 16 bytes = 112 bytes (std430 aligned)
	struct GpuMaterial {
		float albedo[4];            // rgb = linear albedo, a = base_alpha
		float emission[4];          // rgb = linear emission * energy, a = metallic
		float pbr_params[4];        // x = roughness, y = alpha_scissor, z = specular, w = ao_light_affect
		float uv_scale_offset[4];   // xy = uv1_scale, zw = uv1_offset
		int32_t tex_layers[4];      // x = albedo, y = roughness, z = metallic, w = emission
		int32_t chan_normal[4];     // x = rough_chan, y = metal_chan, z = normal_layer, w = float_bits(normal_scale)
		int32_t extra_tex[4];       // x = ao_layer, y = ao_chan, z = alpha_mode, w = flags
	};

	struct GpuLight {
		float pos_range[4];
		float dir_type[4];
		float color_atten[4];
		float spot_params[4];
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
		int gi_w = 0;
		int gi_h = 0;
		float cam_pos[3] = { 0, 0, 0 };
		float view_proj_mat[16] = {};
		float inv_view_proj[16] = {};
		float inv_proj_mat[16] = {};
		float cam_frustum_planes[6][4] = {};

		int tonemap_mode = 0;
		float tonemap_exposure = 1.0f;
		float tonemap_white = 1.0f;
		bool hdr = true;
		bool keep_3d_linear = false;
		bool vflip = false;
	};

	struct ViewportRenderState {
		GL43CanvasOverlay overlay;
		Ref<ImageTexture> gi_texture;
		Ref<ImageTexture> gi_albedo_texture;
		Ref<ImageTexture> gi_normal_texture;
		int tex_w = 0;
		int tex_h = 0;
		int full_tex_w = 0;
		int full_tex_h = 0;

		uint32_t gbuf_fbo = 0;
		uint32_t gbuf_depth = 0;
		int gbuf_w = 0;
		int gbuf_h = 0;

		uint32_t low_normal_mask = 0;
		uint32_t low_depth = 0;
		int low_w = 0;
		int low_h = 0;

		uint32_t gi_work_a = 0;
		uint32_t gi_work_b = 0;
		uint32_t gi_history_tex[2] = { 0, 0 };
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

		int vol_res[3] = { 24, 12, 24 };
		float vol_min[3] = { 0, 0, 0 };
		float vol_size[3] = { 48, 24, 48 };
		float vol_voxel_size = 2.0f;

		float sky_top[3] = { 0.376f, 0.672f, 0.880f };
		float sky_horizon[3] = { 0.672f, 0.823f, 0.956f };
		float sky_ground_horizon[3] = { 0.153f, 0.127f, 0.105f };
		float sky_ground[3] = { 0.021f, 0.018f, 0.014f };
		float sky_curve = 0.09f;
		float ground_curve = 0.02f;
		float sky_energy = 1.0f;
		float sky_sun_dir[3] = { 0.0f, 0.5736f, -0.8192f };
		float sky_sun_color[3] = { 1.0f, 1.0f, 1.0f };
		float sky_sun_angle_min = 1.0f;
		float sky_sun_angle_max = 100.0f;
		float sky_sun_curve = 0.05f;
		float sky_sun_energy = 1.0f;
		bool use_panorama = false;
		RID panorama_rid;
		uint64_t sky_hash = 0;
		float sky_inv_basis[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

		std::vector<GpuMaterial> materials;
		std::vector<GpuLight> lights;
	};

	// Core
	bool enabled = true;
	bool editor_preview = true;
	bool auto_composite = true;
	float gi_intensity = 1.0f;
	uint32_t gi_cull_mask = 0xFFFFFFFF;
	uint32_t gi_viewing_cull_mask = 0xFFFFFFFF;

	// Resolution & Denoising
	float resolution_scale = 0.5f;
	bool bilateral_upsample = true;
	bool checkerboard_tracing = false;
	bool temporal_accumulation = true;
	float temporal_blend = 0.12f;
	bool temporal_noise = true;
	int spatial_denoise_passes = 3;
	float firefly_clamp = 10.0f;

	// Tracing & Bounces
	int rays_per_pixel = 2;
	int max_bounces = 2;
	float max_ray_distance = 60.0f;
	float bounce_distance_falloff = 0.75f;
	float bounce_energy_boost = 1.0f;
	float bounce_throughput_threshold = 0.02f;
	float emissive_intensity = 1.0f;
	bool bounce_local_lights = true;
	bool normal_maps_in_bounces = true;
	bool respect_material_cull_mode = true;
	bool skinned_meshes_in_gi = true;
	float ray_origin_bias = 0.005f;

	// Sky & Irradiance
	bool sky_indirect_enabled = true;
	float sky_indirect_energy = 1.0f;
	CustomSkyMode custom_sky_mode = SKY_MODE_ENVIRONMENT;
	Ref<Sky> custom_sky;
	Color custom_sky_color = Color(0.2f, 0.25f, 0.3f);
	float custom_sky_energy = 1.0f;
	Vector3 custom_sky_rotation = Vector3(0.0f, 0.0f, 0.0f);
	float ambient_bounce_approximation = 0.0f;

	// 3D Irradiance Volume (Infinite-Bounce Cache)
	bool volume_enabled = true;
	Vector3 volume_extents = Vector3(24.0f, 12.0f, 24.0f);
	Vector3 volume_resolution = Vector3(24, 12, 24);
	int volume_rays_per_voxel = 6;
	float volume_ray_distance = 40.0f;
	float volume_feedback = 0.65f;
	int volume_update_interval = 2;

	// Material Alpha
	bool alpha_consideration = true;
	bool alpha_ray_transparency = true;
	int alpha_max_steps = 4;
	bool alpha_shadow_transmission = true;

	// Mesh Simplification
	bool mesh_simplification_enabled = false;
	SimplificationMode mesh_simplification_mode = SIMPLIFY_QEM;
	float mesh_simplification_ratio = 0.5f;
	int mesh_simplification_min_triangles = 128;
	float mesh_simplification_max_error = 0.05f;
	bool mesh_simplification_preserve_seams = true;
	bool mesh_simplification_skinned = false;
	bool mesh_simplification_gbuffer = false;

	// Shadows
	bool trace_shadows = true;
	int shadow_bounces = 2;
	float shadow_max_distance = 80.0f;
	float shadow_ray_bias = 0.005f;

	// Scalability & Culling
	int update_interval = 1;
	int blas_builds_per_frame = 0;
	int max_tlas_instances = 0;
	bool frustum_cull_gbuffer = true;
	TextureResolution texture_array_resolution = TEX_RES_256;
	int max_texture_layers = 64;
	float texture_lod_bias = 0.5f;
	float distance_lod_strength = 1.0f;
	float mesh_cull_distance = 45.0f;
	float mesh_cull_min_angle = 0.012f;
	float light_cull_distance = 120.0f;

	// Debug
	DebugMode debug_mode = DEBUG_NONE;

	// VisualServer Overlay & Textures
	Ref<ImageTexture> gi_texture;
	Ref<ImageTexture> gi_albedo_texture;
	Ref<Texture3D> gi_volume_texture;
	Ref<Shader> composite_shader;
	std::unordered_map<uint32_t, ViewportRenderState> viewport_states;
	uint64_t vp_collect_counter = 0;
	bool vs_resources_created = false;

	// Runtime State & Caches
	Mutex scene_mutex;
	SceneSnapshot snapshot;
	GL43Raytracer raytracer;
	GL43TextureArrayPool tex_pool;
	Ref<Texture> pinned_panorama;

	uint64_t frame_counter = 0;
	uint64_t process_frame_counter = 0;
	uint64_t volume_frame_counter = 0;
	uint64_t last_sky_sh_hash = 0;
	bool sky_sh_dirty = true;
	bool bvh_dirty = true;

	int alloc_vol_x = 0;
	int alloc_vol_y = 0;
	int alloc_vol_z = 0;

	// OpenGL 4.3 Resources
	bool gl_attempted = false;
	bool gl_ready = false;
	Thread::ID gl_thread_id = 0;

	uint32_t gbuf_program = 0;
	uint32_t downsample_gbuf_program = 0;
	uint32_t sh_program = 0;
	uint32_t volume_program = 0;
	uint32_t volume_filter_program = 0;
	uint32_t rt_program = 0;
	uint32_t reconstruct_program = 0;
	uint32_t denoise_program = 0;

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
	float rendered_vol_voxel_size = 2.0f;

	uint32_t dummy_white_tex = 0;
	uint32_t dummy_gl_3d = 0;
	uint32_t pano_sampler_mip = 0;
	uint32_t pano_sampler_nomip = 0;

	uint32_t ssbo_materials = 0;
	uint32_t ssbo_lights = 0;
	uint32_t ssbo_sky_sh = 0;

	GL43Raytracer::SimplificationConfig _get_simplification_config() const;
	void _create_vs_resources();
	void _free_vs_resources();
	void _detach_all_viewports();
	ViewportRenderState &_get_or_create_vp_state(Viewport *p_vp);
	void _prune_stale_viewports();

	bool _init_gl_resources();
	void _cleanup_gl_resources();
	void _ensure_gbuffer_size(ViewportRenderState &r_state, int p_w, int p_h);
	void _ensure_low_gbuf_size(ViewportRenderState &r_state, int p_w, int p_h);
	void _ensure_vol_raw_size(int p_x, int p_y, int p_z);
	void _ensure_work_textures(ViewportRenderState &r_state, int p_w, int p_h, bool p_need_history);

	void _collect_scene_snapshot();
	void _on_frame_pre_draw();

protected:
	static void _bind_methods();
	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const;

public:
	RTGI();
	~RTGI();

	// Core
	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_editor_preview(bool p_preview);
	bool get_editor_preview() const { return editor_preview; }

	void set_auto_composite(bool p_auto);
	bool get_auto_composite() const { return auto_composite; }

	void set_gi_intensity(float p_intensity);
	float get_gi_intensity() const { return gi_intensity; }

	void set_gi_cull_mask(uint32_t p_mask) { gi_cull_mask = p_mask; }
	uint32_t get_gi_cull_mask() const { return gi_cull_mask; }

	void set_gi_viewing_cull_mask(uint32_t p_mask) { gi_viewing_cull_mask = p_mask; }
	uint32_t get_gi_viewing_cull_mask() const { return gi_viewing_cull_mask; }

	// Resolution & Denoising
	void set_resolution_scale(float p_scale);
	float get_resolution_scale() const { return resolution_scale; }

	void set_bilateral_upsample(bool p_enable) { bilateral_upsample = p_enable; }
	bool get_bilateral_upsample() const { return bilateral_upsample; }

	void set_checkerboard_tracing(bool p_enable) { checkerboard_tracing = p_enable; }
	bool get_checkerboard_tracing() const { return checkerboard_tracing; }

	void set_temporal_accumulation(bool p_enable);
	bool get_temporal_accumulation() const { return temporal_accumulation; }

	void set_temporal_blend(float p_blend);
	float get_temporal_blend() const { return temporal_blend; }

	void set_temporal_noise(bool p_enable) { temporal_noise = p_enable; }
	bool get_temporal_noise() const { return temporal_noise; }

	void set_spatial_denoise_passes(int p_passes);
	int get_spatial_denoise_passes() const { return spatial_denoise_passes; }

	void set_firefly_clamp(float p_clamp);
	float get_firefly_clamp() const { return firefly_clamp; }

	// Tracing & Bounces
	void set_rays_per_pixel(int p_rays);
	int get_rays_per_pixel() const { return rays_per_pixel; }

	void set_max_bounces(int p_bounces);
	int get_max_bounces() const { return max_bounces; }

	void set_max_ray_distance(float p_dist);
	float get_max_ray_distance() const { return max_ray_distance; }

	void set_bounce_distance_falloff(float p_falloff);
	float get_bounce_distance_falloff() const { return bounce_distance_falloff; }

	void set_bounce_energy_boost(float p_boost);
	float get_bounce_energy_boost() const { return bounce_energy_boost; }

	void set_bounce_throughput_threshold(float p_thresh);
	float get_bounce_throughput_threshold() const { return bounce_throughput_threshold; }

	void set_emissive_intensity(float p_intensity);
	float get_emissive_intensity() const { return emissive_intensity; }

	void set_bounce_local_lights(bool p_enable) { bounce_local_lights = p_enable; }
	bool get_bounce_local_lights() const { return bounce_local_lights; }

	void set_normal_maps_in_bounces(bool p_enable) { normal_maps_in_bounces = p_enable; }
	bool get_normal_maps_in_bounces() const { return normal_maps_in_bounces; }

	void set_respect_material_cull_mode(bool p_enable) { respect_material_cull_mode = p_enable; }
	bool get_respect_material_cull_mode() const { return respect_material_cull_mode; }

	void set_skinned_meshes_in_gi(bool p_enable) { skinned_meshes_in_gi = p_enable; }
	bool get_skinned_meshes_in_gi() const { return skinned_meshes_in_gi; }

	void set_ray_origin_bias(float p_bias);
	float get_ray_origin_bias() const { return ray_origin_bias; }

	// Sky & Irradiance
	void set_sky_indirect_enabled(bool p_enable);
	bool is_sky_indirect_enabled() const { return sky_indirect_enabled; }

	void set_sky_indirect_energy(float p_energy);
	float get_sky_indirect_energy() const { return sky_indirect_energy; }

	void set_custom_sky_mode(CustomSkyMode p_mode);
	CustomSkyMode get_custom_sky_mode() const { return custom_sky_mode; }

	void set_custom_sky(const Ref<Sky> &p_sky);
	Ref<Sky> get_custom_sky() const { return custom_sky; }

	void set_custom_sky_color(const Color &p_color);
	Color get_custom_sky_color() const { return custom_sky_color; }

	void set_custom_sky_energy(float p_energy);
	float get_custom_sky_energy() const { return custom_sky_energy; }

	void set_custom_sky_rotation(const Vector3 &p_rot_deg) { custom_sky_rotation = p_rot_deg; }
	Vector3 get_custom_sky_rotation() const { return custom_sky_rotation; }

	void set_ambient_bounce_approximation(float p_approx);
	float get_ambient_bounce_approximation() const { return ambient_bounce_approximation; }

	// 3D Irradiance Volume
	void set_volume_enabled(bool p_enable);
	bool is_volume_enabled() const { return volume_enabled; }

	void set_volume_extents(const Vector3 &p_extents);
	Vector3 get_volume_extents() const { return volume_extents; }

	void set_volume_resolution(const Vector3 &p_res);
	Vector3 get_volume_resolution() const { return volume_resolution; }

	void set_volume_rays_per_voxel(int p_rays);
	int get_volume_rays_per_voxel() const { return volume_rays_per_voxel; }

	void set_volume_ray_distance(float p_dist);
	float get_volume_ray_distance() const { return volume_ray_distance; }

	void set_volume_feedback(float p_feedback);
	float get_volume_feedback() const { return volume_feedback; }

	void set_volume_update_interval(int p_interval);
	int get_volume_update_interval() const { return volume_update_interval; }

	// Material Alpha
	void set_alpha_consideration(bool p_enable);
	bool get_alpha_consideration() const { return alpha_consideration; }

	void set_alpha_ray_transparency(bool p_enable) { alpha_ray_transparency = p_enable; }
	bool get_alpha_ray_transparency() const { return alpha_ray_transparency; }

	void set_alpha_max_steps(int p_steps);
	int get_alpha_max_steps() const { return alpha_max_steps; }

	void set_alpha_shadow_transmission(bool p_enable) { alpha_shadow_transmission = p_enable; }
	bool get_alpha_shadow_transmission() const { return alpha_shadow_transmission; }

	// Mesh Simplification
	void set_mesh_simplification_enabled(bool p_enable);
	bool is_mesh_simplification_enabled() const { return mesh_simplification_enabled; }

	void set_mesh_simplification_mode(SimplificationMode p_mode);
	SimplificationMode get_mesh_simplification_mode() const { return mesh_simplification_mode; }

	void set_mesh_simplification_ratio(float p_ratio);
	float get_mesh_simplification_ratio() const { return mesh_simplification_ratio; }

	void set_mesh_simplification_min_triangles(int p_min_tris);
	int get_mesh_simplification_min_triangles() const { return mesh_simplification_min_triangles; }

	void set_mesh_simplification_max_error(float p_max_error);
	float get_mesh_simplification_max_error() const { return mesh_simplification_max_error; }

	void set_mesh_simplification_preserve_seams(bool p_preserve);
	bool get_mesh_simplification_preserve_seams() const { return mesh_simplification_preserve_seams; }

	void set_mesh_simplification_skinned(bool p_enable);
	bool get_mesh_simplification_skinned() const { return mesh_simplification_skinned; }

	void set_mesh_simplification_gbuffer(bool p_enable);
	bool get_mesh_simplification_gbuffer() const { return mesh_simplification_gbuffer; }

	// Shadows
	void set_trace_shadows(bool p_enable);
	bool get_trace_shadows() const { return trace_shadows; }

	void set_shadow_bounces(int p_bounces);
	int get_shadow_bounces() const { return shadow_bounces; }

	void set_shadow_max_distance(float p_dist);
	float get_shadow_max_distance() const { return shadow_max_distance; }

	void set_shadow_ray_bias(float p_bias);
	float get_shadow_ray_bias() const { return shadow_ray_bias; }

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

	void set_texture_lod_bias(float p_bias);
	float get_texture_lod_bias() const { return texture_lod_bias; }

	void set_distance_lod_strength(float p_strength);
	float get_distance_lod_strength() const { return distance_lod_strength; }

	void set_mesh_cull_distance(float p_dist);
	float get_mesh_cull_distance() const { return mesh_cull_distance; }

	void set_mesh_cull_min_angle(float p_angle);
	float get_mesh_cull_min_angle() const { return mesh_cull_min_angle; }

	void set_light_cull_distance(float p_dist);
	float get_light_cull_distance() const { return light_cull_distance; }

	void set_debug_mode(DebugMode p_mode) { debug_mode = p_mode; }
	DebugMode get_debug_mode() const { return debug_mode; }

	void apply_quality_preset(QualityPreset p_preset);
	void mark_geometry_dirty();
	void reset_history();

	Ref<Texture> get_gi_texture() const { return gi_texture; }
	Ref<Texture> get_gi_albedo_texture() const { return gi_albedo_texture; }
	Ref<Texture3D> get_gi_volume_texture() const { return gi_volume_texture; }
};

VARIANT_ENUM_CAST(RTGI::QualityPreset);
VARIANT_ENUM_CAST(RTGI::CustomSkyMode);
VARIANT_ENUM_CAST(RTGI::SimplificationMode);
VARIANT_ENUM_CAST(RTGI::TextureResolution);
VARIANT_ENUM_CAST(RTGI::DebugMode);

#endif // RTGI_H

