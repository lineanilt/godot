#ifndef RT_REFLECTIONS_H
#define RT_REFLECTIONS_H

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

class MeshInstance;
class Camera;
class Viewport;
class Skeleton;

class RTReflections : public Node {
    GDCLASS(RTReflections, Node);

public:
    enum ReflectionFiltering {
        FILTER_BILINEAR = 0,
        FILTER_BICUBIC = 1
    };

    enum QualityPreset {
        PRESET_LOW = 0,
        PRESET_MEDIUM = 1,
        PRESET_HIGH = 2,
        PRESET_ULTRA = 3
    };

    enum CustomSkyMode {
        SKY_MODE_ENVIRONMENT = 0,
        SKY_MODE_CUSTOM_SKY = 1,
        SKY_MODE_COLOR = 2
    };

    enum AmbientLightMode {
        AMBIENT_MODE_ENVIRONMENT = 0,
        AMBIENT_MODE_CUSTOM = 1,
        AMBIENT_MODE_DISABLED = 2
    };

    enum SimplificationMode {
        SIMPLIFY_QEM = 0,
        SIMPLIFY_VERTEX_CLUSTERING = 1
    };

    enum TextureResolution {
        TEX_RES_128 = 128,
        TEX_RES_256 = 256,
        TEX_RES_512 = 512,
        TEX_RES_1024 = 1024
    };

    // 7 x 16 bytes = 112 bytes (std430 aligned)
    struct GpuMaterial {
        float albedo[4];            // rgb = linear albedo, a = base_alpha
        float emission[4];          // rgb = linear emission * energy, a = metallic
        float pbr_params[4];        // x = roughness, y = alpha_scissor, z = specular, w = ao_light_affect
        float uv_scale_offset[4];   // xy = uv1_scale, zw = uv1_offset
        int32_t tex_layers[4];      // x = albedo, y = roughness, z = metallic, w = emission
        int32_t chan_normal[4];     // x = rough_chan, y = metal_chan, z = normal_layer, w = float_bits(normal_scale)
        int32_t extra_tex[4];       // x = ao_layer, y = ao_chan, z = alpha_mode (0=opaque,1=scissor,2=transparent), w = flags
    };

    struct GpuLight {
        float pos_range[4];
        float dir_type[4];
        float color_atten[4];
        float spot_params[4];
    };

private:
    enum InstanceUserFlags {
        INST_FLAG_SKIP_REFLECTION   = 1 << 0,
        INST_FLAG_HAS_OPAQUE_CUTOUT = 1 << 1,
        INST_FLAG_HAS_TRANSPARENT   = 1 << 2,
    };

    struct AttachedPass {
        Ref<SpatialMaterial> mat;
        Ref<ShaderMaterial> pass;
        int cull_mode = 0;
        uint64_t last_seen = 0;
    };

    struct SceneSnapshot {
        bool valid = false;
        RID color_tex_rid;
        int viewport_w = 0;
        int viewport_h = 0;
        float cam_pos[3] = { 0, 0, 0 };
        float view_mat[16] = {};
        float proj_mat[16] = {};
        float view_proj_mat[16] = {};
        float inv_view_proj[16] = {};
        float sky_top[3] = { 0.376f, 0.672f, 0.880f };
        float sky_horizon[3] = { 0.672f, 0.823f, 0.956f };
        float sky_ground_horizon[3] = { 0.153f, 0.127f, 0.105f };
        float sky_ground[3] = { 0.021f, 0.018f, 0.014f };
        float sky_curve = 0.09f;
        float ground_curve = 0.02f;
        float sky_contrib = 1.0f;
        float sky_energy = 1.0f;
        float sky_ground_energy = 1.0f;
        float ambient_color[3] = { 0.0f, 0.0f, 0.0f };
        float ambient_energy = 1.0f;
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
        bool has_transparent_gbuf = false;
        bool has_alpha_materials = false;
        int num_dir_lights = 0;
        std::vector<GpuMaterial> materials;
        std::vector<GpuLight> lights;
    };

    // Core
    bool enabled = true;
    bool editor_preview = true;
    bool auto_composite = true;
    bool auto_attach_materials = true;
    float reflection_intensity = 1.0f;
    uint32_t reflection_cull_mask = 0xFFFFFFFF;
    uint32_t reflection_viewing_cull_mask = 0xFFFFFFFF;

    // Resolution & Filtering
    float reflection_scale = 0.5f;
    ReflectionFiltering reflection_filtering = FILTER_BICUBIC;
    bool bilateral_upsample = true;
    bool temporal_accumulation = true;
    float temporal_blend = 0.15f;
    bool checkerboard_tracing = false;
    int spatial_denoise_passes = 2;
    float max_roughness = 0.65f;
    float roughness_fade_range = 0.15f;
    float blur_strength = 1.0f;
    float blur_lod_scale = 5.0f;

    // Tracing & Bounces
    int ray_samples_per_pixel = 1;
    float glossy_roughness_spread = 0.25f;
    int max_bounces = 1;
    float max_ray_distance = 200.0f;
    float bounce_distance_falloff = 0.5f;
    float bounce_throughput_threshold = 0.02f;
    bool bounce_local_lights = true;
    int pbr_texture_bounces = 1;
    bool normal_maps_in_bounces = true;
    bool respect_material_cull_mode = true;
    bool skinned_meshes_in_reflections = true;
    bool final_bounce_env_fallback = true;
    float ray_origin_bias = 0.004f;

    // Custom Sky
    CustomSkyMode custom_sky_mode = SKY_MODE_ENVIRONMENT;
    Ref<Sky> custom_sky;
    Color custom_sky_color = Color(0.2f, 0.25f, 0.3f);
    float custom_sky_energy = 1.0f;
    Vector3 custom_sky_rotation = Vector3(0.0f, 0.0f, 0.0f);
    float sky_reflection_energy = 1.0f;

    // Ambient Light & Radiance
    AmbientLightMode ambient_light_mode = AMBIENT_MODE_ENVIRONMENT;
    Color ambient_light_color = Color(0.1f, 0.1f, 0.12f);
    float ambient_light_energy = 1.0f;
    float ambient_light_sky_contribution = 1.0f;
    float ambient_radiance_boost = 1.0f;
    float ambient_bounce_approximation = 0.25f;
    int ambient_occlusion_samples = 0;
    float ambient_occlusion_radius = 1.5f;
    float ambient_occlusion_intensity = 0.75f;

    // Material Alpha
    bool alpha_consideration = true;
    bool alpha_ray_transparency = true;
    int alpha_max_steps = 6;
    bool alpha_shadow_transmission = true;
    bool transparent_receivers = true;
    float transparent_receiver_min_alpha = 0.02f;
    float transparent_fresnel_boost = 0.75f;

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
    int shadow_bounces = 1;
    float shadow_max_distance = 80.0f;
    float shadow_ray_bias = 0.004f;

    // Scalability & Texture Pool
    int update_interval = 1;
    int blas_builds_per_frame = 0;
    int max_tlas_instances = 0;
    bool frustum_cull_gbuffer = true;
    TextureResolution texture_array_resolution = TEX_RES_512;
    int max_texture_layers = 64;

    // LOD & Culling
    float texture_lod_bias = 0.0f;
    float distance_lod_strength = 1.0f;
    float mesh_cull_distance = 45.0f;
    float mesh_cull_min_angle = 0.012f;
    float light_cull_distance = 120.0f;

    // Runtime state
    Ref<ImageTexture> refl_texture;
    Ref<ImageTexture> composite_meta_texture;
    bool hook_registered = false;
    int tex_w = 0;
    int tex_h = 0;
    int gbuf_w = 0;
    int gbuf_h = 0;
    int work_w = 0;
    int work_h = 0;

    Mutex scene_mutex;
    GL43Raytracer raytracer;
    uint64_t frame_counter = 0;
    uint64_t snapshot_counter = 0;
    uint64_t process_frame_counter = 0;
    uint64_t last_sky_sh_hash = 0;
    bool sky_sh_dirty = true;

    bool history_valid = false;
    int history_idx = 0;
    float prev_view_proj[16] = {};
    float prev_cam_pos[3] = { 0.0f, 0.0f, 0.0f };

    std::unordered_map<uint32_t, AttachedPass> attached_passes;
    GL43TextureArrayPool tex_pool;
    Ref<Texture> pinned_panorama;
    Ref<Shader> shared_next_pass_shaders[3]; // 0 = CULL_BACK, 1 = CULL_FRONT, 2 = CULL_DISABLED
    SceneSnapshot snapshot;
    bool bvh_dirty = true;

    bool gl_attempted = false;
    bool gl_ready = false;
    Thread::ID gl_thread_id = 0;

    uint32_t gbuf_fbo = 0;
    uint32_t gbuf_normal_rough = 0;
    uint32_t gbuf_albedo_metal = 0;
    uint32_t gbuf_extra = 0;
    uint32_t gbuf_depth = 0;
    uint32_t gbuf_program = 0;

    uint32_t rt_program = 0;
    uint32_t sh_program = 0;
    uint32_t reconstruct_program = 0;
    uint32_t denoise_program = 0;
    uint32_t composite_program = 0;

    uint32_t rt_work_a = 0;
    uint32_t rt_work_b = 0;
    uint32_t rt_history_tex[2] = { 0, 0 };

    uint32_t dummy_white_tex = 0;
    uint32_t pano_sampler_mip = 0;
    uint32_t pano_sampler_nomip = 0;

    uint32_t ssbo_materials = 0;
    uint32_t ssbo_lights = 0;
    uint32_t ssbo_sky_sh = 0;

    GL43Raytracer::SimplificationConfig _get_simplification_config() const;
    void invalidate_blas_cache();
    bool init_gl_extensions_and_shaders();
    void cleanup_gl_resources();
    void ensure_gbuffer_size(int w, int h);
    void ensure_work_textures(int w, int h, bool need_history);
    void collect_scene_snapshot();
    void _register_scene_hook();
    void _unregister_scene_hook();
    void attach_next_pass(Ref<SpatialMaterial> smat);
    void detach_next_pass(uint32_t mat_id);
    void clear_attached_passes();

    void _on_scene_hook(const GL43EffectHelpers::SceneHookContext &p_ctx);
    void _on_frame_pre_draw();

protected:
    static void _bind_methods();
    void _notification(int p_what);
    virtual void _validate_property(PropertyInfo &property) const;

public:
    RTReflections();
    ~RTReflections();

    // Core
    void set_enabled(bool p_enabled);
    bool is_enabled() const { return enabled; }

    void set_editor_preview(bool p_preview);
    bool get_editor_preview() const { return editor_preview; }

    void set_auto_composite(bool p_auto);
    bool get_auto_composite() const { return auto_composite; }

    void set_auto_attach_materials(bool p_auto);
    bool get_auto_attach_materials() const { return auto_attach_materials; }

    void set_reflection_intensity(float p_intensity);
    float get_reflection_intensity() const { return reflection_intensity; }

    void set_reflection_cull_mask(uint32_t p_mask);
    uint32_t get_reflection_cull_mask() const { return reflection_cull_mask; }

    void set_reflection_viewing_cull_mask(uint32_t p_mask);
    uint32_t get_reflection_viewing_cull_mask() const { return reflection_viewing_cull_mask; }

    // Resolution & Filtering
    void set_reflection_scale(float p_scale);
    float get_reflection_scale() const { return reflection_scale; }

    void set_reflection_filtering(ReflectionFiltering p_filtering) { reflection_filtering = p_filtering; }
    ReflectionFiltering get_reflection_filtering() const { return reflection_filtering; }

    void set_bilateral_upsample(bool p_enable) { bilateral_upsample = p_enable; }
    bool get_bilateral_upsample() const { return bilateral_upsample; }

    void set_temporal_accumulation(bool p_enable);
    bool get_temporal_accumulation() const { return temporal_accumulation; }

    void set_temporal_blend(float p_blend);
    float get_temporal_blend() const { return temporal_blend; }

    void set_checkerboard_tracing(bool p_enable) { checkerboard_tracing = p_enable; }
    bool get_checkerboard_tracing() const { return checkerboard_tracing; }

    void set_spatial_denoise_passes(int p_passes);
    int get_spatial_denoise_passes() const { return spatial_denoise_passes; }

    void set_max_roughness(float p_rough);
    float get_max_roughness() const { return max_roughness; }

    void set_roughness_fade_range(float p_range);
    float get_roughness_fade_range() const { return roughness_fade_range; }

    void set_blur_strength(float p_strength);
    float get_blur_strength() const { return blur_strength; }

    void set_blur_lod_scale(float p_scale);
    float get_blur_lod_scale() const { return blur_lod_scale; }

    // Tracing & Bounces
    void set_ray_samples_per_pixel(int p_samples);
    int get_ray_samples_per_pixel() const { return ray_samples_per_pixel; }

    void set_glossy_roughness_spread(float p_spread);
    float get_glossy_roughness_spread() const { return glossy_roughness_spread; }

    void set_max_bounces(int p_bounces);
    int get_max_bounces() const { return max_bounces; }

    void set_max_ray_distance(float p_dist);
    float get_max_ray_distance() const { return max_ray_distance; }

    void set_bounce_distance_falloff(float p_falloff);
    float get_bounce_distance_falloff() const { return bounce_distance_falloff; }

    void set_bounce_throughput_threshold(float p_thresh);
    float get_bounce_throughput_threshold() const { return bounce_throughput_threshold; }

    void set_bounce_local_lights(bool p_enable) { bounce_local_lights = p_enable; }
    bool get_bounce_local_lights() const { return bounce_local_lights; }

    void set_pbr_texture_bounces(int p_bounces);
    int get_pbr_texture_bounces() const { return pbr_texture_bounces; }

    void set_normal_maps_in_bounces(bool p_enable) { normal_maps_in_bounces = p_enable; }
    bool get_normal_maps_in_bounces() const { return normal_maps_in_bounces; }

    void set_respect_material_cull_mode(bool p_enable) { respect_material_cull_mode = p_enable; }
    bool get_respect_material_cull_mode() const { return respect_material_cull_mode; }

    void set_skinned_meshes_in_reflections(bool p_enable) { skinned_meshes_in_reflections = p_enable; }
    bool get_skinned_meshes_in_reflections() const { return skinned_meshes_in_reflections; }

    void set_final_bounce_env_fallback(bool p_enable) { final_bounce_env_fallback = p_enable; }
    bool get_final_bounce_env_fallback() const { return final_bounce_env_fallback; }

    void set_ray_origin_bias(float p_bias);
    float get_ray_origin_bias() const { return ray_origin_bias; }

    // Custom Sky
    void set_custom_sky_mode(CustomSkyMode p_mode);
    CustomSkyMode get_custom_sky_mode() const { return custom_sky_mode; }

    void set_custom_sky(const Ref<Sky> &p_sky);
    Ref<Sky> get_custom_sky() const { return custom_sky; }

    void set_custom_sky_color(const Color &p_color);
    Color get_custom_sky_color() const { return custom_sky_color; }

    void set_custom_sky_energy(float p_energy);
    float get_custom_sky_energy() const { return custom_sky_energy; }

    void set_custom_sky_rotation(const Vector3 &p_rot_deg);
    Vector3 get_custom_sky_rotation() const { return custom_sky_rotation; }

    void set_sky_reflection_energy(float p_energy);
    float get_sky_reflection_energy() const { return sky_reflection_energy; }

    // Ambient Light & Radiance
    void set_ambient_light_mode(AmbientLightMode p_mode);
    AmbientLightMode get_ambient_light_mode() const { return ambient_light_mode; }

    void set_ambient_light_color(const Color &p_color) { ambient_light_color = p_color; }
    Color get_ambient_light_color() const { return ambient_light_color; }

    void set_ambient_light_energy(float p_energy);
    float get_ambient_light_energy() const { return ambient_light_energy; }

    void set_ambient_light_sky_contribution(float p_contrib);
    float get_ambient_light_sky_contribution() const { return ambient_light_sky_contribution; }

    void set_ambient_radiance_boost(float p_boost);
    float get_ambient_radiance_boost() const { return ambient_radiance_boost; }

    void set_ambient_bounce_approximation(float p_approx);
    float get_ambient_bounce_approximation() const { return ambient_bounce_approximation; }

    void set_ambient_occlusion_samples(int p_samples);
    int get_ambient_occlusion_samples() const { return ambient_occlusion_samples; }

    void set_ambient_occlusion_radius(float p_radius);
    float get_ambient_occlusion_radius() const { return ambient_occlusion_radius; }

    void set_ambient_occlusion_intensity(float p_intensity);
    float get_ambient_occlusion_intensity() const { return ambient_occlusion_intensity; }

    // Material Alpha
    void set_alpha_consideration(bool p_enable);
    bool get_alpha_consideration() const { return alpha_consideration; }

    void set_alpha_ray_transparency(bool p_enable) { alpha_ray_transparency = p_enable; }
    bool get_alpha_ray_transparency() const { return alpha_ray_transparency; }

    void set_alpha_max_steps(int p_steps);
    int get_alpha_max_steps() const { return alpha_max_steps; }

    void set_alpha_shadow_transmission(bool p_enable) { alpha_shadow_transmission = p_enable; }
    bool get_alpha_shadow_transmission() const { return alpha_shadow_transmission; }

    void set_transparent_receivers(bool p_enable) { transparent_receivers = p_enable; }
    bool get_transparent_receivers() const { return transparent_receivers; }

    void set_transparent_receiver_min_alpha(float p_alpha);
    float get_transparent_receiver_min_alpha() const { return transparent_receiver_min_alpha; }

    void set_transparent_fresnel_boost(float p_boost);
    float get_transparent_fresnel_boost() const { return transparent_fresnel_boost; }

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
    void set_trace_shadows(bool p_enable) { trace_shadows = p_enable; }
    bool get_trace_shadows() const { return trace_shadows; }

    void set_shadow_bounces(int p_bounces);
    int get_shadow_bounces() const { return shadow_bounces; }

    void set_shadow_max_distance(float p_dist);
    float get_shadow_max_distance() const { return shadow_max_distance; }

    void set_shadow_ray_bias(float p_bias);
    float get_shadow_ray_bias() const { return shadow_ray_bias; }

    // Scalability & Texture Pool
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

    // LOD & Culling
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

    void apply_quality_preset(QualityPreset p_preset);

    Ref<Texture> get_reflection_texture() const { return refl_texture; }
    void mark_geometry_dirty() { bvh_dirty = true; }
};

VARIANT_ENUM_CAST(RTReflections::ReflectionFiltering);
VARIANT_ENUM_CAST(RTReflections::QualityPreset);
VARIANT_ENUM_CAST(RTReflections::CustomSkyMode);
VARIANT_ENUM_CAST(RTReflections::AmbientLightMode);
VARIANT_ENUM_CAST(RTReflections::SimplificationMode);
VARIANT_ENUM_CAST(RTReflections::TextureResolution);

#endif
