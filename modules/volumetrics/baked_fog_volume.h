#ifndef BAKED_FOG_VOLUME_H
#define BAKED_FOG_VOLUME_H

#include "scene/3d/spatial.h"
#include "scene/resources/texture.h"

class BakedFogVolume : public Spatial {
	GDCLASS(BakedFogVolume, Spatial);

private:
	bool enabled = true;
	Vector3 extents = Vector3(8.0f, 4.0f, 8.0f);
	Vector3 resolution = Vector3(32, 16, 32);
	Ref<Texture3D> baked_texture;
	PoolByteArray baked_data;
	Vector3 baked_data_resolution = Vector3(0, 0, 0);

	void _rebuild_texture_from_baked_data();

	float density_multiplier = 1.0f;
	float emission_multiplier = 1.0f;
	Color albedo_tint = Color(1.0f, 1.0f, 1.0f, 1.0f);
	float edge_falloff = 0.1f;

	float bake_base_density = 0.05f;
	Color bake_base_albedo = Color(0.9f, 0.92f, 0.96f, 1.0f);
	Color bake_base_emission = Color(0.0f, 0.0f, 0.0f, 1.0f);
	bool bake_local_volumes = true;
	bool bake_lights = true;
	bool bake_shadows = true;
	int bake_shadow_samples = 1;
	float bake_shadow_bias = 0.05f;
	float bake_noise_strength = 0.0f;
	float bake_noise_scale = 0.25f;
	bool bake_filter = true;
	uint32_t cull_mask = 0xFFFFFFFF;
	uint32_t shadow_cull_mask = 0xFFFFFFFF;

protected:
	static void _bind_methods();

public:
	BakedFogVolume();

	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const { return extents; }

	void set_resolution(const Vector3 &p_res);
	Vector3 get_resolution() const { return resolution; }

	void set_baked_texture(const Ref<Texture3D> &p_tex);
	Ref<Texture3D> get_baked_texture() const { return baked_texture; }

	void set_baked_data(const PoolByteArray &p_data);
	PoolByteArray get_baked_data() const { return baked_data; }

	void set_baked_data_resolution(const Vector3 &p_res);
	Vector3 get_baked_data_resolution() const { return baked_data_resolution; }

	void set_density_multiplier(float p_mult);
	float get_density_multiplier() const { return density_multiplier; }

	void set_emission_multiplier(float p_mult);
	float get_emission_multiplier() const { return emission_multiplier; }

	void set_albedo_tint(const Color &p_tint) { albedo_tint = p_tint; }
	Color get_albedo_tint() const { return albedo_tint; }

	void set_edge_falloff(float p_falloff);
	float get_edge_falloff() const { return edge_falloff; }

	void set_bake_base_density(float p_density);
	float get_bake_base_density() const { return bake_base_density; }

	void set_bake_base_albedo(const Color &p_albedo) { bake_base_albedo = p_albedo; }
	Color get_bake_base_albedo() const { return bake_base_albedo; }

	void set_bake_base_emission(const Color &p_emission) { bake_base_emission = p_emission; }
	Color get_bake_base_emission() const { return bake_base_emission; }

	void set_bake_local_volumes(bool p_enable) { bake_local_volumes = p_enable; }
	bool get_bake_local_volumes() const { return bake_local_volumes; }

	void set_bake_lights(bool p_enable) { bake_lights = p_enable; }
	bool get_bake_lights() const { return bake_lights; }

	void set_bake_shadows(bool p_enable) { bake_shadows = p_enable; }
	bool get_bake_shadows() const { return bake_shadows; }

	void set_bake_shadow_samples(int p_samples);
	int get_bake_shadow_samples() const { return bake_shadow_samples; }

	void set_bake_shadow_bias(float p_bias);
	float get_bake_shadow_bias() const { return bake_shadow_bias; }

	void set_bake_noise_strength(float p_strength);
	float get_bake_noise_strength() const { return bake_noise_strength; }

	void set_bake_noise_scale(float p_scale);
	float get_bake_noise_scale() const { return bake_noise_scale; }

	void set_bake_filter(bool p_filter) { bake_filter = p_filter; }
	bool get_bake_filter() const { return bake_filter; }

	void set_cull_mask(uint32_t p_mask) { cull_mask = p_mask; }
	uint32_t get_cull_mask() const { return cull_mask; }

	void set_shadow_cull_mask(uint32_t p_mask) { shadow_cull_mask = p_mask; }
	uint32_t get_shadow_cull_mask() const { return shadow_cull_mask; }

	AABB get_world_aabb() const;
	bool bake();
};

#endif // BAKED_FOG_VOLUME_H
