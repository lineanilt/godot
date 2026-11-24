#ifndef LOCAL_FOG_VOLUME_H
#define LOCAL_FOG_VOLUME_H

#include "scene/3d/spatial.h"

class LocalFogVolume : public Spatial {
	GDCLASS(LocalFogVolume, Spatial);

public:
	enum Shape {
		SHAPE_BOX = 0,
		SHAPE_SPHERE = 1,
		SHAPE_ELLIPSOID = 2,
		SHAPE_CYLINDER = 3,
		SHAPE_CONE = 4,
	};

	enum BlendMode {
		BLEND_ADDITIVE = 0,
		BLEND_MAX = 1,
		BLEND_MULTIPLY = 2,
		BLEND_SUBTRACT = 3,
	};

private:
	bool enabled = true;
	Shape shape = SHAPE_BOX;
	Vector3 extents = Vector3(2.0f, 2.0f, 2.0f);
	float density = 0.5f;
	Color albedo = Color(1.0f, 1.0f, 1.0f, 1.0f);
	Color emission = Color(0.0f, 0.0f, 0.0f, 1.0f);
	float emission_energy = 1.0f;
	float edge_falloff = 0.25f;
	BlendMode blend_mode = BLEND_ADDITIVE;
	float noise_strength = 0.0f;
	float noise_scale = 1.0f;
	Vector3 noise_velocity = Vector3(0.0f, 0.0f, 0.0f);
	int priority = 0;
	uint32_t cull_mask = 0xFFFFFFFF;

protected:
	static void _bind_methods();

public:
	LocalFogVolume();

	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_shape(Shape p_shape);
	Shape get_shape() const { return shape; }

	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const { return extents; }

	void set_density(float p_density);
	float get_density() const { return density; }

	void set_albedo(const Color &p_albedo) { albedo = p_albedo; }
	Color get_albedo() const { return albedo; }

	void set_emission(const Color &p_emission) { emission = p_emission; }
	Color get_emission() const { return emission; }

	void set_emission_energy(float p_energy);
	float get_emission_energy() const { return emission_energy; }

	void set_edge_falloff(float p_falloff);
	float get_edge_falloff() const { return edge_falloff; }

	void set_blend_mode(BlendMode p_mode) { blend_mode = p_mode; }
	BlendMode get_blend_mode() const { return blend_mode; }

	void set_noise_strength(float p_strength);
	float get_noise_strength() const { return noise_strength; }

	void set_noise_scale(float p_scale);
	float get_noise_scale() const { return noise_scale; }

	void set_noise_velocity(const Vector3 &p_vel) { noise_velocity = p_vel; }
	Vector3 get_noise_velocity() const { return noise_velocity; }

	void set_priority(int p_priority) { priority = p_priority; }
	int get_priority() const { return priority; }

	void set_cull_mask(uint32_t p_mask) { cull_mask = p_mask; }
	uint32_t get_cull_mask() const { return cull_mask; }

	AABB get_world_aabb() const;
	float sample_shape_factor(const Vector3 &p_world_pos) const;
};

VARIANT_ENUM_CAST(LocalFogVolume::Shape);
VARIANT_ENUM_CAST(LocalFogVolume::BlendMode);

#endif // LOCAL_FOG_VOLUME_H
