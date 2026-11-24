#ifndef LOCAL_RTAO_H
#define LOCAL_RTAO_H

#include "scene/3d/spatial.h"

class LocalRTAO : public Spatial {
	GDCLASS(LocalRTAO, Spatial);

public:
	enum Shape {
		SHAPE_BOX = 0,
		SHAPE_SPHERE = 1,
		SHAPE_ELLIPSOID = 2,
		SHAPE_CYLINDER = 3,
		SHAPE_CONE = 4,
	};

	enum Mode {
		MODE_DIRECT = 0,
		MODE_MODULATE_TRACE = 1,
	};

	enum BlendMode {
		BLEND_MULTIPLY = 0,
		BLEND_MIN = 1,
		BLEND_MAX = 2,
		BLEND_OVERRIDE = 3,
	};

private:
	bool enabled = true;
	Mode mode = MODE_DIRECT;
	Shape shape = SHAPE_BOX;
	Vector3 extents = Vector3(2.0f, 2.0f, 2.0f);
	float occlusion = 0.5f;
	Color ao_color = Color(0.0f, 0.0f, 0.0f, 1.0f);
	float radius_multiplier = 1.0f;
	float intensity_multiplier = 1.0f;
	float edge_falloff = 0.25f;
	BlendMode blend_mode = BLEND_MULTIPLY;

	float noise_strength = 0.0f;
	float noise_scale = 1.0f;
	Vector3 noise_velocity = Vector3(0.0f, 0.0f, 0.0f);

	int priority = 0;
	uint32_t cull_mask = 0xFFFFFFFF;

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const;

public:
	LocalRTAO();

	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_mode(Mode p_mode);
	Mode get_mode() const { return mode; }

	void set_shape(Shape p_shape);
	Shape get_shape() const { return shape; }

	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const { return extents; }

	void set_occlusion(float p_occlusion);
	float get_occlusion() const { return occlusion; }

	void set_ao_color(const Color &p_color) { ao_color = p_color; }
	Color get_ao_color() const { return ao_color; }

	void set_radius_multiplier(float p_mult);
	float get_radius_multiplier() const { return radius_multiplier; }

	void set_intensity_multiplier(float p_mult);
	float get_intensity_multiplier() const { return intensity_multiplier; }

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

VARIANT_ENUM_CAST(LocalRTAO::Shape);
VARIANT_ENUM_CAST(LocalRTAO::Mode);
VARIANT_ENUM_CAST(LocalRTAO::BlendMode);

#endif // LOCAL_RTAO_H
