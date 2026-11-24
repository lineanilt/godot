#ifndef VOLUMETRIC_FOG_FOCUS_H
#define VOLUMETRIC_FOG_FOCUS_H

#include "scene/3d/spatial.h"

class VolumetricFogFocus : public Spatial {
	GDCLASS(VolumetricFogFocus, Spatial);

private:
	bool enabled = true;
	int priority = 0;
	float radius = 30.0f;
	float depth_range = 120.0f;
	float camera_bias = 0.0f;
	float density_multiplier = 1.0f;
	float focus_falloff = 0.25f;
	float max_ray_distance_override = 0.0f;

protected:
	static void _bind_methods();

public:
	VolumetricFogFocus();

	void set_enabled(bool p_enabled);
	bool is_enabled() const { return enabled; }

	void set_priority(int p_priority) { priority = p_priority; }
	int get_priority() const { return priority; }

	void set_radius(float p_radius);
	float get_radius() const { return radius; }

	void set_depth_range(float p_range);
	float get_depth_range() const { return depth_range; }

	void set_camera_bias(float p_bias);
	float get_camera_bias() const { return camera_bias; }

	void set_density_multiplier(float p_mult);
	float get_density_multiplier() const { return density_multiplier; }

	void set_focus_falloff(float p_falloff);
	float get_focus_falloff() const { return focus_falloff; }

	void set_max_ray_distance_override(float p_dist);
	float get_max_ray_distance_override() const { return max_ray_distance_override; }
};

#endif // VOLUMETRIC_FOG_FOCUS_H
