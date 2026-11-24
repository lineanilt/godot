#include "volumetric_fog_focus.h"

VolumetricFogFocus::VolumetricFogFocus() {
}

void VolumetricFogFocus::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &VolumetricFogFocus::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &VolumetricFogFocus::is_enabled);

	ClassDB::bind_method(D_METHOD("set_priority", "priority"), &VolumetricFogFocus::set_priority);
	ClassDB::bind_method(D_METHOD("get_priority"), &VolumetricFogFocus::get_priority);

	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &VolumetricFogFocus::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &VolumetricFogFocus::get_radius);

	ClassDB::bind_method(D_METHOD("set_depth_range", "depth_range"), &VolumetricFogFocus::set_depth_range);
	ClassDB::bind_method(D_METHOD("get_depth_range"), &VolumetricFogFocus::get_depth_range);

	ClassDB::bind_method(D_METHOD("set_camera_bias", "bias"), &VolumetricFogFocus::set_camera_bias);
	ClassDB::bind_method(D_METHOD("get_camera_bias"), &VolumetricFogFocus::get_camera_bias);

	ClassDB::bind_method(D_METHOD("set_density_multiplier", "multiplier"), &VolumetricFogFocus::set_density_multiplier);
	ClassDB::bind_method(D_METHOD("get_density_multiplier"), &VolumetricFogFocus::get_density_multiplier);

	ClassDB::bind_method(D_METHOD("set_focus_falloff", "falloff"), &VolumetricFogFocus::set_focus_falloff);
	ClassDB::bind_method(D_METHOD("get_focus_falloff"), &VolumetricFogFocus::get_focus_falloff);

	ClassDB::bind_method(D_METHOD("set_max_ray_distance_override", "distance"), &VolumetricFogFocus::set_max_ray_distance_override);
	ClassDB::bind_method(D_METHOD("get_max_ray_distance_override"), &VolumetricFogFocus::get_max_ray_distance_override);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "priority", PROPERTY_HINT_RANGE, "-100,100,1"), "set_priority", "get_priority");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius", PROPERTY_HINT_RANGE, "1,500,0.5,or_greater"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "depth_range", PROPERTY_HINT_RANGE, "5,1000,1,or_greater"), "set_depth_range", "get_depth_range");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "camera_bias", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_camera_bias", "get_camera_bias");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "density_multiplier", PROPERTY_HINT_RANGE, "0,8,0.05,or_greater"), "set_density_multiplier", "get_density_multiplier");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "focus_falloff", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_focus_falloff", "get_focus_falloff");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "max_ray_distance_override", PROPERTY_HINT_RANGE, "0,2000,1,or_greater"), "set_max_ray_distance_override", "get_max_ray_distance_override");
}

void VolumetricFogFocus::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	update_gizmo();
}

void VolumetricFogFocus::set_radius(float p_radius) {
	radius = MAX(p_radius, 0.5f);
	update_gizmo();
}

void VolumetricFogFocus::set_depth_range(float p_range) {
	depth_range = MAX(p_range, 1.0f);
	update_gizmo();
}

void VolumetricFogFocus::set_camera_bias(float p_bias) {
	camera_bias = CLAMP(p_bias, 0.0f, 1.0f);
}

void VolumetricFogFocus::set_density_multiplier(float p_mult) {
	density_multiplier = MAX(p_mult, 0.0f);
}

void VolumetricFogFocus::set_focus_falloff(float p_falloff) {
	focus_falloff = CLAMP(p_falloff, 0.0f, 1.0f);
}

void VolumetricFogFocus::set_max_ray_distance_override(float p_dist) {
	max_ray_distance_override = MAX(p_dist, 0.0f);
}
