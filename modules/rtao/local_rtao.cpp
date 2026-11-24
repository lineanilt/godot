#include "local_rtao.h"

#include "core/math/math_funcs.h"

LocalRTAO::LocalRTAO() {
}

void LocalRTAO::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &LocalRTAO::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &LocalRTAO::is_enabled);

	ClassDB::bind_method(D_METHOD("set_mode", "mode"), &LocalRTAO::set_mode);
	ClassDB::bind_method(D_METHOD("get_mode"), &LocalRTAO::get_mode);

	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &LocalRTAO::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &LocalRTAO::get_shape);

	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &LocalRTAO::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &LocalRTAO::get_extents);

	ClassDB::bind_method(D_METHOD("set_occlusion", "occlusion"), &LocalRTAO::set_occlusion);
	ClassDB::bind_method(D_METHOD("get_occlusion"), &LocalRTAO::get_occlusion);

	ClassDB::bind_method(D_METHOD("set_ao_color", "color"), &LocalRTAO::set_ao_color);
	ClassDB::bind_method(D_METHOD("get_ao_color"), &LocalRTAO::get_ao_color);

	ClassDB::bind_method(D_METHOD("set_radius_multiplier", "multiplier"), &LocalRTAO::set_radius_multiplier);
	ClassDB::bind_method(D_METHOD("get_radius_multiplier"), &LocalRTAO::get_radius_multiplier);

	ClassDB::bind_method(D_METHOD("set_intensity_multiplier", "multiplier"), &LocalRTAO::set_intensity_multiplier);
	ClassDB::bind_method(D_METHOD("get_intensity_multiplier"), &LocalRTAO::get_intensity_multiplier);

	ClassDB::bind_method(D_METHOD("set_edge_falloff", "falloff"), &LocalRTAO::set_edge_falloff);
	ClassDB::bind_method(D_METHOD("get_edge_falloff"), &LocalRTAO::get_edge_falloff);

	ClassDB::bind_method(D_METHOD("set_blend_mode", "mode"), &LocalRTAO::set_blend_mode);
	ClassDB::bind_method(D_METHOD("get_blend_mode"), &LocalRTAO::get_blend_mode);

	ClassDB::bind_method(D_METHOD("set_noise_strength", "strength"), &LocalRTAO::set_noise_strength);
	ClassDB::bind_method(D_METHOD("get_noise_strength"), &LocalRTAO::get_noise_strength);

	ClassDB::bind_method(D_METHOD("set_noise_scale", "scale"), &LocalRTAO::set_noise_scale);
	ClassDB::bind_method(D_METHOD("get_noise_scale"), &LocalRTAO::get_noise_scale);

	ClassDB::bind_method(D_METHOD("set_noise_velocity", "velocity"), &LocalRTAO::set_noise_velocity);
	ClassDB::bind_method(D_METHOD("get_noise_velocity"), &LocalRTAO::get_noise_velocity);

	ClassDB::bind_method(D_METHOD("set_priority", "priority"), &LocalRTAO::set_priority);
	ClassDB::bind_method(D_METHOD("get_priority"), &LocalRTAO::get_priority);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &LocalRTAO::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &LocalRTAO::get_cull_mask);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mode", PROPERTY_HINT_ENUM, "Direct Occlusion,Modulate Trace"), "set_mode", "get_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shape", PROPERTY_HINT_ENUM, "Box,Sphere,Ellipsoid,Cylinder,Cone"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents"), "set_extents", "get_extents");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "occlusion", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_occlusion", "get_occlusion");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "ao_color", PROPERTY_HINT_COLOR_NO_ALPHA), "set_ao_color", "get_ao_color");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius_multiplier", PROPERTY_HINT_RANGE, "0,4,0.05,or_greater"), "set_radius_multiplier", "get_radius_multiplier");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "intensity_multiplier", PROPERTY_HINT_RANGE, "0,4,0.05,or_greater"), "set_intensity_multiplier", "get_intensity_multiplier");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "edge_falloff", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_edge_falloff", "get_edge_falloff");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_mode", PROPERTY_HINT_ENUM, "Multiply,Min,Max,Override"), "set_blend_mode", "get_blend_mode");

	ADD_GROUP("Noise", "noise_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "noise_strength", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_noise_strength", "get_noise_strength");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "noise_scale", PROPERTY_HINT_RANGE, "0.01,20,0.05,or_greater"), "set_noise_scale", "get_noise_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "noise_velocity"), "set_noise_velocity", "get_noise_velocity");

	ADD_GROUP("Visibility", "");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "priority", PROPERTY_HINT_RANGE, "-100,100,1"), "set_priority", "get_priority");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");

	BIND_ENUM_CONSTANT(SHAPE_BOX);
	BIND_ENUM_CONSTANT(SHAPE_SPHERE);
	BIND_ENUM_CONSTANT(SHAPE_ELLIPSOID);
	BIND_ENUM_CONSTANT(SHAPE_CYLINDER);
	BIND_ENUM_CONSTANT(SHAPE_CONE);

	BIND_ENUM_CONSTANT(MODE_DIRECT);
	BIND_ENUM_CONSTANT(MODE_MODULATE_TRACE);

	BIND_ENUM_CONSTANT(BLEND_MULTIPLY);
	BIND_ENUM_CONSTANT(BLEND_MIN);
	BIND_ENUM_CONSTANT(BLEND_MAX);
	BIND_ENUM_CONSTANT(BLEND_OVERRIDE);
}

void LocalRTAO::_validate_property(PropertyInfo &property) const {
	if (mode == MODE_DIRECT && (property.name == "radius_multiplier" || property.name == "intensity_multiplier")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
	if (mode == MODE_MODULATE_TRACE && (property.name == "occlusion" || property.name == "ao_color" || property.name == "blend_mode")) {
		property.usage = PROPERTY_USAGE_NOEDITOR;
	}
}

void LocalRTAO::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	update_gizmo();
}

void LocalRTAO::set_mode(Mode p_mode) {
	mode = p_mode;
	_change_notify();
}

void LocalRTAO::set_shape(Shape p_shape) {
	shape = p_shape;
	update_gizmo();
}

void LocalRTAO::set_extents(const Vector3 &p_extents) {
	extents = Vector3(MAX(p_extents.x, 0.01f), MAX(p_extents.y, 0.01f), MAX(p_extents.z, 0.01f));
	update_gizmo();
}

void LocalRTAO::set_occlusion(float p_occlusion) {
	occlusion = CLAMP(p_occlusion, 0.0f, 1.0f);
}

void LocalRTAO::set_radius_multiplier(float p_mult) {
	radius_multiplier = MAX(p_mult, 0.0f);
}

void LocalRTAO::set_intensity_multiplier(float p_mult) {
	intensity_multiplier = MAX(p_mult, 0.0f);
}

void LocalRTAO::set_edge_falloff(float p_falloff) {
	edge_falloff = CLAMP(p_falloff, 0.0f, 1.0f);
}

void LocalRTAO::set_noise_strength(float p_strength) {
	noise_strength = CLAMP(p_strength, 0.0f, 1.0f);
}

void LocalRTAO::set_noise_scale(float p_scale) {
	noise_scale = MAX(p_scale, 0.001f);
}

AABB LocalRTAO::get_world_aabb() const {
	Vector3 e = extents;
	if (shape == SHAPE_SPHERE) {
		float r = MAX(e.x, MAX(e.y, e.z));
		e = Vector3(r, r, r);
	}
	AABB local(-e, e * 2.0f);
	return get_global_transform().xform(local);
}

float LocalRTAO::sample_shape_factor(const Vector3 &p_world_pos) const {
	if (!enabled) {
		return 0.0f;
	}
	Vector3 local = get_global_transform().affine_inverse().xform(p_world_pos);
	Vector3 e = extents;
	if (shape == SHAPE_SPHERE) {
		float r = MAX(e.x, MAX(e.y, e.z));
		e = Vector3(r, r, r);
	}
	Vector3 n(
			local.x / MAX(e.x, 0.0001f),
			local.y / MAX(e.y, 0.0001f),
			local.z / MAX(e.z, 0.0001f));

	float d = 1.0f;
	switch (shape) {
		case SHAPE_BOX:
			d = MAX(ABS(n.x), MAX(ABS(n.y), ABS(n.z)));
			break;
		case SHAPE_SPHERE:
		case SHAPE_ELLIPSOID:
			d = n.length();
			break;
		case SHAPE_CYLINDER:
			d = MAX(Math::sqrt(n.x * n.x + n.z * n.z), ABS(n.y));
			break;
		case SHAPE_CONE: {
			float ty = CLAMP(0.5f - n.y * 0.5f, 0.0f, 1.0f);
			float r_cone = MAX(ty, 0.001f);
			d = MAX(Math::sqrt(n.x * n.x + n.z * n.z) / r_cone, ABS(n.y));
		} break;
	}

	if (d >= 1.0f) {
		return 0.0f;
	}
	if (edge_falloff <= 0.001f) {
		return 1.0f;
	}
	float edge0 = MAX(0.0f, 1.0f - edge_falloff);
	return 1.0f - Math::smoothstep(edge0, 1.0f, d);
}
