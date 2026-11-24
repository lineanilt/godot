#include "local_fog_volume.h"

#include "core/math/math_funcs.h"

LocalFogVolume::LocalFogVolume() {
}

void LocalFogVolume::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &LocalFogVolume::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &LocalFogVolume::is_enabled);

	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &LocalFogVolume::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &LocalFogVolume::get_shape);

	ClassDB::bind_method(D_METHOD("set_extents", "extents"), &LocalFogVolume::set_extents);
	ClassDB::bind_method(D_METHOD("get_extents"), &LocalFogVolume::get_extents);

	ClassDB::bind_method(D_METHOD("set_density", "density"), &LocalFogVolume::set_density);
	ClassDB::bind_method(D_METHOD("get_density"), &LocalFogVolume::get_density);

	ClassDB::bind_method(D_METHOD("set_albedo", "albedo"), &LocalFogVolume::set_albedo);
	ClassDB::bind_method(D_METHOD("get_albedo"), &LocalFogVolume::get_albedo);

	ClassDB::bind_method(D_METHOD("set_emission", "emission"), &LocalFogVolume::set_emission);
	ClassDB::bind_method(D_METHOD("get_emission"), &LocalFogVolume::get_emission);

	ClassDB::bind_method(D_METHOD("set_emission_energy", "energy"), &LocalFogVolume::set_emission_energy);
	ClassDB::bind_method(D_METHOD("get_emission_energy"), &LocalFogVolume::get_emission_energy);

	ClassDB::bind_method(D_METHOD("set_edge_falloff", "falloff"), &LocalFogVolume::set_edge_falloff);
	ClassDB::bind_method(D_METHOD("get_edge_falloff"), &LocalFogVolume::get_edge_falloff);

	ClassDB::bind_method(D_METHOD("set_blend_mode", "mode"), &LocalFogVolume::set_blend_mode);
	ClassDB::bind_method(D_METHOD("get_blend_mode"), &LocalFogVolume::get_blend_mode);

	ClassDB::bind_method(D_METHOD("set_noise_strength", "strength"), &LocalFogVolume::set_noise_strength);
	ClassDB::bind_method(D_METHOD("get_noise_strength"), &LocalFogVolume::get_noise_strength);

	ClassDB::bind_method(D_METHOD("set_noise_scale", "scale"), &LocalFogVolume::set_noise_scale);
	ClassDB::bind_method(D_METHOD("get_noise_scale"), &LocalFogVolume::get_noise_scale);

	ClassDB::bind_method(D_METHOD("set_noise_velocity", "velocity"), &LocalFogVolume::set_noise_velocity);
	ClassDB::bind_method(D_METHOD("get_noise_velocity"), &LocalFogVolume::get_noise_velocity);

	ClassDB::bind_method(D_METHOD("set_priority", "priority"), &LocalFogVolume::set_priority);
	ClassDB::bind_method(D_METHOD("get_priority"), &LocalFogVolume::get_priority);

	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &LocalFogVolume::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &LocalFogVolume::get_cull_mask);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "shape", PROPERTY_HINT_ENUM, "Box,Sphere,Ellipsoid,Cylinder,Cone"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents"), "set_extents", "get_extents");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "density", PROPERTY_HINT_RANGE, "0,16,0.01,or_greater"), "set_density", "get_density");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "albedo", PROPERTY_HINT_COLOR_NO_ALPHA), "set_albedo", "get_albedo");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "emission", PROPERTY_HINT_COLOR_NO_ALPHA), "set_emission", "get_emission");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "emission_energy", PROPERTY_HINT_RANGE, "0,16,0.01,or_greater"), "set_emission_energy", "get_emission_energy");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "edge_falloff", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_edge_falloff", "get_edge_falloff");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_mode", PROPERTY_HINT_ENUM, "Additive,Max,Multiply,Subtract"), "set_blend_mode", "get_blend_mode");

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

	BIND_ENUM_CONSTANT(BLEND_ADDITIVE);
	BIND_ENUM_CONSTANT(BLEND_MAX);
	BIND_ENUM_CONSTANT(BLEND_MULTIPLY);
	BIND_ENUM_CONSTANT(BLEND_SUBTRACT);
}

void LocalFogVolume::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	update_gizmo();
}

void LocalFogVolume::set_shape(Shape p_shape) {
	shape = p_shape;
	update_gizmo();
}

void LocalFogVolume::set_extents(const Vector3 &p_extents) {
	extents = Vector3(MAX(p_extents.x, 0.01f), MAX(p_extents.y, 0.01f), MAX(p_extents.z, 0.01f));
	update_gizmo();
}

void LocalFogVolume::set_density(float p_density) {
	density = MAX(p_density, 0.0f);
}

void LocalFogVolume::set_emission_energy(float p_energy) {
	emission_energy = MAX(p_energy, 0.0f);
}

void LocalFogVolume::set_edge_falloff(float p_falloff) {
	edge_falloff = CLAMP(p_falloff, 0.0f, 1.0f);
}

void LocalFogVolume::set_noise_strength(float p_strength) {
	noise_strength = CLAMP(p_strength, 0.0f, 1.0f);
}

void LocalFogVolume::set_noise_scale(float p_scale) {
	noise_scale = MAX(p_scale, 0.001f);
}

AABB LocalFogVolume::get_world_aabb() const {
	Vector3 e = extents;
	if (shape == SHAPE_SPHERE) {
		float r = MAX(e.x, MAX(e.y, e.z));
		e = Vector3(r, r, r);
	}
	AABB local(-e, e * 2.0f);
	return get_global_transform().xform(local);
}

float LocalFogVolume::sample_shape_factor(const Vector3 &p_world_pos) const {
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
