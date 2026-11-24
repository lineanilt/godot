#ifndef MOTION_BLUR_SHADERS_H
#define MOTION_BLUR_SHADERS_H

namespace MotionBlurShaders {

static const char *TILE_MAX_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutTileMax;

uniform sampler2D uVelocityTex;
uniform int uTileSize;

void main() {
	ivec2 tile = ivec2(gl_GlobalInvocationID.xy);
	ivec2 outSz = imageSize(uOutTileMax);
	if (tile.x >= outSz.x || tile.y >= outSz.y) {
		return;
	}

	ivec2 velSz = textureSize(uVelocityTex, 0);
	ivec2 base = tile * uTileSize;

	vec2 bestVel = vec2(0.0);
	float bestSpeed = 0.0;

	for (int dy = 0; dy < uTileSize; dy++) {
		int y = base.y + dy;
		if (y >= velSz.y) {
			break;
		}
		for (int dx = 0; dx < uTileSize; dx++) {
			int x = base.x + dx;
			if (x >= velSz.x) {
				break;
			}
			vec4 v = texelFetch(uVelocityTex, ivec2(x, y), 0);
			if (v.a > bestSpeed) {
				bestSpeed = v.a;
				bestVel = v.rg;
			}
		}
	}

	imageStore(uOutTileMax, tile, vec4(bestVel, 0.0, bestSpeed));
}
)GLSL";

static const char *NEIGHBOR_MAX_COMPUTE_GLSL = R"GLSL(
#version 430 core
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (rgba16f, binding = 0) writeonly uniform image2D uOutNeighborMax;

uniform sampler2D uTileMaxTex;
uniform int uTileRadius;

void main() {
	ivec2 tile = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(uOutNeighborMax);
	if (tile.x >= sz.x || tile.y >= sz.y) {
		return;
	}

	vec2 bestVel = vec2(0.0);
	float bestSpeed = 0.0;

	for (int dy = -uTileRadius; dy <= uTileRadius; dy++) {
		for (int dx = -uTileRadius; dx <= uTileRadius; dx++) {
			ivec2 nt = clamp(tile + ivec2(dx, dy), ivec2(0), sz - 1);
			vec4 v = texelFetch(uTileMaxTex, nt, 0);
			if (v.a > bestSpeed) {
				bestSpeed = v.a;
				bestVel = v.rg;
			}
		}
	}

	imageStore(uOutNeighborMax, tile, vec4(bestVel, 0.0, bestSpeed));
}
)GLSL";

static const char *COMPOSITE_CANVAS_SHADER = R"SHADER(
shader_type canvas_item;
render_mode unshaded, blend_mix;

uniform sampler2D u_velocity_tex;
uniform sampler2D u_neighbor_max_tex;
uniform vec2 u_viewport_size = vec2(1280.0, 720.0);
uniform int u_samples = 15;
uniform float u_jitter = 1.0;
uniform float u_shutter_offset = 0.0;
uniform float u_depth_softness = 0.15;
uniform float u_min_velocity_px = 0.5;
uniform float u_max_velocity_px = 64.0;
uniform float u_keep_3d_linear = 0.0;
uniform float u_vflip = 0.0;
uniform int u_debug_mode = 0;

float ign(vec2 p) {
	return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

vec3 srgb_to_linear(vec3 c) {
	c = max(c, vec3(0.0));
	return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

vec3 linear_to_srgb(vec3 c) {
	c = max(c, vec3(0.0));
	return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

float is_closer(float z_candidate, float z_ref, float extent) {
	return clamp(1.0 - (z_candidate - z_ref) / max(extent, 0.001), 0.0, 1.0);
}

float cone(float dist_px, float speed_px) {
	return clamp(1.0 - dist_px / max(speed_px, 0.5), 0.0, 1.0);
}

float cylinder(float dist_px, float speed_px) {
	return 1.0 - smoothstep(0.95 * speed_px, 1.05 * speed_px, dist_px);
}

void fragment() {
	vec2 m_uv = (u_vflip > 0.5) ? vec2(SCREEN_UV.x, 1.0 - SCREEN_UV.y) : SCREEN_UV;
	vec4 center_vel = textureLod(u_velocity_tex, m_uv, 0.0);
	vec4 nmax_vel = textureLod(u_neighbor_max_tex, m_uv, 0.0);

	vec2 v_c = center_vel.rg;
	vec2 v_n = nmax_vel.rg;
	if (u_vflip > 0.5) {
		v_c.y = -v_c.y;
		v_n.y = -v_n.y;
	}

	if (u_debug_mode == 1) {
		COLOR = vec4(abs(v_c) * 20.0, 0.0, 1.0);
	} else if (u_debug_mode == 2) {
		COLOR = vec4(abs(v_n) * 20.0, 0.0, 1.0);
	} else if (u_debug_mode == 3) {
		float sp = clamp(center_vel.a / max(u_max_velocity_px, 1.0), 0.0, 1.0);
		COLOR = vec4(vec3(sp), 1.0);
	} else {
		vec4 center_col = textureLod(SCREEN_TEXTURE, SCREEN_UV, 0.0);
		float speed_n = length(v_n * u_viewport_size);
		if (speed_n < u_min_velocity_px) {
			COLOR = vec4(center_col.rgb, 1.0);
		} else {
			float z_c = center_vel.b;
			float speed_c = length(v_c * u_viewport_size);
			float eff_extent = u_depth_softness * max(0.5, z_c * 0.05);

			vec3 center_lin = (u_keep_3d_linear > 0.5) ? center_col.rgb : srgb_to_linear(center_col.rgb);
			float total_weight = 1.0 / max(speed_c, 0.5);
			vec3 accum = center_lin * total_weight;

			float jitter_val = (ign(FRAGCOORD.xy) - 0.5) * u_jitter;
			int sample_count = clamp(u_samples, 3, 32);

			for (int i = 0; i < 32; i++) {
				if (i >= sample_count) {
					break;
				}

				float t = (float(i) + 0.5 + jitter_val) / float(sample_count) - 0.5 + u_shutter_offset;
				vec2 dir = ((i & 1) == 1 && speed_c > 0.5) ? v_c : v_n;
				float dist_px = abs(t) * length(dir * u_viewport_size);

				vec2 s_uv = clamp(SCREEN_UV + dir * t, vec2(0.0005), vec2(0.9995));
				vec2 s_m_uv = (u_vflip > 0.5) ? vec2(s_uv.x, 1.0 - s_uv.y) : s_uv;

				vec4 s_col = textureLod(SCREEN_TEXTURE, s_uv, 0.0);
				vec3 s_lin = (u_keep_3d_linear > 0.5) ? s_col.rgb : srgb_to_linear(s_col.rgb);

				vec4 s_vel = textureLod(u_velocity_tex, s_m_uv, 0.0);
				float z_y = s_vel.b;
				float speed_y = length(s_vel.rg * u_viewport_size);

				float f = is_closer(z_y, z_c, eff_extent);
				float b = is_closer(z_c, z_y, eff_extent);

				float w_y = f * cone(dist_px, speed_y)
						  + b * cone(dist_px, speed_c)
						  + cylinder(dist_px, speed_y) * cylinder(dist_px, speed_c) * 2.0;

				accum += s_lin * w_y;
				total_weight += w_y;
			}

			vec3 out_lin = accum / max(total_weight, 1e-5);
			vec3 out_rgb = (u_keep_3d_linear > 0.5) ? out_lin : linear_to_srgb(out_lin);
			COLOR = vec4(out_rgb, 1.0);
		}
	}
}
)SHADER";

} // namespace MotionBlurShaders

#endif // MOTION_BLUR_SHADERS_H
