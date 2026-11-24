#ifndef TAA_SHADERS_H
#define TAA_SHADERS_H

namespace TAAShaders {

static const char *COMPOSITE_CANVAS_SHADER = R"SHADER(
shader_type canvas_item;
render_mode unshaded, blend_mix;

uniform sampler2D u_history_tex;
uniform sampler2D u_velocity_tex;
uniform float u_history_valid = 0.0;
uniform float u_history_blend = 0.9;
uniform float u_motion_rejection = 0.35;
uniform float u_variance_gamma = 1.25;
uniform float u_sharpen_intensity = 0.35;
uniform bool u_catmull_rom = true;
uniform bool u_variance_clipping = true;
uniform float u_keep_3d_linear = 0.0;
uniform float u_vflip = 0.0;
uniform vec2 u_jitter_px = vec2(0.0);
uniform int u_debug_mode = 0;

vec3 srgb_to_linear(vec3 c) {
	c = max(c, vec3(0.0));
	return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

vec3 linear_to_srgb(vec3 c) {
	c = max(c, vec3(0.0));
	return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}

vec3 rgb_to_ycocg(vec3 c) {
	return vec3(
		 0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
		 0.5  * c.r - 0.5 * c.b,
		-0.25 * c.r + 0.5 * c.g - 0.25 * c.b
	);
}

vec3 ycocg_to_rgb(vec3 c) {
	float tmp = c.x - c.z;
	return max(vec3(tmp + c.y, c.x + c.z, tmp - c.y), vec3(0.0));
}

float karis_weight(vec3 lin) {
	float luma = dot(lin, vec3(0.2126, 0.7152, 0.0722));
	return 1.0 / (1.0 + max(luma, 0.0));
}

vec3 clip_aabb(vec3 aabb_min, vec3 aabb_max, vec3 history) {
	vec3 p_clip = 0.5 * (aabb_max + aabb_min);
	vec3 e_clip = 0.5 * (aabb_max - aabb_min) + 1e-5;
	vec3 v_clip = history - p_clip;
	vec3 v_unit = v_clip / e_clip;
	vec3 a_unit = abs(v_unit);
	float ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
	if (ma_unit > 1.0) {
		return p_clip + v_clip / ma_unit;
	}
	return history;
}

vec3 sample_history_catmull_rom(sampler2D tex, vec2 uv) {
	vec2 tex_size = vec2(textureSize(tex, 0));
	vec2 inv_tex_size = 1.0 / max(tex_size, vec2(1.0));
	vec2 sample_pos = uv * tex_size;
	vec2 tc1 = floor(sample_pos - 0.5) + 0.5;
	vec2 f = sample_pos - tc1;
	vec2 f2 = f * f;
	vec2 f3 = f2 * f;

	vec2 w0 = f2 - 0.5 * (f3 + f);
	vec2 w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
	vec2 w3 = 0.5 * (f3 - f2);
	vec2 w2 = 1.0 - w0 - w1 - w3;

	vec2 w12 = w1 + w2;
	vec2 tc0 = (tc1 - 1.0) * inv_tex_size;
	vec2 tc3 = (tc1 + 2.0) * inv_tex_size;
	vec2 tc12 = (tc1 + w2 / max(w12, vec2(1e-5))) * inv_tex_size;

	vec3 res =
		textureLod(tex, vec2(tc12.x, tc0.y), 0.0).rgb * (w12.x * w0.y) +
		textureLod(tex, vec2(tc0.x, tc12.y), 0.0).rgb * (w0.x * w12.y) +
		textureLod(tex, vec2(tc12.x, tc12.y), 0.0).rgb * (w12.x * w12.y) +
		textureLod(tex, vec2(tc3.x, tc12.y), 0.0).rgb * (w3.x * w12.y) +
		textureLod(tex, vec2(tc12.x, tc3.y), 0.0).rgb * (w12.x * w3.y);

	float w_sum = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
	return max(res / max(w_sum, 1e-5), vec3(0.0));
}

vec2 find_closest_velocity(vec2 m_uv, out float out_speed_px) {
	vec2 vel_texel = 1.0 / max(vec2(textureSize(u_velocity_tex, 0)), vec2(1.0));
	float best_z = 1e20;
	vec2 best_vel = vec2(0.0);
	out_speed_px = 0.0;

	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			vec2 suv = clamp(m_uv + vec2(float(dx), float(dy)) * vel_texel, vec2(0.0), vec2(1.0));
			vec4 v = textureLod(u_velocity_tex, suv, 0.0);
			if (v.b < best_z) {
				best_z = v.b;
				best_vel = v.rg;
				out_speed_px = v.a;
			}
		}
	}

	if (u_vflip > 0.5) {
		best_vel.y = -best_vel.y;
	}
	return best_vel;
}

void fragment() {
	vec2 m_uv = (u_vflip > 0.5) ? vec2(SCREEN_UV.x, 1.0 - SCREEN_UV.y) : SCREEN_UV;
	float speed_px = 0.0;
	vec2 vel_uv = find_closest_velocity(m_uv, speed_px);

	if (u_debug_mode == 1) {
		COLOR = vec4(abs(vel_uv) * 25.0, 0.0, 1.0);
	} else if (u_debug_mode == 2) {
		COLOR = vec4(abs(u_jitter_px) + 0.25, 0.0, 1.0);
	} else {
		vec3 cur_raw = textureLod(SCREEN_TEXTURE, SCREEN_UV, 0.0).rgb;
		vec3 cur_lin = (u_keep_3d_linear > 0.5) ? cur_raw : srgb_to_linear(cur_raw);

		if (u_history_valid < 0.5) {
			vec3 init_rgb = (u_keep_3d_linear > 0.5) ? cur_lin : linear_to_srgb(cur_lin);
			COLOR = vec4(init_rgb, 1.0);
		} else {
			vec2 px = SCREEN_PIXEL_SIZE;
			vec3 m1 = vec3(0.0);
			vec3 m2 = vec3(0.0);
			vec3 n_min = vec3(1e10);
			vec3 n_max = vec3(-1e10);
			vec3 karis_sum = vec3(0.0);
			float karis_w_sum = 0.0;

			// 3x3 neighborhood statistics in YCoCg + RCAS neighbor bounds in linear RGB
			vec3 c_n = cur_lin;
			vec3 c_s = cur_lin;
			vec3 c_e = cur_lin;
			vec3 c_w = cur_lin;

			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					vec2 suv = clamp(SCREEN_UV + vec2(float(dx), float(dy)) * px, vec2(0.0005), vec2(0.9995));
					vec3 s_raw = textureLod(SCREEN_TEXTURE, suv, 0.0).rgb;
					vec3 s_lin = (u_keep_3d_linear > 0.5) ? s_raw : srgb_to_linear(s_raw);
					vec3 s_yc = rgb_to_ycocg(s_lin);

					m1 += s_yc;
					m2 += s_yc * s_yc;
					n_min = min(n_min, s_yc);
					n_max = max(n_max, s_yc);

					float kw = karis_weight(s_lin);
					karis_sum += s_yc * kw;
					karis_w_sum += kw;

					if (dx == 0 && dy == -1) c_n = s_lin;
					else if (dx == 0 && dy == 1) c_s = s_lin;
					else if (dx == 1 && dy == 0) c_e = s_lin;
					else if (dx == -1 && dy == 0) c_w = s_lin;
				}
			}

			vec2 prev_uv = SCREEN_UV - vel_uv;
			bool on_screen = (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0);

			vec3 resolved_lin = cur_lin;
			float was_clipped = 0.0;

			if (on_screen) {
				vec3 hist_raw = u_catmull_rom
					? sample_history_catmull_rom(u_history_tex, prev_uv)
					: textureLod(u_history_tex, prev_uv, 0.0).rgb;
				vec3 hist_lin = (u_keep_3d_linear > 0.5) ? hist_raw : srgb_to_linear(hist_raw);
				vec3 hist_yc = rgb_to_ycocg(hist_lin);

				vec3 box_min = n_min;
				vec3 box_max = n_max;
				if (u_variance_clipping) {
					vec3 mean = m1 / 9.0;
					vec3 stddev = sqrt(max(m2 / 9.0 - mean * mean, vec3(0.0)));
					box_min = max(n_min, mean - stddev * u_variance_gamma);
					box_max = min(n_max, mean + stddev * u_variance_gamma);
				}

				vec3 clipped_yc = clip_aabb(box_min, box_max, hist_yc);
				was_clipped = clamp(length(clipped_yc - hist_yc) * 4.0, 0.0, 1.0);

				vec3 cur_yc = rgb_to_ycocg(cur_lin);
				float w_hist = clamp(u_history_blend, 0.0, 0.97);
				w_hist *= 1.0 - clamp((speed_px / 24.0) * u_motion_rejection, 0.0, 0.65);

				// Karis luma-weighted blend reduces specular shimmering
				vec3 clipped_lin = ycocg_to_rgb(clipped_yc);
				float kw_cur = karis_weight(cur_lin);
				float kw_hist = karis_weight(clipped_lin);
				float a_cur = (1.0 - w_hist) * kw_cur;
				float a_hist = w_hist * kw_hist;
				resolved_lin = (cur_lin * a_cur + clipped_lin * a_hist) / max(a_cur + a_hist, 1e-5);
			}

			// Contrast-adaptive RCAS-style sharpening pass
			if (u_sharpen_intensity > 0.001) {
				vec3 min_ring = min(min(c_n, c_s), min(c_e, c_w));
				vec3 max_ring = max(max(c_n, c_s), max(c_e, c_w));
				vec3 avg_ring = (c_n + c_s + c_e + c_w) * 0.25;
				vec3 sharp = resolved_lin + (resolved_lin - avg_ring) * (u_sharpen_intensity * 1.25);
				resolved_lin = clamp(sharp, min(resolved_lin, min_ring), max(resolved_lin, max_ring));
			}

			if (u_debug_mode == 3) {
				vec3 diff = abs(resolved_lin - cur_lin) * 6.0;
				COLOR = vec4(clamp(diff + vec3(was_clipped * 0.35, 0.0, 0.0), 0.0, 1.0), 1.0);
			} else {
				vec3 out_rgb = (u_keep_3d_linear > 0.5) ? resolved_lin : linear_to_srgb(resolved_lin);
				COLOR = vec4(clamp(out_rgb, vec3(0.0), vec3(1.0)), 1.0);
			}
		}
	}
}
)SHADER";

} // namespace TAAShaders

#endif // TAA_SHADERS_H
