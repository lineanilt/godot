/* clang-format off */
[vertex]

layout(location = 0) in highp vec4 vertex_attrib;
/* clang-format on */
layout(location = 4) in vec2 uv_in;

out vec2 uv_interp;
out vec2 pos_interp;

void main() {
	uv_interp = uv_in;
	gl_Position = vertex_attrib;
	pos_interp.xy = gl_Position.xy;
}

/* clang-format off */
[fragment]

in vec2 uv_interp;
/* clang-format on */
in vec2 pos_interp;

uniform sampler2D source_diffuse; //texunit:0
uniform sampler2D source_normal_roughness; //texunit:1
uniform sampler2D source_depth; //texunit:2

uniform float camera_z_near;
uniform float camera_z_far;

uniform vec2 viewport_size;
uniform vec2 pixel_size;

uniform float filter_mipmap_levels;

uniform mat4 inverse_projection;
uniform mat4 projection;

uniform int num_steps;
uniform float depth_tolerance;
uniform float distance_fade;
uniform float curve_fade_in;

layout(location = 0) out vec4 frag_color;

#define M_PI 3.14159265358979323846

vec2 view_to_screen(vec3 view_pos, out float w) {
	vec4 projected = projection * vec4(view_pos, 1.0);
	projected.xyz /= projected.w;
	projected.xy = projected.xy * 0.5 + 0.5;
	w = projected.w;
	return projected.xy;
}

float fetch_linear_depth(vec2 screen_pos) {
	float d = texture(source_depth, screen_pos * pixel_size).r * 2.0 - 1.0;
#ifdef USE_ORTHOGONAL_PROJECTION
	d = ((d + (camera_z_far + camera_z_near) / (camera_z_far - camera_z_near)) * (camera_z_far - camera_z_near)) / 2.0;
#else
	d = 2.0 * camera_z_near * camera_z_far / (camera_z_far + camera_z_near - d * (camera_z_far - camera_z_near));
#endif
	return -d;
}

void main() {
	vec4 diffuse = texture(source_diffuse, uv_interp);
	vec4 normal_roughness = texture(source_normal_roughness, uv_interp);

	vec3 normal = normal_roughness.xyz * 2.0 - 1.0;
	float roughness = clamp(normal_roughness.w, 0.0, 1.0);

	float depth_tex = texture(source_depth, uv_interp).r;

	vec4 world_pos = inverse_projection * vec4(uv_interp * 2.0 - 1.0, depth_tex * 2.0 - 1.0, 1.0);
	vec3 vertex = world_pos.xyz / world_pos.w;

#ifdef USE_ORTHOGONAL_PROJECTION
	vec3 view_dir = vec3(0.0, 0.0, -1.0);
#else
	vec3 view_dir = normalize(vertex);
#endif

	// Analytical specular reflection vector (no stochastic microfacet jitter)
	vec3 ray_dir = normalize(reflect(view_dir, normal));

	if (dot(ray_dir, normal) < 0.001) {
		frag_color = vec4(0.0);
		return;
	}

	// Normal lift to avoid self-intersecting the surface on step 0
	vec3 ray_origin = vertex + normal * max(0.03, -vertex.z * 0.005);

	float ray_len = (ray_origin.z + ray_dir.z * camera_z_far) > -camera_z_near ? (-camera_z_near - ray_origin.z) / ray_dir.z : camera_z_far;
	vec3 ray_end = ray_origin + ray_dir * ray_len;

	float w_begin;
	vec2 vp_line_begin = view_to_screen(ray_origin, w_begin);
	float w_end;
	vec2 vp_line_end = view_to_screen(ray_end, w_end);
	vec2 vp_line_dir = vp_line_end - vp_line_begin;

	w_begin = 1.0 / w_begin;
	w_end = 1.0 / w_end;

	float z_begin = ray_origin.z * w_begin;
	float z_end = ray_end.z * w_end;

	vec2 line_begin = vp_line_begin / pixel_size;
	vec2 line_dir = vp_line_dir / pixel_size;
	float z_dir = z_end - z_begin;
	float w_dir = w_end - w_begin;

	// Clip the line to viewport edges
	float scale_max_x = min(1.0, 0.99 * (1.0 - vp_line_begin.x) / max(1e-5, vp_line_dir.x));
	float scale_max_y = min(1.0, 0.99 * (1.0 - vp_line_begin.y) / max(1e-5, vp_line_dir.y));
	float scale_min_x = min(1.0, 0.99 * vp_line_begin.x / max(1e-5, -vp_line_dir.x));
	float scale_min_y = min(1.0, 0.99 * vp_line_begin.y / max(1e-5, -vp_line_dir.y));
	float line_clip = min(scale_max_x, scale_max_y) * min(scale_min_x, scale_min_y);
	line_dir *= line_clip;
	z_dir *= line_clip;
	w_dir *= line_clip;

	vec2 line_advance = normalize(line_dir);
	float step_size = length(line_advance) / length(line_dir);
	float z_advance = z_dir * step_size;
	float w_advance = w_dir * step_size;

	float advance_angle_adj = 1.0 / max(abs(line_advance.x), abs(line_advance.y));
	line_advance *= advance_angle_adj;
	z_advance *= advance_angle_adj;
	w_advance *= advance_angle_adj;

	// Dynamic step scaling: ensure num_steps can reach the end of the ray
	float total_pixels = length(line_dir);
	float max_possible_reach = float(num_steps) * advance_angle_adj;
	if (total_pixels > max_possible_reach) {
		float step_stride = total_pixels / max_possible_reach;
		line_advance *= step_stride;
		z_advance *= step_stride;
		w_advance *= step_stride;
	}

	// Deterministic half-step offset to center ray evaluation per cell without noise dither
	vec2 pos = line_begin + line_advance * 0.5;
	float z = z_begin + z_advance * 0.5;
	float w = w_begin + w_advance * 0.5;

	float z_from = z / w;
	float z_to = z_from;
	float depth;
	vec2 prev_pos = pos;
	float prev_z = z;
	float prev_w = w;

	bool found = false;
	float steps_taken = 0.0;

	for (int i = 0; i < num_steps; i++) {
		pos += line_advance;
		z += z_advance;
		w += w_advance;

		depth = fetch_linear_depth(pos);
		z_from = z_to;
		z_to = z / w;

		// If ray passes behind depth
		if (depth > z_to) {
			// Binary Search Refinement (4 steps)
			vec2 p_start = prev_pos;
			vec2 p_end = pos;
			float z_s = prev_z;
			float z_e = z;
			float w_s = prev_w;
			float w_e = w;

			for (int b = 0; b < 4; b++) {
				vec2 p_mid = (p_start + p_end) * 0.5;
				float z_mid = (z_s + z_e) * 0.5;
				float w_mid = (w_s + w_e) * 0.5;
				float d = fetch_linear_depth(p_mid);

				if (d > (z_mid / w_mid)) {
					p_end = p_mid;
					z_e = z_mid;
					w_e = w_mid;
				} else {
					p_start = p_mid;
					z_s = z_mid;
					w_s = w_mid;
				}
			}

			float hit_depth = fetch_linear_depth(p_end);
			float hit_ray_z = z_e / w_e;
			float dynamic_tol = max(depth_tolerance, abs((z_e - z_s) / w_e) * 2.0);

			// Check refined hit
			if ((hit_depth <= hit_ray_z + dynamic_tol) && (-hit_depth < camera_z_far)) {
				pos = p_end;
				found = true;
				break;
			}
		}

		steps_taken += 1.0;
		prev_pos = pos;
		prev_z = z;
		prev_w = w;
	}

	if (found) {
		if (any(bvec4(lessThan(pos, vec2(0.0)), greaterThan(pos, viewport_size * 0.5)))) {
			frag_color = vec4(0.0);
			return;
		}

		vec2 margin = vec2((viewport_size.x + viewport_size.y) * 0.5 * 0.05);
		vec2 margin_grad = mix(viewport_size * 0.5 - pos, pos, lessThan(pos, viewport_size * 0.25));
		float margin_blend = smoothstep(0.0, margin.x * margin.y, margin_grad.x * margin_grad.y);

		vec2 final_pos = pos;
		float grad = (steps_taken + 1.0) / float(num_steps);
		float initial_fade = curve_fade_in == 0.0 ? 1.0 : pow(clamp(grad, 0.0, 1.0), curve_fade_in);
		float fade = pow(clamp(1.0 - grad, 0.0, 1.0), distance_fade) * initial_fade;

#ifdef REFLECT_ROUGHNESS
		// Analytical screen-space cone tracing:
		// Convert perceptual roughness to GGX alpha (alpha = roughness^2)
		float alpha = roughness * roughness;
		float hit_dist = length(final_pos - line_begin);

		// Analytical cone footprint subtended in pixel space
		float cone_angle = alpha * (M_PI * 0.25);
		float cone_diameter = max(1.0, 2.0 * hit_dist * tan(cone_angle));
		float hit_mip = clamp(log2(cone_diameter), 0.0, filter_mipmap_levels - 1.0);

		// Analytical roughness attenuation (rough surfaces disperse specular energy to probe fallbacks)
		float roughness_fade = clamp(1.0 - alpha, 0.0, 1.0);

		vec4 final_color = textureLod(source_diffuse, final_pos * pixel_size, hit_mip);
		frag_color = vec4(final_color.rgb, fade * margin_blend * roughness_fade);
#else
		frag_color = vec4(textureLod(source_diffuse, final_pos * pixel_size, 0.0).rgb, fade * margin_blend);
#endif

	} else {
		frag_color = vec4(0.0);
	}
}