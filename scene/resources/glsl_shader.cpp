#include "glsl_shader.h"

#include "core/os/file_access.h"

#include "servers/visual_server.h"

void GLSLShader::set_code(const String &p_code) {
	if (glsl_code == p_code) {
		return;
	}
	glsl_code = p_code;
	if (get_rid().is_valid() && VisualServer::get_singleton()) {
		VisualServer::get_singleton()->shader_set_code(get_rid(), String("//GODOT_RAW_GLSL\n") + glsl_code);
	}
	_mark_params_dirty();
	emit_changed();
}

String GLSLShader::get_code() const {
	return glsl_code;
}

void GLSLShader::_bind_methods() {
}

GLSLShader::GLSLShader() {
	glsl_code =
			"// render_mode cull_back, depth_draw_opaque;\n"
			"\n"
			"[vertex]\n"
			"layout(location = 0) in vec3 a_position;\n"
			"layout(location = 1) in vec3 a_normal;\n"
			"layout(location = 4) in vec2 a_uv;\n"
			"\n"
			"uniform mat4 u_mvp_matrix;\n"
			"uniform mat3 u_normal_matrix;\n"
			"\n"
			"out vec3 v_normal;\n"
			"out vec2 v_uv;\n"
			"\n"
			"void main() {\n"
			"\tv_normal = normalize(u_normal_matrix * a_normal);\n"
			"\tv_uv = a_uv;\n"
			"\tgl_Position = u_mvp_matrix * vec4(a_position, 1.0);\n"
			"}\n"
			"\n"
			"[fragment]\n"
			"in vec3 v_normal;\n"
			"in vec2 v_uv;\n"
			"\n"
			"layout(location = 0) out vec4 frag_color;\n"
			"\n"
			"void main() {\n"
			"\tvec3 n = normalize(v_normal) * 0.5 + 0.5;\n"
			"\tfrag_color = vec4(n, 1.0);\n"
			"}\n";

	if (get_rid().is_valid() && VisualServer::get_singleton()) {
		VisualServer::get_singleton()->shader_set_code(get_rid(), String("//GODOT_RAW_GLSL\n") + glsl_code);
	}
}

GLSLShader::~GLSLShader() {
}

RES ResourceFormatLoaderGLSLShader::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_no_subresource_cache) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V_MSG(err != OK, RES(), "Cannot open GLSL file '" + p_path + "'.");

	String str = f->get_as_utf8_string();
	memdelete(f);

	Ref<GLSLShader> shader;
	shader.instance();
	shader->set_code(str);

	if (r_error) {
		*r_error = OK;
	}
	return shader;
}

void ResourceFormatLoaderGLSLShader::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("glsl");
	p_extensions->push_back("comp");
	p_extensions->push_back("vert");
	p_extensions->push_back("frag");
}

bool ResourceFormatLoaderGLSLShader::handles_type(const String &p_type) const {
	return p_type == "GLSLShader";
}

String ResourceFormatLoaderGLSLShader::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "glsl" || el == "comp" || el == "vert" || el == "frag") {
		return "GLSLShader";
	}
	return "";
}

Error ResourceFormatSaverGLSLShader::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {
	Ref<GLSLShader> shader = p_resource;
	ERR_FAIL_COND_V(shader.is_null(), ERR_INVALID_PARAMETER);

	Error err;
	FileAccess *file = FileAccess::open(p_path, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save GLSL file '" + p_path + "'.");

	file->store_string(shader->get_code());
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		memdelete(file);
		return ERR_CANT_CREATE;
	}
	memdelete(file);

	return OK;
}

void ResourceFormatSaverGLSLShader::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<GLSLShader>(p_resource.ptr())) {
		p_extensions->push_back("glsl");
		p_extensions->push_back("comp");
		p_extensions->push_back("vert");
		p_extensions->push_back("frag");
	}
}

bool ResourceFormatSaverGLSLShader::recognize(const RES &p_resource) const {
	return Object::cast_to<GLSLShader>(p_resource.ptr()) != nullptr;
}
