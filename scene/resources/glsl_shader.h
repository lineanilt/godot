#ifndef GLSL_SHADER_H
#define GLSL_SHADER_H

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "scene/resources/shader.h"

class GLSLShader : public Shader {
	GDCLASS(GLSLShader, Shader);
	OBJ_SAVE_TYPE(GLSLShader);

private:
	String glsl_code;

protected:
	static void _bind_methods();

public:
	virtual void set_code(const String &p_code);
	virtual String get_code() const;

	GLSLShader();
	~GLSLShader();
};

class ResourceFormatLoaderGLSLShader : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_no_subresource_cache = false);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

class ResourceFormatSaverGLSLShader : public ResourceFormatSaver {
public:
	virtual Error save(const String &p_path, const RES &p_resource, uint32_t p_flags = 0);
	virtual void get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const;
	virtual bool recognize(const RES &p_resource) const;
};

#endif // GLSL_SHADER_H
