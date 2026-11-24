/**************************************************************************/
/*  shader_editor_plugin.cpp                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "shader_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/version_generated.gen.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/property_editor.h"
#include "servers/visual/shader_types.h"

namespace {

struct GLSLBuiltinFuncDoc {
	const char *name;
	const char *signature;
	bool zero_args;
};

static const GLSLBuiltinFuncDoc GLSL_BUILTIN_FUNCTIONS[] = {
	// Angle & Trigonometry
	{ "radians", "genType radians(genType degrees)", false },
	{ "degrees", "genType degrees(genType radians)", false },
	{ "sin", "genType sin(genType angle)", false },
	{ "cos", "genType cos(genType angle)", false },
	{ "tan", "genType tan(genType angle)", false },
	{ "asin", "genType asin(genType x)", false },
	{ "acos", "genType acos(genType x)", false },
	{ "atan", "genType atan(genType y, genType x) | atan(genType y_over_x)", false },
	{ "sinh", "genType sinh(genType x)", false },
	{ "cosh", "genType cosh(genType x)", false },
	{ "tanh", "genType tanh(genType x)", false },
	{ "asinh", "genType asinh(genType x)", false },
	{ "acosh", "genType acosh(genType x)", false },
	{ "atanh", "genType atanh(genType x)", false },

	// Exponential
	{ "pow", "genType pow(genType x, genType y)", false },
	{ "exp", "genType exp(genType x)", false },
	{ "log", "genType log(genType x)", false },
	{ "exp2", "genType exp2(genType x)", false },
	{ "log2", "genType log2(genType x)", false },
	{ "sqrt", "genType sqrt(genType x)", false },
	{ "inversesqrt", "genType inversesqrt(genType x)", false },

	// Common
	{ "abs", "genType abs(genType x)", false },
	{ "sign", "genType sign(genType x)", false },
	{ "floor", "genType floor(genType x)", false },
	{ "trunc", "genType trunc(genType x)", false },
	{ "round", "genType round(genType x)", false },
	{ "roundEven", "genType roundEven(genType x)", false },
	{ "ceil", "genType ceil(genType x)", false },
	{ "fract", "genType fract(genType x)", false },
	{ "mod", "genType mod(genType x, genType y)", false },
	{ "modf", "genType modf(genType x, out genType i)", false },
	{ "min", "genType min(genType x, genType y)", false },
	{ "max", "genType max(genType x, genType y)", false },
	{ "clamp", "genType clamp(genType x, genType minVal, genType maxVal)", false },
	{ "mix", "genType mix(genType x, genType y, genType a)", false },
	{ "step", "genType step(genType edge, genType x)", false },
	{ "smoothstep", "genType smoothstep(genType edge0, genType edge1, genType x)", false },
	{ "isnan", "genBType isnan(genType x)", false },
	{ "isinf", "genBType isinf(genType x)", false },
	{ "fma", "genType fma(genType a, genType b, genType c)", false },

	// Geometric
	{ "length", "float length(genType x)", false },
	{ "distance", "float distance(genType p0, genType p1)", false },
	{ "dot", "float dot(genType x, genType y)", false },
	{ "cross", "vec3 cross(vec3 x, vec3 y)", false },
	{ "normalize", "genType normalize(genType x)", false },
	{ "faceforward", "genType faceforward(genType N, genType I, genType Nref)", false },
	{ "reflect", "genType reflect(genType I, genType N)", false },
	{ "refract", "genType refract(genType I, genType N, float eta)", false },

	// Matrix
	{ "matrixCompMult", "mat matrixCompMult(mat x, mat y)", false },
	{ "outerProduct", "mat outerProduct(vec c, vec r)", false },
	{ "transpose", "mat transpose(mat m)", false },
	{ "determinant", "float determinant(mat m)", false },
	{ "inverse", "mat inverse(mat m)", false },

	// Vector Relational
	{ "lessThan", "bvec lessThan(vec x, vec y)", false },
	{ "lessThanEqual", "bvec lessThanEqual(vec x, vec y)", false },
	{ "greaterThan", "bvec greaterThan(vec x, vec y)", false },
	{ "greaterThanEqual", "bvec greaterThanEqual(vec x, vec y)", false },
	{ "equal", "bvec equal(vec x, vec y)", false },
	{ "notEqual", "bvec notEqual(vec x, vec y)", false },
	{ "any", "bool any(bvec x)", false },
	{ "all", "bool all(bvec x)", false },
	{ "not", "bvec not(bvec x)", false },

	// Bit / Packing
	{ "floatBitsToInt", "genIType floatBitsToInt(genType value)", false },
	{ "floatBitsToUint", "genUType floatBitsToUint(genType value)", false },
	{ "intBitsToFloat", "genType intBitsToFloat(genIType value)", false },
	{ "uintBitsToFloat", "genType uintBitsToFloat(genUType value)", false },
	{ "packHalf2x16", "uint packHalf2x16(vec2 v)", false },
	{ "unpackHalf2x16", "vec2 unpackHalf2x16(uint v)", false },
	{ "packUnorm4x8", "uint packUnorm4x8(vec4 v)", false },
	{ "unpackUnorm4x8", "vec4 unpackUnorm4x8(uint v)", false },
	{ "packSnorm4x8", "uint packSnorm4x8(vec4 v)", false },
	{ "unpackSnorm4x8", "vec4 unpackSnorm4x8(uint v)", false },
	{ "bitfieldExtract", "genIType bitfieldExtract(genIType value, int offset, int bits)", false },
	{ "bitfieldInsert", "genIType bitfieldInsert(genIType base, genIType insert, int offset, int bits)", false },
	{ "bitCount", "genIType bitCount(genIType value)", false },
	{ "findLSB", "genIType findLSB(genIType value)", false },
	{ "findMSB", "genIType findMSB(genIType value)", false },

	// Texture Sampling
	{ "texture", "gvec4 texture(gsampler sampler, vec P [, float bias])", false },
	{ "textureProj", "gvec4 textureProj(gsampler sampler, vec P [, float bias])", false },
	{ "textureLod", "gvec4 textureLod(gsampler sampler, vec P, float lod)", false },
	{ "textureOffset", "gvec4 textureOffset(gsampler sampler, vec P, ivec offset [, float bias])", false },
	{ "textureGrad", "gvec4 textureGrad(gsampler sampler, vec P, vec dPdx, vec dPdy)", false },
	{ "texelFetch", "gvec4 texelFetch(gsampler sampler, ivec P, int lod)", false },
	{ "texelFetchOffset", "gvec4 texelFetchOffset(gsampler sampler, ivec P, int lod, ivec offset)", false },
	{ "textureGather", "gvec4 textureGather(gsampler sampler, vec P [, int comp])", false },
	{ "textureSize", "ivec textureSize(gsampler sampler, int lod)", false },
	{ "textureQueryLevels", "int textureQueryLevels(gsampler sampler)", false },

	// Image Load/Store (Compute)
	{ "imageLoad", "gvec4 imageLoad(gimage image, ivec P)", false },
	{ "imageStore", "void imageStore(gimage image, ivec P, gvec4 data)", false },
	{ "imageSize", "ivec imageSize(gimage image)", false },

	// Atomics & Barriers
	{ "atomicAdd", "uint atomicAdd(inout uint mem, uint data)", false },
	{ "atomicMin", "uint atomicMin(inout uint mem, uint data)", false },
	{ "atomicMax", "uint atomicMax(inout uint mem, uint data)", false },
	{ "atomicAnd", "uint atomicAnd(inout uint mem, uint data)", false },
	{ "atomicOr", "uint atomicOr(inout uint mem, uint data)", false },
	{ "atomicXor", "uint atomicXor(inout uint mem, uint data)", false },
	{ "atomicExchange", "uint atomicExchange(inout uint mem, uint data)", false },
	{ "atomicCompSwap", "uint atomicCompSwap(inout uint mem, uint compare, uint data)", false },
	{ "barrier", "void barrier()", true },
	{ "memoryBarrier", "void memoryBarrier()", true },
	{ "memoryBarrierShared", "void memoryBarrierShared()", true },
	{ "memoryBarrierImage", "void memoryBarrierImage()", true },
	{ "memoryBarrierBuffer", "void memoryBarrierBuffer()", true },
	{ "memoryBarrierAtomicCounter", "void memoryBarrierAtomicCounter()", true },
	{ "groupMemoryBarrier", "void groupMemoryBarrier()", true },

	// Derivatives (Fragment)
	{ "dFdx", "genType dFdx(genType p)", false },
	{ "dFdy", "genType dFdy(genType p)", false },
	{ "dFdxFine", "genType dFdxFine(genType p)", false },
	{ "dFdyFine", "genType dFdyFine(genType p)", false },
	{ "fwidth", "genType fwidth(genType p)", false },
	{ "fwidthFine", "genType fwidthFine(genType p)", false },

	{ nullptr, nullptr, false }
};

static bool _glsl_is_ident_start(CharType c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool _glsl_is_ident_char(CharType c) {
	return _glsl_is_ident_start(c) || (c >= '0' && c <= '9');
}

static String _glsl_strip_comments_and_strings(const String &p_code) {
	String out;
	int len = p_code.length();
	bool in_line_comment = false;
	bool in_block_comment = false;
	bool in_string = false;

	for (int i = 0; i < len; i++) {
		CharType c = p_code[i];
		CharType next = (i + 1 < len) ? p_code[i + 1] : 0;

		if (in_line_comment) {
			if (c == '\n') {
				in_line_comment = false;
				out += '\n';
			} else {
				out += ' ';
			}
		} else if (in_block_comment) {
			if (c == '*' && next == '/') {
				in_block_comment = false;
				out += "  ";
				i++;
			} else if (c == '\n') {
				out += '\n';
			} else {
				out += ' ';
			}
		} else if (in_string) {
			if (c == '\\' && i + 1 < len) {
				out += "  ";
				i++;
			} else if (c == '"') {
				in_string = false;
				out += ' ';
			} else if (c == '\n') {
				in_string = false;
				out += '\n';
			} else {
				out += ' ';
			}
		} else {
			if (c == '/' && next == '/') {
				in_line_comment = true;
				out += "  ";
				i++;
			} else if (c == '/' && next == '*') {
				in_block_comment = true;
				out += "  ";
				i++;
			} else if (c == '"') {
				in_string = true;
				out += ' ';
			} else {
				out += c;
			}
		}
	}
	return out;
}

static bool _glsl_is_keyword_token(const String &p_tok) {
	static const char *kw[] = {
		"if", "else", "for", "while", "do", "switch", "case", "default",
		"break", "continue", "return", "discard", "struct", "uniform",
		"buffer", "shared", "in", "out", "inout", "const", "layout",
		"flat", "smooth", "noperspective", "centroid", "highp", "mediump",
		"lowp", "precision", "readonly", "writeonly", "coherent",
		"volatile", "restrict", "true", "false", nullptr
	};
	for (int i = 0; kw[i]; i++) {
		if (p_tok == kw[i]) {
			return true;
		}
	}
	return false;
}

struct GLSLParsedInfo {
	Map<String, Vector<Pair<String, String>>> structs; // struct_name -> [(type, member_name)]
	Map<String, String> var_types; // var_name -> type_name
	Map<String, String> func_sigs; // func_name -> signature
	Map<String, bool> func_zero_args;
	Vector<String> constants;
	Vector<String> globals;
	Vector<String> locals;
};

static void _glsl_parse_symbols(const String &p_clean_code, int p_cursor_idx, GLSLParsedInfo &r_info) {
	int len = p_clean_code.length();
	int i = 0;

	// 1. Parse #define macros
	Vector<String> raw_lines = p_clean_code.split("\n");
	for (int li = 0; li < raw_lines.size(); li++) {
		String t = raw_lines[li].strip_edges();
		if (t.begins_with("#define")) {
			String rest = t.substr(7, t.length() - 7).strip_edges();
			int id_end = 0;
			while (id_end < rest.length() && _glsl_is_ident_char(rest[id_end])) {
				id_end++;
			}
			if (id_end > 0) {
				String macro_name = rest.substr(0, id_end);
				if (id_end < rest.length() && rest[id_end] == '(') {
					int cp = rest.find(")", id_end);
					String args = (cp != -1) ? rest.substr(id_end, cp - id_end + 1) : "()";
					r_info.func_sigs[macro_name] = "#define " + macro_name + args;
					r_info.func_zero_args[macro_name] = (args == "()");
				} else {
					r_info.constants.push_back(macro_name);
				}
			}
		}
	}

	auto skip_ws = [&](int &pos) {
		while (pos < len && (p_clean_code[pos] == ' ' || p_clean_code[pos] == '\t' || p_clean_code[pos] == '\r' || p_clean_code[pos] == '\n')) {
			pos++;
		}
	};

	auto read_ident = [&](int &pos) -> String {
		int start = pos;
		if (pos < len && _glsl_is_ident_start(p_clean_code[pos])) {
			pos++;
			while (pos < len && _glsl_is_ident_char(p_clean_code[pos])) {
				pos++;
			}
		}
		return p_clean_code.substr(start, pos - start);
	};

	auto skip_paren = [&](int &pos) {
		if (pos >= len || p_clean_code[pos] != '(') {
			return;
		}
		int depth = 1;
		pos++;
		while (pos < len && depth > 0) {
			if (p_clean_code[pos] == '(') {
				depth++;
			} else if (p_clean_code[pos] == ')') {
				depth--;
			}
			pos++;
		}
	};

	auto skip_brackets = [&](int &pos) {
		while (true) {
			skip_ws(pos);
			if (pos >= len || p_clean_code[pos] != '[') {
				break;
			}
			int depth = 1;
			pos++;
			while (pos < len && depth > 0) {
				if (p_clean_code[pos] == '[') {
					depth++;
				} else if (p_clean_code[pos] == ']') {
					depth--;
				}
				pos++;
			}
		}
	};

	auto is_qualifier = [](const String &tok) -> bool {
		static const char *quals[] = {
			"layout", "uniform", "buffer", "shared", "in", "out", "inout",
			"const", "readonly", "writeonly", "coherent", "volatile", "restrict",
			"flat", "smooth", "noperspective", "centroid", "highp", "mediump", "lowp", nullptr
		};
		for (int q = 0; quals[q]; q++) {
			if (tok == quals[q]) {
				return true;
			}
		}
		return false;
	};

	int brace_depth = 0;
	int cur_func_brace_depth = -1;
	Vector<Pair<int, String>> scoped_locals; // (depth, name)

	while (i < len) {
		skip_ws(i);
		if (i >= len) {
			break;
		}

		CharType ch = p_clean_code[i];
		if (ch == '#' || ch == '[') {
			while (i < len && p_clean_code[i] != '\n') {
				i++;
			}
			continue;
		}
		if (ch == '{') {
			brace_depth++;
			i++;
			continue;
		}
		if (ch == '}') {
			brace_depth = MAX(0, brace_depth - 1);
			if (i < p_cursor_idx) {
				for (int s = scoped_locals.size() - 1; s >= 0; s--) {
					if (scoped_locals[s].first > brace_depth) {
						scoped_locals.remove(s);
					}
				}
			}
			if (brace_depth == 0) {
				cur_func_brace_depth = -1;
			}
			i++;
			continue;
		}

		if (!_glsl_is_ident_start(ch)) {
			i++;
			continue;
		}

		// Read declaration / statement prefix tokens on top-level or inside current function before cursor
		int stmt_start = i;
		Vector<String> tokens;
		bool saw_const = false;
		bool saw_uniform = false;

		while (i < len) {
			skip_ws(i);
			if (i >= len) {
				break;
			}
			CharType c = p_clean_code[i];
			if (c == '(' && tokens.size() == 1 && tokens[0] == "layout") {
				skip_paren(i);
				continue;
			}
			if (_glsl_is_ident_start(c)) {
				String tok = read_ident(i);
				if (tok == "const") {
					saw_const = true;
				}
				if (tok == "uniform") {
					saw_uniform = true;
				}
				if (is_qualifier(tok)) {
					continue;
				}
				tokens.push_back(tok);
				if (tokens.size() >= 2) {
					break;
				}
			} else {
				break;
			}
		}

		if (tokens.size() == 2 && tokens[0] == "struct") {
			String sname = tokens[1];
			skip_ws(i);
			if (i < len && p_clean_code[i] == '{') {
				i++;
				Vector<Pair<String, String>> members;
				while (i < len && p_clean_code[i] != '}') {
					skip_ws(i);
					if (i >= len || p_clean_code[i] == '}') {
						break;
					}
					String mtype;
					while (i < len && _glsl_is_ident_start(p_clean_code[i])) {
						String mt = read_ident(i);
						if (!is_qualifier(mt)) {
							mtype = mt;
							break;
						}
						skip_ws(i);
					}
					if (!mtype.empty()) {
						while (i < len && p_clean_code[i] != ';' && p_clean_code[i] != '}') {
							skip_ws(i);
							if (i < len && _glsl_is_ident_start(p_clean_code[i])) {
								String mname = read_ident(i);
								members.push_back(Pair<String, String>(mtype, mname));
							}
							skip_brackets(i);
							while (i < len && p_clean_code[i] != ',' && p_clean_code[i] != ';' && p_clean_code[i] != '}') {
								i++;
							}
							if (i < len && p_clean_code[i] == ',') {
								i++;
							}
						}
					} else {
						while (i < len && p_clean_code[i] != ';' && p_clean_code[i] != '}') {
							i++;
						}
					}
					if (i < len && p_clean_code[i] == ';') {
						i++;
					}
				}
				if (i < len && p_clean_code[i] == '}') {
					i++;
				}
				r_info.structs[sname] = members;
			}
			continue;
		}

		// Handle buffer block: buffer Name { Type member; ... } instance_name;
		if (tokens.size() == 1 && (p_clean_code.substr(stmt_start, i - stmt_start).find("buffer") != -1)) {
			skip_ws(i);
			if (i < len && p_clean_code[i] == '{') {
				i++;
				while (i < len && p_clean_code[i] != '}') {
					skip_ws(i);
					if (i >= len || p_clean_code[i] == '}') {
						break;
					}
					String mtype;
					while (i < len && _glsl_is_ident_start(p_clean_code[i])) {
						String mt = read_ident(i);
						if (!is_qualifier(mt)) {
							mtype = mt;
							break;
						}
						skip_ws(i);
					}
					if (!mtype.empty()) {
						while (i < len && p_clean_code[i] != ';' && p_clean_code[i] != '}') {
							skip_ws(i);
							if (i < len && _glsl_is_ident_start(p_clean_code[i])) {
								String mname = read_ident(i);
								r_info.var_types[mname] = mtype;
								r_info.globals.push_back(mname);
							}
							skip_brackets(i);
							while (i < len && p_clean_code[i] != ',' && p_clean_code[i] != ';' && p_clean_code[i] != '}') {
								i++;
							}
							if (i < len && p_clean_code[i] == ',') {
								i++;
							}
						}
					} else {
						while (i < len && p_clean_code[i] != ';' && p_clean_code[i] != '}') {
							i++;
						}
					}
					if (i < len && p_clean_code[i] == ';') {
						i++;
					}
				}
				if (i < len && p_clean_code[i] == '}') {
					i++;
				}
				continue;
			}
		}

		if (tokens.size() == 2 && !_glsl_is_keyword_token(tokens[0]) && !_glsl_is_keyword_token(tokens[1])) {
			String type_name = tokens[0];
			String decl_name = tokens[1];
			skip_ws(i);

			if (brace_depth == 0 && i < len && p_clean_code[i] == '(') {
				// Function declaration / definition
				int pstart = i;
				skip_paren(i);
				String raw_params = p_clean_code.substr(pstart, i - pstart).strip_edges();
				String compact_params = raw_params.replace("\n", " ").replace("\t", " ");
				while (compact_params.find("  ") != -1) {
					compact_params = compact_params.replace("  ", " ");
				}
				r_info.func_sigs[decl_name] = type_name + " " + decl_name + compact_params;
				r_info.func_zero_args[decl_name] = (compact_params == "()" || compact_params == "(void)" || compact_params == "( void )");

				skip_ws(i);
				if (i < len && p_clean_code[i] == '{') {
					// Record function parameters if cursor is inside this function body
					int body_start = i;
					int bd = 1;
					int scan = i + 1;
					while (scan < len && bd > 0) {
						if (p_clean_code[scan] == '{') {
							bd++;
						} else if (p_clean_code[scan] == '}') {
							bd--;
						}
						scan++;
					}
					if (p_cursor_idx > body_start && p_cursor_idx <= scan) {
						cur_func_brace_depth = 1;
						String inner = compact_params.substr(1, MAX(0, compact_params.length() - 2));
						Vector<String> plist = inner.split(",");
						for (int p = 0; p < plist.size(); p++) {
							Vector<String> ptoks = plist[p].strip_edges().split(" ", false);
							if (ptoks.size() >= 2) {
								String pname = ptoks[ptoks.size() - 1];
								int bpos = pname.find("[");
								if (bpos != -1) {
									pname = pname.substr(0, bpos);
								}
								String ptype = ptoks[ptoks.size() - 2];
								if (!pname.empty() && _glsl_is_ident_start(pname[0])) {
									r_info.var_types[pname] = ptype;
									scoped_locals.push_back(Pair<int, String>(1, pname));
								}
							}
						}
					}
					// Step into function body if cursor is inside, otherwise skip it
					if (p_cursor_idx <= body_start || p_cursor_idx > scan) {
						i = scan;
					} else {
						brace_depth = 1;
						i = body_start + 1;
					}
				}
				continue;
			} else if (brace_depth == 0 || i < p_cursor_idx) {
				// Variable / uniform / constant declaration
				auto add_var = [&](const String &vname) {
					r_info.var_types[vname] = type_name;
					if (brace_depth == 0) {
						if (saw_const && !saw_uniform) {
							r_info.constants.push_back(vname);
						} else {
							r_info.globals.push_back(vname);
						}
					} else if (cur_func_brace_depth >= 0) {
						scoped_locals.push_back(Pair<int, String>(brace_depth, vname));
					}
				};

				add_var(decl_name);
				skip_brackets(i);
				while (i < len && p_clean_code[i] != ';' && p_clean_code[i] != '{' && p_clean_code[i] != '}') {
					if (p_clean_code[i] == '(') {
						skip_paren(i);
						continue;
					}
					if (p_clean_code[i] == ',') {
						i++;
						skip_ws(i);
						if (i < len && _glsl_is_ident_start(p_clean_code[i])) {
							String extra_name = read_ident(i);
							add_var(extra_name);
							skip_brackets(i);
							continue;
						}
					}
					i++;
				}
				if (i < len && p_clean_code[i] == ';') {
					i++;
				}
				continue;
			}
		}

		// Advance past remaining non-brace characters until next statement boundary
		while (i < len) {
			CharType c = p_clean_code[i];
			if (c == ';' || c == '{' || c == '}' || c == '\n') {
				if (c == ';') {
					i++;
				}
				break;
			}
			i++;
		}
	}

	for (int s = 0; s < scoped_locals.size(); s++) {
		r_info.locals.push_back(scoped_locals[s].second);
	}
}

static String _glsl_find_calltip(const String &p_clean_code, int p_cursor_idx, const GLSLParsedInfo &p_info) {
	int depth = 0;
	int open_paren = -1;
	int arg_index = 0;
	for (int i = MIN(p_cursor_idx - 1, p_clean_code.length() - 1); i >= 0; i--) {
		CharType c = p_clean_code[i];
		if (c == ';') {
			break;
		}
		if (c == ')') {
			depth++;
		} else if (c == '(') {
			if (depth == 0) {
				open_paren = i;
				break;
			}
			depth--;
		} else if (c == ',' && depth == 0) {
			arg_index++;
		}
	}

	if (open_paren <= 0) {
		return String();
	}

	int end = open_paren - 1;
	while (end >= 0 && (p_clean_code[end] == ' ' || p_clean_code[end] == '\t')) {
		end--;
	}
	if (end < 0 || !_glsl_is_ident_char(p_clean_code[end])) {
		return String();
	}
	int start = end;
	while (start > 0 && _glsl_is_ident_char(p_clean_code[start - 1])) {
		start--;
	}
	if (!_glsl_is_ident_start(p_clean_code[start])) {
		return String();
	}
	String fn_name = p_clean_code.substr(start, end - start + 1);
	if (_glsl_is_keyword_token(fn_name)) {
		return String();
	}

	if (p_info.func_sigs.has(fn_name)) {
		return p_info.func_sigs[fn_name];
	}

	for (int b = 0; GLSL_BUILTIN_FUNCTIONS[b].name; b++) {
		if (fn_name == GLSL_BUILTIN_FUNCTIONS[b].name) {
			return String(GLSL_BUILTIN_FUNCTIONS[b].signature);
		}
	}

	return String();
}

} // namespace

/*** SHADER SCRIPT EDITOR ****/

Ref<Shader> ShaderTextEditor::get_edited_shader() const {
	return shader;
}
void ShaderTextEditor::set_edited_shader(const Ref<Shader> &p_shader) {
	if (shader == p_shader) {
		return;
	}
	shader = p_shader;
	glsl_shader.unref();

	_load_theme_settings();

	get_text_edit()->set_text(p_shader->get_code());
	get_text_edit()->clear_undo_history();
	get_text_edit()->call_deferred("set_h_scroll", 0);
	get_text_edit()->call_deferred("set_v_scroll", 0);

	_validate_script();
	_line_col_changed();
}

Ref<GLSLShader> ShaderTextEditor::get_edited_glsl_shader() const {
	return glsl_shader;
}

void ShaderTextEditor::set_edited_glsl_shader(const Ref<GLSLShader> &p_shader) {
	if (glsl_shader == p_shader) {
		return;
	}
	glsl_shader = p_shader;
	shader.unref();

	_load_theme_settings();

	get_text_edit()->set_text(p_shader->get_code());
	get_text_edit()->clear_undo_history();
	get_text_edit()->call_deferred("set_h_scroll", 0);
	get_text_edit()->call_deferred("set_v_scroll", 0);

	_validate_script();
	_line_col_changed();
}

void ShaderTextEditor::reload_text() {
	ERR_FAIL_COND(shader.is_null() && glsl_shader.is_null());

	TextEdit *te = get_text_edit();
	int column = te->cursor_get_column();
	int row = te->cursor_get_line();
	int h = te->get_h_scroll();
	int v = te->get_v_scroll();

	te->set_text(glsl_shader.is_valid() ? glsl_shader->get_code() : shader->get_code());
	te->cursor_set_line(row);
	te->cursor_set_column(column);
	te->set_h_scroll(h);
	te->set_v_scroll(v);

	te->tag_saved_version();

	update_line_and_column();
}

void ShaderTextEditor::_load_theme_settings() {
	get_text_edit()->clear_colors();

	Color background_color = EDITOR_GET("text_editor/highlighting/background_color");
	Color completion_background_color = EDITOR_GET("text_editor/highlighting/completion_background_color");
	Color completion_selected_color = EDITOR_GET("text_editor/highlighting/completion_selected_color");
	Color completion_existing_color = EDITOR_GET("text_editor/highlighting/completion_existing_color");
	Color completion_scroll_color = EDITOR_GET("text_editor/highlighting/completion_scroll_color");
	Color completion_font_color = EDITOR_GET("text_editor/highlighting/completion_font_color");
	Color text_color = EDITOR_GET("text_editor/highlighting/text_color");
	Color line_number_color = EDITOR_GET("text_editor/highlighting/line_number_color");
	Color caret_color = EDITOR_GET("text_editor/highlighting/caret_color");
	Color caret_background_color = EDITOR_GET("text_editor/highlighting/caret_background_color");
	Color text_selected_color = EDITOR_GET("text_editor/highlighting/text_selected_color");
	Color selection_color = EDITOR_GET("text_editor/highlighting/selection_color");
	Color brace_mismatch_color = EDITOR_GET("text_editor/highlighting/brace_mismatch_color");
	Color current_line_color = EDITOR_GET("text_editor/highlighting/current_line_color");
	Color line_length_guideline_color = EDITOR_GET("text_editor/highlighting/line_length_guideline_color");
	Color word_highlighted_color = EDITOR_GET("text_editor/highlighting/word_highlighted_color");
	Color number_color = EDITOR_GET("text_editor/highlighting/number_color");
	Color function_color = EDITOR_GET("text_editor/highlighting/function_color");
	Color member_variable_color = EDITOR_GET("text_editor/highlighting/member_variable_color");
	Color mark_color = EDITOR_GET("text_editor/highlighting/mark_color");
	Color bookmark_color = EDITOR_GET("text_editor/highlighting/bookmark_color");
	Color breakpoint_color = EDITOR_GET("text_editor/highlighting/breakpoint_color");
	Color executing_line_color = EDITOR_GET("text_editor/highlighting/executing_line_color");
	Color code_folding_color = EDITOR_GET("text_editor/highlighting/code_folding_color");
	Color search_result_color = EDITOR_GET("text_editor/highlighting/search_result_color");
	Color search_result_border_color = EDITOR_GET("text_editor/highlighting/search_result_border_color");
	Color symbol_color = EDITOR_GET("text_editor/highlighting/symbol_color");
	Color keyword_color = EDITOR_GET("text_editor/highlighting/keyword_color");
	Color control_flow_keyword_color = EDITOR_GET("text_editor/highlighting/control_flow_keyword_color");
	Color comment_color = EDITOR_GET("text_editor/highlighting/comment_color");

	get_text_edit()->add_color_override("background_color", background_color);
	get_text_edit()->add_color_override("completion_background_color", completion_background_color);
	get_text_edit()->add_color_override("completion_selected_color", completion_selected_color);
	get_text_edit()->add_color_override("completion_existing_color", completion_existing_color);
	get_text_edit()->add_color_override("completion_scroll_color", completion_scroll_color);
	get_text_edit()->add_color_override("completion_font_color", completion_font_color);
	get_text_edit()->add_color_override("font_color", text_color);
	get_text_edit()->add_color_override("line_number_color", line_number_color);
	get_text_edit()->add_color_override("caret_color", caret_color);
	get_text_edit()->add_color_override("caret_background_color", caret_background_color);
	get_text_edit()->add_color_override("font_color_selected", text_selected_color);
	get_text_edit()->add_color_override("selection_color", selection_color);
	get_text_edit()->add_color_override("brace_mismatch_color", brace_mismatch_color);
	get_text_edit()->add_color_override("current_line_color", current_line_color);
	get_text_edit()->add_color_override("line_length_guideline_color", line_length_guideline_color);
	get_text_edit()->add_color_override("word_highlighted_color", word_highlighted_color);
	get_text_edit()->add_color_override("number_color", number_color);
	get_text_edit()->add_color_override("function_color", function_color);
	get_text_edit()->add_color_override("member_variable_color", member_variable_color);
	get_text_edit()->add_color_override("mark_color", mark_color);
	get_text_edit()->add_color_override("bookmark_color", bookmark_color);
	get_text_edit()->add_color_override("breakpoint_color", breakpoint_color);
	get_text_edit()->add_color_override("executing_line_color", executing_line_color);
	get_text_edit()->add_color_override("code_folding_color", code_folding_color);
	get_text_edit()->add_color_override("search_result_color", search_result_color);
	get_text_edit()->add_color_override("search_result_border_color", search_result_border_color);
	get_text_edit()->add_color_override("symbol_color", symbol_color);

	const Color user_type_color = EDITOR_GET("text_editor/highlighting/user_type_color");

	if (glsl_shader.is_valid()) {
		static const char *glsl_control_keywords[] = {
			"if", "else", "for", "while", "do", "switch", "case", "default",
			"break", "continue", "return", "discard", nullptr
		};
		static const char *glsl_keywords[] = {
			"void", "bool", "int", "uint", "float", "double",
			"vec2", "vec3", "vec4", "bvec2", "bvec3", "bvec4",
			"ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4",
			"dvec2", "dvec3", "dvec4", "mat2", "mat3", "mat4",
			"mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4",
			"mat4x2", "mat4x3", "mat4x4",
			"sampler2D", "sampler3D", "samplerCube", "sampler2DArray", "sampler2DShadow",
			"sampler2DArrayShadow", "samplerCubeShadow",
			"isampler2D", "usampler2D", "isampler3D", "usampler3D",
			"isampler2DArray", "usampler2DArray",
			"image2D", "image3D", "image2DArray", "iimage2D", "uimage2D",
			"iimage3D", "uimage3D", "iimage2DArray", "uimage2DArray",
			"struct", "uniform", "buffer", "shared", "in", "out", "inout",
			"const", "layout", "location", "binding", "std430", "std140",
			"coherent", "volatile", "restrict", "readonly", "writeonly",
			"flat", "smooth", "noperspective", "centroid",
			"highp", "mediump", "lowp", "precision",
			"local_size_x", "local_size_y", "local_size_z",
			"rgba16f", "rgba32f", "rgba8", "rg16f", "rg32f", "rg8",
			"r16f", "r32f", "r8", "r32i", "r32ui", "rgba16ui", "rgba32ui",
			"true", "false", nullptr
		};
		static const char *glsl_builtins[] = {
			"gl_GlobalInvocationID", "gl_LocalInvocationID", "gl_WorkGroupID",
			"gl_WorkGroupSize", "gl_NumWorkGroups", "gl_LocalInvocationIndex",
			"gl_Position", "gl_FragCoord", "gl_FragDepth", "gl_FrontFacing",
			"gl_PointCoord", "gl_PointSize", "gl_VertexID", "gl_InstanceID",
			"u_projection_matrix", "u_inv_projection_matrix", "u_view_matrix",
			"u_camera_matrix", "u_model_matrix", "u_mvp_matrix",
			"u_normal_matrix", "u_camera_position", "u_time", "u_viewport_size",
			nullptr
		};

		for (int i = 0; glsl_control_keywords[i]; i++) {
			get_text_edit()->add_keyword_color(glsl_control_keywords[i], control_flow_keyword_color);
		}
		for (int i = 0; glsl_keywords[i]; i++) {
			get_text_edit()->add_keyword_color(glsl_keywords[i], keyword_color);
		}
		for (int i = 0; glsl_builtins[i]; i++) {
			get_text_edit()->add_keyword_color(glsl_builtins[i], user_type_color);
		}
		for (int i = 0; GLSL_BUILTIN_FUNCTIONS[i].name; i++) {
			get_text_edit()->add_keyword_color(GLSL_BUILTIN_FUNCTIONS[i].name, function_color);
		}

		get_text_edit()->add_color_region("#", "", member_variable_color, true);
	} else {
		List<String> keywords;
		ShaderLanguage::get_keyword_list(&keywords);

		for (List<String>::Element *E = keywords.front(); E; E = E->next()) {
			if (ShaderLanguage::is_control_flow_keyword(E->get())) {
				get_text_edit()->add_keyword_color(E->get(), control_flow_keyword_color);
			} else {
				get_text_edit()->add_keyword_color(E->get(), keyword_color);
			}
		}

		// Colorize built-ins like `COLOR` differently to make them easier
		// to distinguish from keywords at a quick glance.

		List<String> built_ins;
		if (shader.is_valid()) {
			for (const Map<StringName, ShaderLanguage::FunctionInfo>::Element *E = ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(shader->get_mode())).front(); E; E = E->next()) {
				for (const Map<StringName, ShaderLanguage::BuiltInInfo>::Element *F = E->get().built_ins.front(); F; F = F->next()) {
					built_ins.push_back(F->key());
				}
			}

			for (int i = 0; i < ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode())).size(); i++) {
				built_ins.push_back(ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode()))[i]);
			}
		}

		for (List<String>::Element *E = built_ins.front(); E; E = E->next()) {
			get_text_edit()->add_keyword_color(E->get(), user_type_color);
		}
	}

	// Colorize comments.
	get_text_edit()->add_color_region("/*", "*/", comment_color, false);
	get_text_edit()->add_color_region("//", "", comment_color, false);
}

void ShaderTextEditor::_check_shader_mode() {
	if (glsl_shader.is_valid() || shader.is_null()) {
		return;
	}

	String type = ShaderLanguage::get_shader_type(get_text_edit()->get_text());

	Shader::Mode mode;

	if (type == "canvas_item") {
		mode = Shader::MODE_CANVAS_ITEM;
	} else if (type == "particles") {
		mode = Shader::MODE_PARTICLES;
	} else {
		mode = Shader::MODE_SPATIAL;
	}

	if (shader->get_mode() != mode) {
		shader->set_code(get_text_edit()->get_text());
		_load_theme_settings();
	}
}

void ShaderTextEditor::_code_complete_script(const String &p_code, List<ScriptCodeCompletionOption> *r_options) {
	if (glsl_shader.is_valid()) {
		String cursor_marker = String::chr(0xFFFF);
		int cursor_idx = p_code.find(cursor_marker);
		String clean_raw = p_code.replace(cursor_marker, "");

		if (cursor_idx == -1) {
			TextEdit *te = get_text_edit();
			int target_line = te->cursor_get_line();
			int target_col = te->cursor_get_column();
			int cur_line = 0;
			int idx = 0;
			while (idx < clean_raw.length() && cur_line < target_line) {
				if (clean_raw[idx] == '\n') {
					cur_line++;
				}
				idx++;
			}
			cursor_idx = MIN(idx + target_col, clean_raw.length());
		}

		String stripped = _glsl_strip_comments_and_strings(clean_raw);
		GLSLParsedInfo info;
		_glsl_parse_symbols(stripped, cursor_idx, info);

		String calltip = _glsl_find_calltip(stripped, cursor_idx, info);
		get_text_edit()->set_code_hint(calltip);

		Map<String, ScriptCodeCompletionOption> opts;
		auto add_opt = [&](const String &p_display, const String &p_insert, ScriptCodeCompletionOption::Kind p_kind) {
			if (!opts.has(p_display)) {
				ScriptCodeCompletionOption opt;
				opt.display = p_display;
				opt.insert_text = p_insert;
				opt.kind = p_kind;
				opts[p_display] = opt;
			}
		};

		// Extract partial identifier at cursor
		int ident_start = cursor_idx;
		while (ident_start > 0 && _glsl_is_ident_char(clean_raw[ident_start - 1])) {
			ident_start--;
		}

		int line_start = ident_start;
		while (line_start > 0 && clean_raw[line_start - 1] != '\n') {
			line_start--;
		}
		String line_prefix = clean_raw.substr(line_start, ident_start - line_start);
		String line_prefix_trimmed = line_prefix.strip_edges();

		// 1. Preprocessor completion after '#'
		if (line_prefix_trimmed.begins_with("#") && line_prefix_trimmed.length() == 1) {
			static const char *pp_cmds[] = {
				"version 430 core", "version 330 core", "define ", "undef ",
				"ifdef ", "ifndef ", "if ", "elif ", "else", "endif",
				"extension ", "pragma ", "line ", nullptr
			};
			for (int i = 0; pp_cmds[i]; i++) {
				String s(pp_cmds[i]);
				add_opt("#" + s.strip_edges(), s, ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			}
			for (Map<String, ScriptCodeCompletionOption>::Element *E = opts.front(); E; E = E->next()) {
				r_options->push_back(E->get());
			}
			return;
		}

		// 2. // render_mode completion
		if (line_prefix_trimmed.begins_with("//") && line_prefix_trimmed.find("render_mode") != -1) {
			static const char *render_modes[] = {
				"blend_mix", "blend_add", "blend_sub", "blend_mul",
				"cull_back", "cull_front", "cull_disabled",
				"depth_draw_opaque", "depth_draw_always", "depth_draw_never",
				"depth_test_disable", "transparent", nullptr
			};
			for (int i = 0; render_modes[i]; i++) {
				add_opt(render_modes[i], render_modes[i], ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			}
			for (Map<String, ScriptCodeCompletionOption>::Element *E = opts.front(); E; E = E->next()) {
				r_options->push_back(E->get());
			}
			return;
		}

		// 3. Dot access: struct members or vector swizzles
		int before_ws = ident_start - 1;
		while (before_ws >= 0 && (clean_raw[before_ws] == ' ' || clean_raw[before_ws] == '\t')) {
			before_ws--;
		}
		if (before_ws >= 0 && clean_raw[before_ws] == '.') {
			int base_end = before_ws - 1;
			while (base_end >= 0 && (clean_raw[base_end] == ' ' || clean_raw[base_end] == '\t')) {
				base_end--;
			}
			// Skip trailing array indexing e.g. arr[i].
			if (base_end >= 0 && clean_raw[base_end] == ']') {
				int bdepth = 1;
				base_end--;
				while (base_end >= 0 && bdepth > 0) {
					if (clean_raw[base_end] == ']') {
						bdepth++;
					} else if (clean_raw[base_end] == '[') {
						bdepth--;
					}
					base_end--;
				}
				while (base_end >= 0 && (clean_raw[base_end] == ' ' || clean_raw[base_end] == '\t')) {
					base_end--;
				}
			}

			String base_ident;
			if (base_end >= 0 && _glsl_is_ident_char(clean_raw[base_end])) {
				int base_start = base_end;
				while (base_start > 0 && _glsl_is_ident_char(clean_raw[base_start - 1])) {
					base_start--;
				}
				base_ident = clean_raw.substr(base_start, base_end - base_start + 1);
			}

			String base_type;
			if (!base_ident.empty() && info.var_types.has(base_ident)) {
				base_type = info.var_types[base_ident];
			} else if (base_ident == "gl_GlobalInvocationID" || base_ident == "gl_LocalInvocationID" ||
					   base_ident == "gl_WorkGroupID" || base_ident == "gl_WorkGroupSize" || base_ident == "gl_NumWorkGroups") {
				base_type = "uvec3";
			} else if (base_ident == "gl_Position" || base_ident == "gl_FragCoord") {
				base_type = "vec4";
			} else if (base_ident == "gl_PointCoord") {
				base_type = "vec2";
			} else if (base_ident == "u_camera_position") {
				base_type = "vec3";
			} else if (base_ident == "u_viewport_size") {
				base_type = "vec2";
			}

			if (!base_type.empty() && info.structs.has(base_type)) {
				const Vector<Pair<String, String>> &members = info.structs[base_type];
				for (int m = 0; m < members.size(); m++) {
					add_opt(members[m].second, members[m].second, ScriptCodeCompletionOption::KIND_MEMBER);
				}
			} else {
				int vec_dim = 4;
				if (base_type.ends_with("vec2")) {
					vec_dim = 2;
				} else if (base_type.ends_with("vec3")) {
					vec_dim = 3;
				}
				static const char *swiz2[] = { "x", "y", "xy", "yx", "xx", "yy", "r", "g", "rg", "gr", "s", "t", "st", nullptr };
				static const char *swiz3[] = { "z", "xz", "yz", "xyz", "zyx", "b", "rb", "gb", "rgb", "bgr", "p", "stp", nullptr };
				static const char *swiz4[] = { "w", "xw", "yw", "zw", "xyzw", "a", "ra", "ga", "ba", "rgba", "q", "stpq", nullptr };

				for (int i = 0; swiz2[i]; i++) {
					add_opt(swiz2[i], swiz2[i], ScriptCodeCompletionOption::KIND_MEMBER);
				}
				if (vec_dim >= 3) {
					for (int i = 0; swiz3[i]; i++) {
						add_opt(swiz3[i], swiz3[i], ScriptCodeCompletionOption::KIND_MEMBER);
					}
				}
				if (vec_dim >= 4) {
					for (int i = 0; swiz4[i]; i++) {
						add_opt(swiz4[i], swiz4[i], ScriptCodeCompletionOption::KIND_MEMBER);
					}
				}
			}

			for (Map<String, ScriptCodeCompletionOption>::Element *E = opts.front(); E; E = E->next()) {
				r_options->push_back(E->get());
			}
			return;
		}

		// 4. Inside layout(...)
		int pdepth = 0;
		for (int k = ident_start - 1; k >= line_start; k--) {
			if (clean_raw[k] == ')') {
				pdepth++;
			} else if (clean_raw[k] == '(') {
				if (pdepth == 0) {
					int pre = k - 1;
					while (pre >= line_start && (clean_raw[pre] == ' ' || clean_raw[pre] == '\t')) {
						pre--;
					}
					if (pre >= 5 && clean_raw.substr(pre - 5, 6) == "layout") {
						static const char *layout_opts[] = {
							"location = ", "binding = ", "std430", "std140",
							"local_size_x = ", "local_size_y = ", "local_size_z = ",
							"rgba16f", "rgba32f", "rgba8", "rg16f", "rg32f", "rg8",
							"r16f", "r32f", "r8", "r32i", "r32ui", "rgba16ui", "rgba32ui", nullptr
						};
						for (int l = 0; layout_opts[l]; l++) {
							String s(layout_opts[l]);
							add_opt(s.strip_edges(), s, ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
						}
						for (Map<String, ScriptCodeCompletionOption>::Element *E = opts.front(); E; E = E->next()) {
							r_options->push_back(E->get());
						}
						return;
					}
					break;
				}
				pdepth--;
			}
		}

		// 5. General GLSL completion: user symbols + built-in types/keywords/functions
		for (Map<String, Vector<Pair<String, String>>>::Element *E = info.structs.front(); E; E = E->next()) {
			add_opt(E->key(), E->key(), ScriptCodeCompletionOption::KIND_CLASS);
		}
		for (int i = 0; i < info.constants.size(); i++) {
			add_opt(info.constants[i], info.constants[i], ScriptCodeCompletionOption::KIND_CONSTANT);
		}
		for (int i = 0; i < info.globals.size(); i++) {
			add_opt(info.globals[i], info.globals[i], ScriptCodeCompletionOption::KIND_VARIABLE);
		}
		for (int i = 0; i < info.locals.size(); i++) {
			add_opt(info.locals[i], info.locals[i], ScriptCodeCompletionOption::KIND_VARIABLE);
		}
		for (Map<String, String>::Element *E = info.func_sigs.front(); E; E = E->next()) {
			bool zargs = info.func_zero_args.has(E->key()) && info.func_zero_args[E->key()];
			add_opt(E->key(), E->key() + (zargs ? "()" : "("), ScriptCodeCompletionOption::KIND_FUNCTION);
		}

		static const char *glsl_types_and_kw[] = {
			"void", "bool", "int", "uint", "float", "double",
			"vec2", "vec3", "vec4", "bvec2", "bvec3", "bvec4",
			"ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4",
			"dvec2", "dvec3", "dvec4", "mat2", "mat3", "mat4",
			"mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3", "mat3x4",
			"mat4x2", "mat4x3", "mat4x4",
			"sampler2D", "sampler3D", "samplerCube", "sampler2DArray", "sampler2DShadow",
			"sampler2DArrayShadow", "samplerCubeShadow",
			"isampler2D", "usampler2D", "isampler3D", "usampler3D",
			"isampler2DArray", "usampler2DArray",
			"image2D", "image3D", "image2DArray", "iimage2D", "uimage2D",
			"iimage3D", "uimage3D", "iimage2DArray", "uimage2DArray",
			"struct", "uniform", "buffer", "shared", "in", "out", "inout",
			"const", "layout", "coherent", "volatile", "restrict", "readonly", "writeonly",
			"flat", "smooth", "noperspective", "centroid",
			"highp", "mediump", "lowp", "precision",
			"if", "else", "for", "while", "do", "switch", "case", "default",
			"break", "continue", "return", "discard", "true", "false", nullptr
		};
		for (int i = 0; glsl_types_and_kw[i]; i++) {
			add_opt(glsl_types_and_kw[i], glsl_types_and_kw[i], ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
		}

		static const char *glsl_builtin_vars[] = {
			"gl_GlobalInvocationID", "gl_LocalInvocationID", "gl_WorkGroupID",
			"gl_WorkGroupSize", "gl_NumWorkGroups", "gl_LocalInvocationIndex",
			"gl_Position", "gl_FragCoord", "gl_FragDepth", "gl_FrontFacing",
			"gl_PointCoord", "gl_PointSize", "gl_VertexID", "gl_InstanceID",
			"u_projection_matrix", "u_inv_projection_matrix", "u_view_matrix",
			"u_camera_matrix", "u_model_matrix", "u_mvp_matrix",
			"u_normal_matrix", "u_camera_position", "u_time", "u_viewport_size",
			nullptr
		};
		for (int i = 0; glsl_builtin_vars[i]; i++) {
			add_opt(glsl_builtin_vars[i], glsl_builtin_vars[i], ScriptCodeCompletionOption::KIND_MEMBER);
		}

		if (line_prefix_trimmed.empty()) {
			add_opt("[vertex]", "[vertex]\n", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			add_opt("[fragment]", "[fragment]\n", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
			add_opt("// render_mode ", "// render_mode ", ScriptCodeCompletionOption::KIND_PLAIN_TEXT);
		}

		for (int i = 0; GLSL_BUILTIN_FUNCTIONS[i].name; i++) {
			String fn(GLSL_BUILTIN_FUNCTIONS[i].name);
			add_opt(fn, fn + (GLSL_BUILTIN_FUNCTIONS[i].zero_args ? "()" : "("), ScriptCodeCompletionOption::KIND_FUNCTION);
		}

		for (Map<String, ScriptCodeCompletionOption>::Element *E = opts.front(); E; E = E->next()) {
			r_options->push_back(E->get());
		}
		return;
	}

	_check_shader_mode();

	ShaderLanguage sl;
	String calltip;

	sl.complete(p_code, ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_types(), r_options, calltip);

	get_text_edit()->set_code_hint(calltip);
}

void ShaderTextEditor::_validate_script() {
	if (glsl_shader.is_valid()) {
		TextEdit *te = get_text_edit();
		for (int i = 0; i < te->get_line_count(); i++) {
			te->set_line_as_marked(i, false);
		}

		String code = te->get_text();
		int err_line = -1;
		int err_col = 0;
		String err_msg;

		// 1. Check bracket/brace/paren matching and unclosed block comments
		struct BracketPos {
			CharType ch;
			int line;
			int col;
		};
		Vector<BracketPos> stack;
		bool in_line_comment = false;
		bool in_block_comment = false;
		int block_comment_line = 0;
		bool in_string = false;
		int line = 0;
		int col = 0;

		for (int i = 0; i < code.length() && err_line == -1; i++) {
			CharType c = code[i];
			CharType next = (i + 1 < code.length()) ? code[i + 1] : 0;

			if (in_line_comment) {
				if (c == '\n') {
					in_line_comment = false;
					line++;
					col = 0;
				} else {
					col++;
				}
				continue;
			}
			if (in_block_comment) {
				if (c == '*' && next == '/') {
					in_block_comment = false;
					col += 2;
					i++;
				} else if (c == '\n') {
					line++;
					col = 0;
				} else {
					col++;
				}
				continue;
			}
			if (in_string) {
				if (c == '\\' && i + 1 < code.length()) {
					col += 2;
					i++;
				} else if (c == '"') {
					in_string = false;
					col++;
				} else if (c == '\n') {
					in_string = false;
					line++;
					col = 0;
				} else {
					col++;
				}
				continue;
			}

			if (c == '/' && next == '/') {
				in_line_comment = true;
				col += 2;
				i++;
				continue;
			}
			if (c == '/' && next == '*') {
				in_block_comment = true;
				block_comment_line = line;
				col += 2;
				i++;
				continue;
			}
			if (c == '"') {
				in_string = true;
				col++;
				continue;
			}

			if (c == '(' || c == '[' || c == '{') {
				stack.push_back({ c, line, col });
			} else if (c == ')' || c == ']' || c == '}') {
				CharType expected = (c == ')') ? '(' : ((c == ']') ? '[' : '{');
				if (stack.empty() || stack[stack.size() - 1].ch != expected) {
					err_line = line;
					err_col = col;
					err_msg = vformat("error(%d): Unexpected '%s'.", line + 1, String::chr(c));
					break;
				}
				stack.remove(stack.size() - 1);
			}

			if (c == '\n') {
				line++;
				col = 0;
			} else {
				col++;
			}
		}

		if (err_line == -1 && in_block_comment) {
			err_line = block_comment_line;
			err_msg = vformat("error(%d): Unclosed block comment '/*'.", block_comment_line + 1);
		} else if (err_line == -1 && !stack.empty()) {
			BracketPos unclosed = stack[stack.size() - 1];
			err_line = unclosed.line;
			err_col = unclosed.col;
			err_msg = vformat("error(%d): Unclosed '%s'.", unclosed.line + 1, String::chr(unclosed.ch));
		}

		// 2. Validate [vertex]/[fragment] sections and // render_mode directives
		if (err_line == -1) {
			Vector<String> lines = code.split("\n");
			bool has_vertex = false;
			bool has_fragment = false;
			int vert_line = -1;
			int frag_line = -1;
			int section = 0;
			bool vert_non_empty = false;
			bool frag_non_empty = false;

			for (int li = 0; li < lines.size(); li++) {
				String tr = lines[li].strip_edges();
				if (tr == "[vertex]") {
					has_vertex = true;
					vert_line = li;
					section = 1;
					continue;
				}
				if (tr == "[fragment]") {
					has_fragment = true;
					frag_line = li;
					section = 2;
					continue;
				}

				if (tr.begins_with("//")) {
					String comment = tr.substr(2, tr.length() - 2).strip_edges();
					if (comment.begins_with("render_mode")) {
						String modes_str = comment.substr(11, comment.length() - 11).strip_edges().trim_suffix(";");
						Vector<String> modes = modes_str.split(",");
						for (int mi = 0; mi < modes.size(); mi++) {
							String m = modes[mi].strip_edges();
							if (m.empty()) {
								continue;
							}
							if (m != "blend_mix" && m != "blend_add" && m != "blend_sub" && m != "blend_mul" &&
									m != "cull_back" && m != "cull_front" && m != "cull_disabled" &&
									m != "depth_draw_opaque" && m != "depth_draw_always" && m != "depth_draw_never" &&
									m != "depth_test_disable" && m != "transparent") {
								err_line = li;
								err_msg = vformat("error(%d): Unknown GLSLShader render_mode '%s'.", li + 1, m);
								break;
							}
						}
						if (err_line != -1) {
							break;
						}
					}
				}

				if (!tr.empty() && !tr.begins_with("//")) {
					if (section == 1) {
						vert_non_empty = true;
					} else if (section == 2) {
						frag_non_empty = true;
					}
				}
			}

			if (err_line == -1 && (has_vertex || has_fragment)) {
				if (!has_vertex || !vert_non_empty) {
					err_line = MAX(0, frag_line);
					err_msg = vformat("error(%d): GLSLShader has [fragment] without a non-empty [vertex] section.", err_line + 1);
				} else if (!has_fragment || !frag_non_empty) {
					err_line = MAX(0, vert_line);
					err_msg = vformat("error(%d): GLSLShader has [vertex] without a non-empty [fragment] section.", err_line + 1);
				}
			}
		}

		if (err_line != -1) {
			set_error(err_msg);
			set_error_pos(err_line, err_col);
			te->set_line_as_marked(err_line, true);
		} else {
			set_error("");
		}

		emit_signal("script_changed");
		return;
	}

	_check_shader_mode();

	String code = get_text_edit()->get_text();
	//List<StringName> params;
	//shader->get_param_list(&params);

	ShaderLanguage sl;

	Error err = sl.compile(code, ShaderTypes::get_singleton()->get_functions(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_modes(VisualServer::ShaderMode(shader->get_mode())), ShaderTypes::get_singleton()->get_types());

	if (err != OK) {
		String error_text = "error(" + itos(sl.get_error_line()) + "): " + sl.get_error_text();
		set_error(error_text);
		set_error_pos(sl.get_error_line() - 1, 0);
		for (int i = 0; i < get_text_edit()->get_line_count(); i++) {
			get_text_edit()->set_line_as_marked(i, false);
		}
		get_text_edit()->set_line_as_marked(sl.get_error_line() - 1, true);

	} else {
		for (int i = 0; i < get_text_edit()->get_line_count(); i++) {
			get_text_edit()->set_line_as_marked(i, false);
		}
		set_error("");
	}

	emit_signal("script_changed");
}

void ShaderTextEditor::_bind_methods() {
}

ShaderTextEditor::ShaderTextEditor() {
}

/*** SCRIPT EDITOR ******/

void ShaderEditor::_menu_option(int p_option) {
	switch (p_option) {
		case EDIT_UNDO: {
			shader_editor->get_text_edit()->undo();
		} break;
		case EDIT_REDO: {
			shader_editor->get_text_edit()->redo();
		} break;
		case EDIT_CUT: {
			shader_editor->get_text_edit()->cut();
		} break;
		case EDIT_COPY: {
			shader_editor->get_text_edit()->copy();
		} break;
		case EDIT_PASTE: {
			shader_editor->get_text_edit()->paste();
		} break;
		case EDIT_SELECT_ALL: {
			shader_editor->get_text_edit()->select_all();
		} break;
		case EDIT_MOVE_LINE_UP: {
			shader_editor->move_lines_up();
		} break;
		case EDIT_MOVE_LINE_DOWN: {
			shader_editor->move_lines_down();
		} break;
		case EDIT_INDENT_LEFT: {
			if (shader.is_null() && glsl_shader.is_null()) {
				return;
			}

			TextEdit *tx = shader_editor->get_text_edit();
			tx->indent_left();

		} break;
		case EDIT_INDENT_RIGHT: {
			if (shader.is_null() && glsl_shader.is_null()) {
				return;
			}

			TextEdit *tx = shader_editor->get_text_edit();
			tx->indent_right();

		} break;
		case EDIT_DELETE_LINE: {
			shader_editor->delete_lines();
		} break;
		case EDIT_DUPLICATE_SELECTION: {
			shader_editor->duplicate_selection();
		} break;
		case EDIT_TOGGLE_COMMENT: {
			if (shader.is_null() && glsl_shader.is_null()) {
				return;
			}

			shader_editor->toggle_inline_comment("//");

		} break;
		case EDIT_COMPLETE: {
			shader_editor->get_text_edit()->query_code_comple();
		} break;
		case SEARCH_FIND: {
			shader_editor->get_find_replace_bar()->popup_search();
		} break;
		case SEARCH_FIND_NEXT: {
			shader_editor->get_find_replace_bar()->search_next();
		} break;
		case SEARCH_FIND_PREV: {
			shader_editor->get_find_replace_bar()->search_prev();
		} break;
		case SEARCH_REPLACE: {
			shader_editor->get_find_replace_bar()->popup_replace();
		} break;
		case SEARCH_GOTO_LINE: {
			goto_line_dialog->popup_find_line(shader_editor->get_text_edit());
		} break;
		case BOOKMARK_TOGGLE: {
			shader_editor->toggle_bookmark();
		} break;
		case BOOKMARK_GOTO_NEXT: {
			shader_editor->goto_next_bookmark();
		} break;
		case BOOKMARK_GOTO_PREV: {
			shader_editor->goto_prev_bookmark();
		} break;
		case BOOKMARK_REMOVE_ALL: {
			shader_editor->remove_all_bookmarks();
		} break;
		case HELP_DOCS: {
			OS::get_singleton()->shell_open(vformat("%s/tutorials/shaders/shader_reference/index.html", VERSION_DOCS_URL));
		} break;
	}
	if (p_option != SEARCH_FIND && p_option != SEARCH_REPLACE && p_option != SEARCH_GOTO_LINE) {
		shader_editor->get_text_edit()->call_deferred("grab_focus");
	}
}

void ShaderEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			PopupMenu *popup = help_menu->get_popup();
			popup->set_item_icon(popup->get_item_index(HELP_DOCS), get_icon("ExternalLink", "EditorIcons"));
		} break;

		case MainLoop::NOTIFICATION_WM_FOCUS_IN: {
			_check_for_external_edit();
		} break;
	}
}

void ShaderEditor::_params_changed() {
	shader_editor->_validate_script();
}

void ShaderEditor::_editor_settings_changed() {
	shader_editor->get_text_edit()->set_auto_brace_completion(EditorSettings::get_singleton()->get("text_editor/completion/auto_brace_complete"));
	shader_editor->get_text_edit()->set_scroll_pass_end_of_file(EditorSettings::get_singleton()->get("text_editor/cursor/scroll_past_end_of_file"));
	shader_editor->get_text_edit()->set_indent_size(EditorSettings::get_singleton()->get("text_editor/indent/size"));
	shader_editor->get_text_edit()->set_indent_using_spaces(EditorSettings::get_singleton()->get("text_editor/indent/type"));
	shader_editor->get_text_edit()->set_auto_indent(EditorSettings::get_singleton()->get("text_editor/indent/auto_indent"));
	shader_editor->get_text_edit()->set_draw_tabs(EditorSettings::get_singleton()->get("text_editor/indent/draw_tabs"));
	shader_editor->get_text_edit()->set_draw_spaces(EditorSettings::get_singleton()->get("text_editor/indent/draw_spaces"));
	shader_editor->get_text_edit()->set_show_line_numbers(EditorSettings::get_singleton()->get("text_editor/appearance/show_line_numbers"));
	shader_editor->get_text_edit()->set_syntax_coloring(EditorSettings::get_singleton()->get("text_editor/highlighting/syntax_highlighting"));
	shader_editor->get_text_edit()->set_highlight_all_occurrences(EditorSettings::get_singleton()->get("text_editor/highlighting/highlight_all_occurrences"));
	shader_editor->get_text_edit()->set_highlight_current_line(EditorSettings::get_singleton()->get("text_editor/highlighting/highlight_current_line"));
	shader_editor->get_text_edit()->cursor_set_blink_enabled(EditorSettings::get_singleton()->is_caret_blink_active());
	shader_editor->get_text_edit()->cursor_set_blink_speed(EditorSettings::get_singleton()->get("text_editor/cursor/caret_blink_speed"));
	shader_editor->get_text_edit()->add_constant_override("line_spacing", EditorSettings::get_singleton()->get("text_editor/theme/line_spacing"));
	shader_editor->get_text_edit()->cursor_set_block_mode(EditorSettings::get_singleton()->get("text_editor/cursor/block_caret"));
	shader_editor->get_text_edit()->set_smooth_scroll_enabled(EditorSettings::get_singleton()->get("text_editor/navigation/smooth_scrolling"));
	shader_editor->get_text_edit()->set_v_scroll_speed(EditorSettings::get_singleton()->get("text_editor/navigation/v_scroll_speed"));
	shader_editor->get_text_edit()->set_draw_minimap(EditorSettings::get_singleton()->get("text_editor/navigation/show_minimap"));
	shader_editor->get_text_edit()->set_minimap_width((int)EditorSettings::get_singleton()->get("text_editor/navigation/minimap_width") * EDSCALE);
	shader_editor->get_text_edit()->set_show_line_length_guidelines(EditorSettings::get_singleton()->get("text_editor/appearance/show_line_length_guidelines"));
	shader_editor->get_text_edit()->set_line_length_guideline_soft_column(EditorSettings::get_singleton()->get("text_editor/appearance/line_length_guideline_soft_column"));
	shader_editor->get_text_edit()->set_line_length_guideline_hard_column(EditorSettings::get_singleton()->get("text_editor/appearance/line_length_guideline_hard_column"));
	shader_editor->get_text_edit()->set_breakpoint_gutter_enabled(false);
}

void ShaderEditor::_bind_methods() {
	ClassDB::bind_method("_reload_shader_from_disk", &ShaderEditor::_reload_shader_from_disk);
	ClassDB::bind_method("_editor_settings_changed", &ShaderEditor::_editor_settings_changed);
	ClassDB::bind_method("_text_edit_gui_input", &ShaderEditor::_text_edit_gui_input);

	ClassDB::bind_method("_update_bookmark_list", &ShaderEditor::_update_bookmark_list);
	ClassDB::bind_method("_bookmark_item_pressed", &ShaderEditor::_bookmark_item_pressed);

	ClassDB::bind_method("_menu_option", &ShaderEditor::_menu_option);
	ClassDB::bind_method("_params_changed", &ShaderEditor::_params_changed);
	ClassDB::bind_method("apply_shaders", &ShaderEditor::apply_shaders);
	ClassDB::bind_method("save_external_data", &ShaderEditor::save_external_data);
}

void ShaderEditor::ensure_select_current() {
	/*
	if (tab_container->get_child_count() && tab_container->get_current_tab()>=0) {

		ShaderTextEditor *ste = Object::cast_to<ShaderTextEditor>(tab_container->get_child(tab_container->get_current_tab()));
		if (!ste)
			return;
		Ref<Shader> shader = ste->get_edited_shader();
		get_scene()->get_root_node()->call("_resource_selected",shader);
	}*/
}

void ShaderEditor::goto_line_selection(int p_line, int p_begin, int p_end) {
	shader_editor->goto_line_selection(p_line, p_begin, p_end);
}

void ShaderEditor::_check_for_external_edit() {
	Ref<Resource> edited_res = glsl_shader.is_valid() ? Ref<Resource>(glsl_shader) : Ref<Resource>(shader);
	if (edited_res.is_null() || !edited_res.is_valid()) {
		return;
	}

	// internal shader.
	if (edited_res->get_path() == "" || edited_res->get_path().find("local://") != -1 || edited_res->get_path().find("::") != -1) {
		return;
	}

	bool use_autoreload = EDITOR_GET_CACHED(bool, "text_editor/files/auto_reload_scripts_on_external_change");
	if (edited_res->get_last_modified_time() != FileAccess::get_modified_time(edited_res->get_path())) {
		if (use_autoreload) {
			_reload_shader_from_disk();
		} else {
			disk_changed->call_deferred("popup_centered");
		}
	}
}

void ShaderEditor::_reload_shader_from_disk() {
	if (glsl_shader.is_valid()) {
		Ref<GLSLShader> rel_shader = ResourceLoader::load(glsl_shader->get_path(), glsl_shader->get_class(), true);
		ERR_FAIL_COND(!rel_shader.is_valid());

		glsl_shader->set_code(rel_shader->get_code());
		glsl_shader->set_last_modified_time(rel_shader->get_last_modified_time());
		shader_editor->reload_text();
		return;
	}

	Ref<Shader> rel_shader = ResourceLoader::load(shader->get_path(), shader->get_class(), true);
	ERR_FAIL_COND(!rel_shader.is_valid());

	shader->set_code(rel_shader->get_code());
	shader->set_last_modified_time(rel_shader->get_last_modified_time());
	shader_editor->reload_text();
}

void ShaderEditor::edit(const Ref<Shader> &p_shader) {
	if (p_shader.is_null() || !p_shader->is_text_shader()) {
		return;
	}

	if (shader == p_shader) {
		return;
	}

	glsl_shader.unref();
	shader = p_shader;

	shader_editor->set_edited_shader(p_shader);

	//vertex_editor->set_edited_shader(shader,ShaderLanguage::SHADER_MATERIAL_VERTEX);
	// see if already has it
}

void ShaderEditor::edit_glsl(const Ref<GLSLShader> &p_glsl_shader) {
	if (p_glsl_shader.is_null()) {
		return;
	}

	if (glsl_shader == p_glsl_shader) {
		return;
	}

	shader.unref();
	glsl_shader = p_glsl_shader;

	shader_editor->set_edited_glsl_shader(p_glsl_shader);
}

void ShaderEditor::save_external_data(const String &p_str) {
	Ref<Resource> edited_res = glsl_shader.is_valid() ? Ref<Resource>(glsl_shader) : Ref<Resource>(shader);
	if (edited_res.is_null()) {
		disk_changed->hide();
		return;
	}

	apply_shaders();
	if (edited_res->get_path() != "" && edited_res->get_path().find("local://") == -1 && edited_res->get_path().find("::") == -1) {
		//external shader, save it
		ResourceSaver::save(edited_res->get_path(), edited_res);
	}

	disk_changed->hide();
}

void ShaderEditor::apply_shaders() {
	if (glsl_shader.is_valid()) {
		String shader_code = glsl_shader->get_code();
		String editor_code = shader_editor->get_text_edit()->get_text();
		if (shader_code != editor_code) {
			glsl_shader->set_code(editor_code);
			glsl_shader->set_edited(true);
		}
		return;
	}

	if (shader.is_valid()) {
		String shader_code = shader->get_code();
		String editor_code = shader_editor->get_text_edit()->get_text();
		if (shader_code != editor_code) {
			shader->set_code(editor_code);
			shader->set_edited(true);
		}
	}
}

void ShaderEditor::_text_edit_gui_input(const Ref<InputEvent> &ev) {
	Ref<InputEventMouseButton> mb = ev;

	if (mb.is_valid()) {
		if (mb->get_button_index() == BUTTON_RIGHT && mb->is_pressed()) {
			int col, row;
			TextEdit *tx = shader_editor->get_text_edit();
			tx->_get_mouse_pos(mb->get_global_position() - tx->get_global_position(), row, col);
			tx->set_right_click_moves_caret(EditorSettings::get_singleton()->get("text_editor/cursor/right_click_moves_caret"));

			if (tx->is_right_click_moving_caret()) {
				if (tx->is_selection_active()) {
					int from_line = tx->get_selection_from_line();
					int to_line = tx->get_selection_to_line();
					int from_column = tx->get_selection_from_column();
					int to_column = tx->get_selection_to_column();

					if (row < from_line || row > to_line || (row == from_line && col < from_column) || (row == to_line && col > to_column)) {
						// Right click is outside the selected text
						tx->deselect();
					}
				}
				if (!tx->is_selection_active()) {
					tx->cursor_set_line(row, true, false);
					tx->cursor_set_column(col);
				}
			}
			_make_context_menu(tx->is_selection_active(), get_local_mouse_position());
		}
	}

	Ref<InputEventKey> k = ev;
	if (k.is_valid() && k->is_pressed() && k->get_scancode() == KEY_MENU) {
		TextEdit *tx = shader_editor->get_text_edit();
		_make_context_menu(tx->is_selection_active(), (get_global_transform().inverse() * tx->get_global_transform()).xform(tx->_get_cursor_pixel_pos()));
		context_menu->grab_focus();
	}
}

void ShaderEditor::_update_bookmark_list() {
	bookmarks_menu->clear();

	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/remove_all_bookmarks"), BOOKMARK_REMOVE_ALL);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_next_bookmark"), BOOKMARK_GOTO_NEXT);
	bookmarks_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_previous_bookmark"), BOOKMARK_GOTO_PREV);

	Array bookmark_list = shader_editor->get_text_edit()->get_bookmarks_array();
	if (bookmark_list.size() == 0) {
		return;
	}

	bookmarks_menu->add_separator();

	for (int i = 0; i < bookmark_list.size(); i++) {
		String line = shader_editor->get_text_edit()->get_line(bookmark_list[i]).strip_edges();
		// Limit the size of the line if too big.
		if (line.length() > 50) {
			line = line.substr(0, 50);
		}

		bookmarks_menu->add_item(String::num((int)bookmark_list[i] + 1) + " - \"" + line + "\"");
		bookmarks_menu->set_item_metadata(bookmarks_menu->get_item_count() - 1, bookmark_list[i]);
	}
}

void ShaderEditor::_bookmark_item_pressed(int p_idx) {
	if (p_idx < 4) { // Any item before the separator.
		_menu_option(bookmarks_menu->get_item_id(p_idx));
	} else {
		shader_editor->goto_line(bookmarks_menu->get_item_metadata(p_idx));
	}
}

void ShaderEditor::_make_context_menu(bool p_selection, Vector2 p_position) {
	context_menu->clear();
	if (p_selection) {
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/cut"), EDIT_CUT);
		context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/copy"), EDIT_COPY);
	}

	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/paste"), EDIT_PASTE);
	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/select_all"), EDIT_SELECT_ALL);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/undo"), EDIT_UNDO);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/redo"), EDIT_REDO);

	context_menu->add_separator();
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_left"), EDIT_INDENT_LEFT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_right"), EDIT_INDENT_RIGHT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	context_menu->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_bookmark"), BOOKMARK_TOGGLE);

	context_menu->set_position(get_global_transform().xform(p_position));
	context_menu->set_size(Vector2(1, 1));
	context_menu->popup();
}

ShaderEditor::ShaderEditor(EditorNode *p_node) {
	shader_editor = memnew(ShaderTextEditor);
	shader_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	shader_editor->add_constant_override("separation", 0);
	shader_editor->set_anchors_and_margins_preset(Control::PRESET_WIDE);

	shader_editor->connect("script_changed", this, "apply_shaders");
	EditorSettings::get_singleton()->connect("settings_changed", this, "_editor_settings_changed");

	shader_editor->get_text_edit()->set_callhint_settings(
			EditorSettings::get_singleton()->get("text_editor/completion/put_callhint_tooltip_below_current_line"),
			EditorSettings::get_singleton()->get("text_editor/completion/callhint_tooltip_offset"));

	shader_editor->get_text_edit()->set_select_identifiers_on_hover(true);
	shader_editor->get_text_edit()->set_context_menu_enabled(false);
	shader_editor->get_text_edit()->connect("gui_input", this, "_text_edit_gui_input");

	shader_editor->update_editor_settings();

	context_menu = memnew(PopupMenu);
	add_child(context_menu);
	context_menu->connect("id_pressed", this, "_menu_option");
	context_menu->set_hide_on_window_lose_focus(true);

	VBoxContainer *main_container = memnew(VBoxContainer);
	HBoxContainer *hbc = memnew(HBoxContainer);

	edit_menu = memnew(MenuButton);
	edit_menu->set_text(TTR("Edit"));
	edit_menu->set_switch_on_hover(true);
	edit_menu->get_popup()->set_hide_on_window_lose_focus(true);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/undo"), EDIT_UNDO);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/redo"), EDIT_REDO);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/cut"), EDIT_CUT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/copy"), EDIT_COPY);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/paste"), EDIT_PASTE);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/select_all"), EDIT_SELECT_ALL);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_up"), EDIT_MOVE_LINE_UP);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/move_down"), EDIT_MOVE_LINE_DOWN);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_left"), EDIT_INDENT_LEFT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/indent_right"), EDIT_INDENT_RIGHT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/delete_line"), EDIT_DELETE_LINE);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/toggle_comment"), EDIT_TOGGLE_COMMENT);
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/duplicate_selection"), EDIT_DUPLICATE_SELECTION);
	edit_menu->get_popup()->add_separator();
	edit_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/complete_symbol"), EDIT_COMPLETE);
	edit_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	search_menu = memnew(MenuButton);
	search_menu->set_text(TTR("Search"));
	search_menu->set_switch_on_hover(true);
	search_menu->get_popup()->set_hide_on_window_lose_focus(true);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find"), SEARCH_FIND);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_next"), SEARCH_FIND_NEXT);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/find_previous"), SEARCH_FIND_PREV);
	search_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/replace"), SEARCH_REPLACE);
	search_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	MenuButton *goto_menu = memnew(MenuButton);
	goto_menu->set_text(TTR("Go To"));
	goto_menu->set_switch_on_hover(true);
	goto_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	goto_menu->get_popup()->add_shortcut(ED_GET_SHORTCUT("script_text_editor/goto_line"), SEARCH_GOTO_LINE);
	goto_menu->get_popup()->add_separator();

	bookmarks_menu = memnew(PopupMenu);
	bookmarks_menu->set_name("Bookmarks");
	goto_menu->get_popup()->add_child(bookmarks_menu);
	goto_menu->get_popup()->add_submenu_item(TTR("Bookmarks"), "Bookmarks");
	_update_bookmark_list();
	bookmarks_menu->connect("about_to_show", this, "_update_bookmark_list");
	bookmarks_menu->connect("index_pressed", this, "_bookmark_item_pressed");

	help_menu = memnew(MenuButton);
	help_menu->set_text(TTR("Help"));
	help_menu->set_switch_on_hover(true);
	help_menu->get_popup()->add_item(TTR("Online Docs"), HELP_DOCS);
	help_menu->get_popup()->connect("id_pressed", this, "_menu_option");

	add_child(main_container);
	main_container->add_child(hbc);
	hbc->add_child(search_menu);
	hbc->add_child(edit_menu);
	hbc->add_child(goto_menu);
	hbc->add_child(help_menu);
	hbc->add_style_override("panel", p_node->get_gui_base()->get_stylebox("ScriptEditorPanel", "EditorStyles"));
	main_container->add_child(shader_editor);

	goto_line_dialog = memnew(GotoLineDialog);
	add_child(goto_line_dialog);

	disk_changed = memnew(ConfirmationDialog);

	VBoxContainer *vbc = memnew(VBoxContainer);
	disk_changed->add_child(vbc);

	Label *dl = memnew(Label);
	dl->set_text(TTR("This shader has been modified on on disk.\nWhat action should be taken?"));
	vbc->add_child(dl);

	disk_changed->connect("confirmed", this, "_reload_shader_from_disk");
	disk_changed->get_ok()->set_text(TTR("Reload"));

	disk_changed->add_button(TTR("Resave"), !OS::get_singleton()->get_swap_ok_cancel(), "resave");
	disk_changed->connect("custom_action", this, "save_external_data");

	add_child(disk_changed);

	_editor_settings_changed();
}

void ShaderEditorPlugin::edit(Object *p_object) {
	if (GLSLShader *gs = Object::cast_to<GLSLShader>(p_object)) {
		shader_editor->edit_glsl(gs);
	} else if (Shader *s = Object::cast_to<Shader>(p_object)) {
		shader_editor->edit(s);
	}
}

bool ShaderEditorPlugin::handles(Object *p_object) const {
	Shader *shader = Object::cast_to<Shader>(p_object);
	GLSLShader *glsl_shader = Object::cast_to<GLSLShader>(p_object);
	return (shader != nullptr && shader->is_text_shader()) || glsl_shader != nullptr;
}

void ShaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button->show();
		editor->make_bottom_panel_item_visible(shader_editor);

	} else {
		button->hide();
		if (shader_editor->is_visible_in_tree()) {
			editor->hide_bottom_panel();
		}
		shader_editor->apply_shaders();
	}
}

void ShaderEditorPlugin::selected_notify() {
	shader_editor->ensure_select_current();
}

void ShaderEditorPlugin::save_external_data() {
	shader_editor->save_external_data();
}

void ShaderEditorPlugin::apply_changes() {
	shader_editor->apply_shaders();
}

ShaderEditorPlugin::ShaderEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	shader_editor = memnew(ShaderEditor(p_node));

	shader_editor->set_custom_minimum_size(Size2(0, 300) * EDSCALE);
	button = editor->add_bottom_panel_item(TTR("Shader"), shader_editor);
	button->hide();
}

ShaderEditorPlugin::~ShaderEditorPlugin() {
}
