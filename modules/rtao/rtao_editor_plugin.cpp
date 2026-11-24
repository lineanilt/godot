#ifdef TOOLS_ENABLED

#include "rtao_editor_plugin.h"

#include "baked_rtao.h"
#include "local_rtao.h"

#include "editor/editor_node.h"
#include "editor/plugins/spatial_editor_plugin.h"
#include "scene/gui/tool_button.h"

static void rtao_add_wire_box(Vector<Vector3> &lines, const Vector3 &e) {
	Vector3 pts[8] = {
		Vector3(-e.x, -e.y, -e.z), Vector3(e.x, -e.y, -e.z),
		Vector3(e.x, -e.y, e.z), Vector3(-e.x, -e.y, e.z),
		Vector3(-e.x, e.y, -e.z), Vector3(e.x, e.y, -e.z),
		Vector3(e.x, e.y, e.z), Vector3(-e.x, e.y, e.z),
	};
	int idx[24] = {
		0, 1, 1, 2, 2, 3, 3, 0,
		4, 5, 5, 6, 6, 7, 7, 4,
		0, 4, 1, 5, 2, 6, 3, 7
	};
	for (int i = 0; i < 24; i++) {
		lines.push_back(pts[idx[i]]);
	}
}

static void rtao_add_wire_circle(Vector<Vector3> &lines, const Vector3 &axis_u, const Vector3 &axis_v, const Vector3 &offset, float r_u, float r_v, int segs = 32) {
	for (int i = 0; i < segs; i++) {
		float a0 = (Math_PI * 2.0f * i) / segs;
		float a1 = (Math_PI * 2.0f * (i + 1)) / segs;
		lines.push_back(offset + axis_u * (Math::cos(a0) * r_u) + axis_v * (Math::sin(a0) * r_v));
		lines.push_back(offset + axis_u * (Math::cos(a1) * r_u) + axis_v * (Math::sin(a1) * r_v));
	}
}

RTAOGizmoPlugin::RTAOGizmoPlugin() {
	create_material("rtao_local", Color(0.85f, 0.45f, 1.0f, 0.7f));
	create_material("rtao_baked", Color(0.35f, 0.95f, 0.7f, 0.7f));
}

bool RTAOGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<LocalRTAO>(p_spatial) != nullptr ||
		   Object::cast_to<BakedRTAO>(p_spatial) != nullptr;
}

String RTAOGizmoPlugin::get_name() const {
	return "RTAOGizmo";
}

int RTAOGizmoPlugin::get_priority() const {
	return -1;
}

void RTAOGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	p_gizmo->clear();
	Spatial *sp = p_gizmo->get_spatial_node();
	Vector<Vector3> lines;

	if (LocalRTAO *lv = Object::cast_to<LocalRTAO>(sp)) {
		Vector3 e = lv->get_extents();
		switch (lv->get_shape()) {
			case LocalRTAO::SHAPE_BOX:
				rtao_add_wire_box(lines, e);
				break;
			case LocalRTAO::SHAPE_SPHERE: {
				float r = MAX(e.x, MAX(e.y, e.z));
				rtao_add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(), r, r);
				rtao_add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(), r, r);
				rtao_add_wire_circle(lines, Vector3(0, 1, 0), Vector3(0, 0, 1), Vector3(), r, r);
			} break;
			case LocalRTAO::SHAPE_ELLIPSOID:
				rtao_add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(), e.x, e.y);
				rtao_add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(), e.x, e.z);
				rtao_add_wire_circle(lines, Vector3(0, 1, 0), Vector3(0, 0, 1), Vector3(), e.y, e.z);
				break;
			case LocalRTAO::SHAPE_CYLINDER:
				rtao_add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(0, e.y, 0), e.x, e.z);
				rtao_add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(0, -e.y, 0), e.x, e.z);
				lines.push_back(Vector3(e.x, -e.y, 0)); lines.push_back(Vector3(e.x, e.y, 0));
				lines.push_back(Vector3(-e.x, -e.y, 0)); lines.push_back(Vector3(-e.x, e.y, 0));
				lines.push_back(Vector3(0, -e.y, e.z)); lines.push_back(Vector3(0, e.y, e.z));
				lines.push_back(Vector3(0, -e.y, -e.z)); lines.push_back(Vector3(0, e.y, -e.z));
				break;
			case LocalRTAO::SHAPE_CONE:
				rtao_add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(0, -e.y, 0), e.x, e.z);
				lines.push_back(Vector3(e.x, -e.y, 0)); lines.push_back(Vector3(0, e.y, 0));
				lines.push_back(Vector3(-e.x, -e.y, 0)); lines.push_back(Vector3(0, e.y, 0));
				lines.push_back(Vector3(0, -e.y, e.z)); lines.push_back(Vector3(0, e.y, 0));
				lines.push_back(Vector3(0, -e.y, -e.z)); lines.push_back(Vector3(0, e.y, 0));
				break;
		}
		p_gizmo->add_lines(lines, get_material("rtao_local", p_gizmo));
	} else if (BakedRTAO *bv = Object::cast_to<BakedRTAO>(sp)) {
		rtao_add_wire_box(lines, bv->get_extents());
		p_gizmo->add_lines(lines, get_material("rtao_baked", p_gizmo));
	}
}

RTAOEditorPlugin::RTAOEditorPlugin(EditorNode *p_editor) {
	bake_button = memnew(ToolButton);
	bake_button->set_text(TTR("Bake RTAO Volume"));
	bake_button->hide();
	bake_button->connect("pressed", this, "_bake_pressed");
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake_button);

	gizmo_plugin.instance();
}

RTAOEditorPlugin::~RTAOEditorPlugin() {
}

void RTAOEditorPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_bake_pressed"), &RTAOEditorPlugin::_bake_pressed);
}

void RTAOEditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		if (SpatialEditor::get_singleton() && gizmo_plugin.is_valid()) {
			SpatialEditor::get_singleton()->add_gizmo_plugin(gizmo_plugin);
		}
	} else if (p_what == NOTIFICATION_EXIT_TREE) {
		if (SpatialEditor::get_singleton() && gizmo_plugin.is_valid()) {
			SpatialEditor::get_singleton()->remove_gizmo_plugin(gizmo_plugin);
		}
	}
}

void RTAOEditorPlugin::_bake_pressed() {
	if (selected_baked) {
		selected_baked->bake();
	}
}

bool RTAOEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<BakedRTAO>(p_object) != nullptr;
}

void RTAOEditorPlugin::edit(Object *p_object) {
	selected_baked = Object::cast_to<BakedRTAO>(p_object);
}

void RTAOEditorPlugin::make_visible(bool p_visible) {
	if (bake_button) {
		bake_button->set_visible(p_visible && selected_baked != nullptr);
	}
}

#endif // TOOLS_ENABLED
