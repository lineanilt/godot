#ifdef TOOLS_ENABLED

#include "volumetrics_editor_plugin.h"

#include "baked_fog_volume.h"
#include "local_fog_volume.h"
#include "volumetric_fog_focus.h"

#include "editor/editor_node.h"
#include "editor/plugins/spatial_editor_plugin.h"
#include "scene/gui/check_box.h"
#include "scene/gui/tool_button.h"
#include "scene/main/viewport.h"

static void add_wire_box(Vector<Vector3> &lines, const Vector3 &e) {
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

static void add_wire_circle(Vector<Vector3> &lines, const Vector3 &axis_u, const Vector3 &axis_v, const Vector3 &offset, float r_u, float r_v, int segs = 32) {
	for (int i = 0; i < segs; i++) {
		float a0 = (Math_PI * 2.0f * i) / segs;
		float a1 = (Math_PI * 2.0f * (i + 1)) / segs;
		lines.push_back(offset + axis_u * (Math::cos(a0) * r_u) + axis_v * (Math::sin(a0) * r_v));
		lines.push_back(offset + axis_u * (Math::cos(a1) * r_u) + axis_v * (Math::sin(a1) * r_v));
	}
}

VolumetricsGizmoPlugin::VolumetricsGizmoPlugin() {
	create_material("volumetrics_local", Color(0.35f, 0.85f, 1.0f, 0.7f));
	create_material("volumetrics_focus", Color(1.0f, 0.85f, 0.25f, 0.7f));
	create_material("volumetrics_baked", Color(0.45f, 1.0f, 0.55f, 0.7f));
}

bool VolumetricsGizmoPlugin::has_gizmo(Spatial *p_spatial) {
	return Object::cast_to<LocalFogVolume>(p_spatial) != nullptr ||
		   Object::cast_to<VolumetricFogFocus>(p_spatial) != nullptr ||
		   Object::cast_to<BakedFogVolume>(p_spatial) != nullptr;
}

String VolumetricsGizmoPlugin::get_name() const {
	return "VolumetricsGizmo";
}

int VolumetricsGizmoPlugin::get_priority() const {
	return -1;
}

void VolumetricsGizmoPlugin::redraw(EditorSpatialGizmo *p_gizmo) {
	p_gizmo->clear();
	Spatial *sp = p_gizmo->get_spatial_node();
	Vector<Vector3> lines;

	if (LocalFogVolume *lv = Object::cast_to<LocalFogVolume>(sp)) {
		Vector3 e = lv->get_extents();
		switch (lv->get_shape()) {
			case LocalFogVolume::SHAPE_BOX:
				add_wire_box(lines, e);
				break;
			case LocalFogVolume::SHAPE_SPHERE: {
				float r = MAX(e.x, MAX(e.y, e.z));
				add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(), r, r);
				add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(), r, r);
				add_wire_circle(lines, Vector3(0, 1, 0), Vector3(0, 0, 1), Vector3(), r, r);
			} break;
			case LocalFogVolume::SHAPE_ELLIPSOID:
				add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(), e.x, e.y);
				add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(), e.x, e.z);
				add_wire_circle(lines, Vector3(0, 1, 0), Vector3(0, 0, 1), Vector3(), e.y, e.z);
				break;
			case LocalFogVolume::SHAPE_CYLINDER:
				add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(0, e.y, 0), e.x, e.z);
				add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(0, -e.y, 0), e.x, e.z);
				lines.push_back(Vector3(e.x, -e.y, 0)); lines.push_back(Vector3(e.x, e.y, 0));
				lines.push_back(Vector3(-e.x, -e.y, 0)); lines.push_back(Vector3(-e.x, e.y, 0));
				lines.push_back(Vector3(0, -e.y, e.z)); lines.push_back(Vector3(0, e.y, e.z));
				lines.push_back(Vector3(0, -e.y, -e.z)); lines.push_back(Vector3(0, e.y, -e.z));
				break;
			case LocalFogVolume::SHAPE_CONE:
				add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(0, -e.y, 0), e.x, e.z);
				lines.push_back(Vector3(e.x, -e.y, 0)); lines.push_back(Vector3(0, e.y, 0));
				lines.push_back(Vector3(-e.x, -e.y, 0)); lines.push_back(Vector3(0, e.y, 0));
				lines.push_back(Vector3(0, -e.y, e.z)); lines.push_back(Vector3(0, e.y, 0));
				lines.push_back(Vector3(0, -e.y, -e.z)); lines.push_back(Vector3(0, e.y, 0));
				break;
		}
		p_gizmo->add_lines(lines, get_material("volumetrics_local", p_gizmo));
	} else if (VolumetricFogFocus *vf = Object::cast_to<VolumetricFogFocus>(sp)) {
		float r = vf->get_radius();
		add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(), r, r);
		add_wire_circle(lines, Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(), r, r);
		add_wire_circle(lines, Vector3(0, 1, 0), Vector3(0, 0, 1), Vector3(), r, r);
		p_gizmo->add_lines(lines, get_material("volumetrics_focus", p_gizmo));
	} else if (BakedFogVolume *bv = Object::cast_to<BakedFogVolume>(sp)) {
		add_wire_box(lines, bv->get_extents());
		p_gizmo->add_lines(lines, get_material("volumetrics_baked", p_gizmo));
	}
}

void EditorPropertyViewportRenderEffects::_on_toggled(bool p_pressed) {
	if (updating) {
		return;
	}
	Object *obj = get_edited_object();
	if (!obj) {
		return;
	}
	UndoRedo *ur = EditorNode::get_undo_redo();
	if (ur) {
		bool had_meta = obj->has_meta("render_effects");
		bool prev = had_meta ? bool(obj->get_meta("render_effects")) : true;
		ur->create_action(TTR("Toggle Viewport Render Effects"));
		ur->add_do_method(obj, "set_meta", "render_effects", p_pressed);
		if (had_meta) {
			ur->add_undo_method(obj, "set_meta", "render_effects", prev);
		} else {
			ur->add_undo_method(obj, "remove_meta", "render_effects");
		}
		ur->add_do_method(this, "_refresh_checkbox");
		ur->add_undo_method(this, "_refresh_checkbox");
		ur->commit_action();
	} else {
		obj->set_meta("render_effects", p_pressed);
		_refresh_checkbox();
	}
}

void EditorPropertyViewportRenderEffects::_refresh_checkbox() {
	update_property();
}

void EditorPropertyViewportRenderEffects::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_toggled", "pressed"), &EditorPropertyViewportRenderEffects::_on_toggled);
	ClassDB::bind_method(D_METHOD("_refresh_checkbox"), &EditorPropertyViewportRenderEffects::_refresh_checkbox);
}

void EditorPropertyViewportRenderEffects::update_property() {
	Object *obj = get_edited_object();
	if (!obj) {
		return;
	}
	bool val = obj->has_meta("render_effects") ? bool(obj->get_meta("render_effects")) : true;
	updating = true;
	checkbox->set_pressed(val);
	updating = false;
}

EditorPropertyViewportRenderEffects::EditorPropertyViewportRenderEffects() {
	checkbox = memnew(CheckBox);
	checkbox->set_text(TTR("On"));
	add_child(checkbox);
	add_focusable(checkbox);
	checkbox->connect("toggled", this, "_on_toggled");
}

bool EditorInspectorPluginViewportEffects::can_handle(Object *p_object) {
	return Object::cast_to<Viewport>(p_object) != nullptr;
}

void EditorInspectorPluginViewportEffects::parse_category(Object *p_object, const String &p_category) {
	if (p_category == "Viewport" && Object::cast_to<Viewport>(p_object)) {
		EditorPropertyViewportRenderEffects *prop = memnew(EditorPropertyViewportRenderEffects);
		prop->set_label(TTR("Render Effects"));
		add_property_editor("metadata/render_effects", prop);
	}
}

VolumetricsEditorPlugin::VolumetricsEditorPlugin(EditorNode *p_editor) {
	bake_button = memnew(ToolButton);
	bake_button->set_text(TTR("Bake Fog Volume"));
	bake_button->hide();
	bake_button->connect("pressed", this, "_bake_pressed");
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake_button);

	gizmo_plugin.instance();
	viewport_effects_inspector.instance();
}

VolumetricsEditorPlugin::~VolumetricsEditorPlugin() {
}

void VolumetricsEditorPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_bake_pressed"), &VolumetricsEditorPlugin::_bake_pressed);
}

void VolumetricsEditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		if (SpatialEditor::get_singleton() && gizmo_plugin.is_valid()) {
			SpatialEditor::get_singleton()->add_gizmo_plugin(gizmo_plugin);
		}
		if (viewport_effects_inspector.is_valid()) {
			add_inspector_plugin(viewport_effects_inspector);
		}
	} else if (p_what == NOTIFICATION_EXIT_TREE) {
		if (SpatialEditor::get_singleton() && gizmo_plugin.is_valid()) {
			SpatialEditor::get_singleton()->remove_gizmo_plugin(gizmo_plugin);
		}
		if (viewport_effects_inspector.is_valid()) {
			remove_inspector_plugin(viewport_effects_inspector);
		}
	}
}

void VolumetricsEditorPlugin::_bake_pressed() {
	if (selected_baked) {
		selected_baked->bake();
	}
}

bool VolumetricsEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<BakedFogVolume>(p_object) != nullptr;
}

void VolumetricsEditorPlugin::edit(Object *p_object) {
	selected_baked = Object::cast_to<BakedFogVolume>(p_object);
}

void VolumetricsEditorPlugin::make_visible(bool p_visible) {
	if (bake_button) {
		bake_button->set_visible(p_visible && selected_baked != nullptr);
	}
}

#endif // TOOLS_ENABLED
