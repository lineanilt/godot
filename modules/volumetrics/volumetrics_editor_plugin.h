#ifndef VOLUMETRICS_EDITOR_PLUGIN_H
#define VOLUMETRICS_EDITOR_PLUGIN_H

#ifdef TOOLS_ENABLED

#include "editor/editor_inspector.h"
#include "editor/editor_plugin.h"
#include "editor/spatial_editor_gizmos.h"

class BakedFogVolume;
class CheckBox;
class ToolButton;

class EditorPropertyViewportRenderEffects : public EditorProperty {
	GDCLASS(EditorPropertyViewportRenderEffects, EditorProperty);

	CheckBox *checkbox = nullptr;
	bool updating = false;

	void _on_toggled(bool p_pressed);
	void _refresh_checkbox();

protected:
	static void _bind_methods();

public:
	virtual void update_property();
	EditorPropertyViewportRenderEffects();
};

class EditorInspectorPluginViewportEffects : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginViewportEffects, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object);
	virtual void parse_category(Object *p_object, const String &p_category);
};

class VolumetricsGizmoPlugin : public EditorSpatialGizmoPlugin {
	GDCLASS(VolumetricsGizmoPlugin, EditorSpatialGizmoPlugin);

public:
	VolumetricsGizmoPlugin();

	virtual bool has_gizmo(Spatial *p_spatial);
	virtual String get_name() const;
	virtual int get_priority() const;
	virtual void redraw(EditorSpatialGizmo *p_gizmo);
};

class VolumetricsEditorPlugin : public EditorPlugin {
	GDCLASS(VolumetricsEditorPlugin, EditorPlugin);

	BakedFogVolume *selected_baked = nullptr;
	ToolButton *bake_button = nullptr;
	Ref<VolumetricsGizmoPlugin> gizmo_plugin;
	Ref<EditorInspectorPluginViewportEffects> viewport_effects_inspector;

	void _bake_pressed();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	VolumetricsEditorPlugin(EditorNode *p_editor);
	~VolumetricsEditorPlugin();

	virtual String get_name() const { return "Volumetrics"; }
	virtual bool handles(Object *p_object) const;
	virtual void edit(Object *p_object);
	virtual void make_visible(bool p_visible);
};

#endif // TOOLS_ENABLED

#endif // VOLUMETRICS_EDITOR_PLUGIN_H
