#ifndef RTAO_EDITOR_PLUGIN_H
#define RTAO_EDITOR_PLUGIN_H

#ifdef TOOLS_ENABLED

#include "editor/editor_plugin.h"
#include "editor/spatial_editor_gizmos.h"

class BakedRTAO;
class ToolButton;

class RTAOGizmoPlugin : public EditorSpatialGizmoPlugin {
	GDCLASS(RTAOGizmoPlugin, EditorSpatialGizmoPlugin);

public:
	RTAOGizmoPlugin();

	virtual bool has_gizmo(Spatial *p_spatial);
	virtual String get_name() const;
	virtual int get_priority() const;
	virtual void redraw(EditorSpatialGizmo *p_gizmo);
};

class RTAOEditorPlugin : public EditorPlugin {
	GDCLASS(RTAOEditorPlugin, EditorPlugin);

	BakedRTAO *selected_baked = nullptr;
	ToolButton *bake_button = nullptr;
	Ref<RTAOGizmoPlugin> gizmo_plugin;

	void _bake_pressed();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	RTAOEditorPlugin(EditorNode *p_editor);
	~RTAOEditorPlugin();

	virtual String get_name() const { return "RTAO"; }
	virtual bool handles(Object *p_object) const;
	virtual void edit(Object *p_object);
	virtual void make_visible(bool p_visible);
};

#endif // TOOLS_ENABLED

#endif // RTAO_EDITOR_PLUGIN_H
