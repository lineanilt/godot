#include "register_types.h"

#include "core/class_db.h"

#include "baked_rtao.h"
#include "local_rtao.h"
#include "rtao.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "rtao_editor_plugin.h"
#endif

void register_rtao_types() {
	ClassDB::register_class<RTAO>();
	ClassDB::register_class<LocalRTAO>();
	ClassDB::register_class<BakedRTAO>();

#ifdef TOOLS_ENABLED
	EditorPlugins::add_by_type<RTAOEditorPlugin>();
#endif
}

void unregister_rtao_types() {
}
