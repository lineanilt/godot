#include "register_types.h"

#include "core/class_db.h"

#include "baked_fog_volume.h"
#include "local_fog_volume.h"
#include "volumetric_fog.h"
#include "volumetric_fog_focus.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "volumetrics_editor_plugin.h"
#endif

void register_volumetrics_types() {
	ClassDB::register_class<VolumetricFog>();
	ClassDB::register_class<LocalFogVolume>();
	ClassDB::register_class<VolumetricFogFocus>();
	ClassDB::register_class<BakedFogVolume>();

#ifdef TOOLS_ENABLED
	EditorPlugins::add_by_type<VolumetricsEditorPlugin>();
#endif
}

void unregister_volumetrics_types() {
}
