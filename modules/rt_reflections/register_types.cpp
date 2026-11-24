#include "register_types.h"
#include "core/class_db.h"
#include "rt_reflections.h"

void register_rt_reflections_types() {
    ClassDB::register_class<RTReflections>();
}

void unregister_rt_reflections_types() {
}
