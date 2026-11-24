#include "register_types.h"
#include "core/class_db.h"
#include "motion_blur.h"

void register_motion_blur_types() {
	ClassDB::register_class<MotionBlur>();
}

void unregister_motion_blur_types() {
}
