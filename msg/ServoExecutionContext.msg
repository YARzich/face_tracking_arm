# Internal bounded-queue context emitted by collision-aware Servo.
# Header stamp identifies the control tick; it is not a hardware receipt stamp.
std_msgs/Header header
uint64 tick_sequence
uint64 cycle_duration_ns
uint8 CONTROLLER_TRACKING=0
uint8 CONTROLLER_BRAKING=1
uint8 CONTROLLER_WAITING=2
uint8 controller_mode
bool face_tracking_published
uint32 active_collision_rows
bool collision_distance_available
float64 minimum_world_distance_m
string<=128 closest_pair
bool active_world_pair_available
float64 active_world_pair_distance_m
string<=128 active_world_pair
uint64 hardware_epoch
uint64 hardware_read_sequence
bool target_available
bool target_used_for_reference
uint8 target_mode
int64 target_id
builtin_interfaces/Time target_source_stamp
builtin_interfaces/Time face_stamp
string<=128 target_frame
geometry_msgs/Point target_position
geometry_msgs/Point face_position
# No extra TF lookup is performed for recording. False means no exact transform
# provenance was captured by the command path.
bool tf_provenance_available
geometry_msgs/TransformStamped target_transform
uint64 active_publication_id
uint64 latest_publication_id
uint64 braking_tail_publication_id
builtin_interfaces/Time braking_tail_stationary_stamp
uint64 dropped_context_records
uint64 dropped_publication_events
