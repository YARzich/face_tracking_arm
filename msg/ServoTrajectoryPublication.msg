# Full immutable checked trajectory retained for this publication ID.
# This diagnostic event is emitted from the bounded non-RT drain, not by
# looking up mutable history from a second thread.
uint8 KIND_SERVO_TRAJECTORY=1
uint8 KIND_BOOTSTRAP_RESET=2
uint8 publication_kind
std_msgs/Header header
uint64 publication_id
builtin_interfaces/Time stationary_time
uint32 tail_start_index
uint64 hardware_epoch
uint64 hardware_read_sequence
uint64 control_tick_sequence
trajectory_msgs/JointTrajectory trajectory
uint64 dropped_publication_events
