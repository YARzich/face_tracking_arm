# Command and measurement are expressed in header.frame_id.
std_msgs/Header header
uint8 FACE=0
uint8 HOLD=1
uint8 REST=2
uint8 POSE=3
uint8 SEARCH=4
uint8 mode
geometry_msgs/Pose pose
# Valid only in FACE mode; retains the original observation time.
geometry_msgs/Point face
builtin_interfaces/Time face_stamp
# Valid only in SEARCH: sweep or local_then_sweep. No synthetic face measurement.
string search_pattern
