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
# True only for a short forecast during an explicitly healthy no-face interval.
bool face_is_prediction false
# Forecast validity deadline; zero for a measured face and all other modes.
builtin_interfaces/Time face_prediction_expiry
# Selector track generation; this is not a persistent person identity.
int64 target_id -1
# Valid only in SEARCH: sweep or local_then_sweep. No synthetic face measurement.
string search_pattern
# Perception loss may finish an already adopted, validated joint detour before
# entering idle. False for explicit pause, missing commands and safety holds.
bool allow_recovery false
