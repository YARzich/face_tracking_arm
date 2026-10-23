# Provenance for the last hardware read/write cycle. Header stamp is the
# publication time in the shared ROS clock domain.
uint8 SOURCE_UNSPECIFIED=0
uint8 SOURCE_HARDWARE=1
uint8 SOURCE_SIMULATION=2
uint8 FAULT_NONE=0
uint8 FAULT_READ=1
uint8 FAULT_WRITE=2
uint8 FAULT_WRITE_BOUNDS=3
uint8 FAULT_LIMITS_UNAVAILABLE=4
uint8 CANDIDATE_FAILURE_JOINT_NONE=255

std_msgs/Header header
uint8 source_kind
uint64 epoch
bool runtime_actions_enabled
bool hardware_active
bool fault_latched
uint64 fault_sequence
uint8 fault_kind
int32 fault_code
uint64 fault_read_attempt_sequence
uint64 fault_epoch
builtin_interfaces/Time fault_stamp

# read_sequence advances only after a successful read. A failed attempt updates
# read_attempt_sequence/read_ok/read_code while q/dq/read_stamp remain the last
# successful sample.
uint64 read_attempt_sequence
uint64 read_sequence
# Epoch captured at the beginning of the successful read that produced q/dq.
uint64 read_epoch
bool read_ok
int32 read_code
builtin_interfaces/Time read_stamp
uint32 read_age_us
float64[6] q
float64[6] dq

# Result for the latest hardware write call. The driver-local sequence is not
# an identifier for the application's published trajectory points.
uint64 write_sequence
builtin_interfaces/Time write_stamp
uint64 driver_command_sequence
uint32 write_in_flight
bool write_attempted
bool write_ok
int32 write_code
float64[6] command_double
float32[6] command_float

# Latest sampled candidate, including a rejected candidate that never reached
# the SDK. candidate_sequence advances independently of driver_command_sequence.
uint64 candidate_sequence
float64[6] candidate_double
float32[6] candidate_float
bool candidate_valid
uint8 candidate_failure_joint_index

# Immutable candidate captured with the first latched driver fault. Read faults
# have no candidate and report fault_candidate_joint_index=255.
bool fault_candidate_valid
uint8 fault_candidate_joint_index
float64[6] fault_candidate_double
float32[6] fault_candidate_float

# RobotMsg provenance is retained independently from the read sample.
uint64 robot_msg_sequence
builtin_interfaces/Time robot_msg_stamp
uint32 robot_msg_age_us
uint8 robot_state
uint8 robot_mode
uint8 robot_mt_able
uint8 robot_mt_brake
uint16 robot_error_code
uint16 robot_warn_code

# Driver-side ServoJ inhibit acknowledgement. This does not confirm a physical
# stop; the STOP service result, RobotMsg, and measured standstill are separate.
bool inhibit_latched
uint64 inhibit_epoch
uint64 inhibit_request_id
