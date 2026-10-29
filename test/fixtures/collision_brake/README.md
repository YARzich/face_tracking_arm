# Collision brake fixtures

`primary_event89_max_iterations.qpsnap` is copied from
`/tmp/face_tracking_arm_qp_failures/session_1793107596859180780/event_89_primary.qpsnap`.
It records the first primary-stage OSQP failure (`raw_status=-2`, 2500 iterations)
for the event89 collision-brake state. The test reconstructs this input through
the public `HierarchicalVelocityQp::solve` API and verifies the captured CSC
structure, objective, constraints, warm start, centering shift, and effective
solver settings. SHA-256:

`7b7f3fb1f882b7dc686b55679baa027ac8061006de5e40bfe1990b96a77381d1`

See `/home/yaroslav/ros2_ws/log/collision_brake_implementation_20261026/task3/capture/primary_retry_red_green.md`
for the bounded retry behavior and verification results. The fixture documents
a solver regression input; it is not itself evidence of native runtime acceptance.

`primary_event116_max_iterations.qpsnap` is the exact guarded primary failure
from `/tmp/face_tracking_arm_qp_failures/session_1793110517286996251/event_116_primary.qpsnap`.
It is used with the same public-solver reconstruction as event89 to regress the
bounded alternate-rho retry and its post-clamp hard-row feasibility. SHA-256:

`2f20828a0943024626d6b35ecec45bac38375689204d4b6f8a432a0f84d509bb`
