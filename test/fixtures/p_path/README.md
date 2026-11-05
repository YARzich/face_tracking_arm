# P-PATH profile regression fixtures

`r1_projected_points.csv` contains 944 joint-space points reconstructed from the
recorded folded-path case. The reconstruction is approximate: it produced 360
retained legs and 40.29 s in the old rest-to-rest profile. It reproduces interior
stops; it is not an exact runtime replay or a full-maneuver duration oracle.
The CSV in this directory is the preserved test input; temporary reconstruction
programs and duplicate data were removed after the P-PATH work.

Common settings: xArm6 limits from the test URDF, velocity
`min(3.1, URDF limit)`, acceleration 19 rad/s^2, jerk 400 rad/s^3,
period 0.01 s, speed scale 1.0. Tests check the sampled q/v/a/jerk contract.

An interior near-stop episode means all sampled joint velocities are below
0.01 rad/s, bounded by motion before and after it. This is a diagnostic measure,
not a minimum allowed speed. Whole-profile start/end samples are excluded.

R2 generates the same arc with 10, 50, 100 and 200 segments in
[test_compact_joint_profile.cpp](../../test_compact_joint_profile.cpp).
The old profile took 4.4/12/20/32 s; the smooth profile takes 2.80 s for each
density without interior near-stops. R1 takes 4.38 s without interior near-stops.
These are profile results, not acceptance of every runtime maneuver.
