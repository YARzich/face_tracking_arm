// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#ifndef FIXTURES__COLLISION_BRAKE__LOGGED_EVENT_STATES_HPP_
#define FIXTURES__COLLISION_BRAKE__LOGGED_EVENT_STATES_HPP_

#include <array>

namespace face_tracking_arm::test::collision_brake
{

// Logged event 1 primary witness and accepted-by-rows hypothetical candidate from
// collision_brake_analysis_20261026/contract/geometry_probe.cpp.
inline constexpr std::array<double, 6> kEvent1Position{
  0.0082355120576991618, -0.050712033457800523, -0.58718823378481888,
  0.013227675976160929, -0.93331927523153213, -0.0076932299365188063};
inline constexpr std::array<double, 6> kEvent1Velocity{
  0.080195680764502938, 0.08955492464146747, 0.0036012902018541749,
  0.13459930918538238, -0.13803302183868624, -0.079162999081462487};
inline constexpr std::array<double, 6> kEvent1Acceleration{
  -0.64604542612339133, 1.1164903695710187, 0.067929345660259696,
  -1.1817005703488563, -0.88271784028562117, 0.67195627276711023};
inline constexpr std::array<double, 6> kEvent1CandidateVelocity{
  0.075696523432921131, 0.098782584852869795, 0.006126016950557007,
  0.12081154829110537, -0.1486516162167901, -0.070443436353791383};
inline constexpr std::array<double, 6> kEvent1DistanceGradient{
  0.0510788659403, -0.0828237042464, 0.203991365044,
  -0.0385962063407, -0.0247159045628, 8.70478819234e-17};

// Logged event 2 primary witness and candidate, which was already outside the
// stopping viability envelope and is evaluated separately from event 1.
inline constexpr std::array<double, 6> kEvent2Position{
  0.011049705917356852, -0.039325668794130242, -0.58740496644642526,
  0.01200235531448517, -0.9504842526359476, -0.008982050187418581};
inline constexpr std::array<double, 6> kEvent2Velocity{
  -0.00021707533436989392, 0.062736029758331271, 0.004,
  -0.095914003455475558, -0.14810361424023899, 0.020748312125836759};
inline constexpr std::array<double, 6> kEvent2Acceleration{
  0.22170773940491198, -1.2129710797087079, 0.2,
  -0.63681279854482653, 0.31538428096996585, -0.072726088834840913};
inline constexpr std::array<double, 6> kEvent2CandidateVelocity{
  2.0596792257854268e-09, 0.048606318961244191, 0.008,
  -0.10028213144092382, -0.14694942969074817, 0.018021051237488348};
inline constexpr std::array<double, 6> kEvent2DistanceGradient{
  -0.0505747727884, -0.0801249752515, 0.205969059256,
  0.043426202903, -0.028026567733, 2.68285827582e-16};

}  // namespace face_tracking_arm::test::collision_brake

#endif  // FIXTURES__COLLISION_BRAKE__LOGGED_EVENT_STATES_HPP_
