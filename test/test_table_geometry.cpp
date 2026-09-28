// Copyright 2026 YARzich
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include "face_tracking_arm/table_geometry.hpp"

namespace face_tracking_arm::tracking
{
TEST(TableGeometry, SupportsBoxCylinderAndMoveItPoseFactoring)
{
  const auto table = make_table("box", {1.2, .8, .05}, {.1, .2, .795}, .3);
  moveit_msgs::msg::PlanningScene scene;
  scene.world.collision_objects.push_back(table);
  EXPECT_TRUE(has_table(scene, table));
  auto & measured = scene.world.collision_objects[0];
  measured.pose = measured.primitive_poses[0];
  measured.primitive_poses[0] = geometry_msgs::msg::Pose();
  measured.primitive_poses[0].orientation.w = 1.0;
  EXPECT_TRUE(has_table(scene, table));
  measured.pose.position.x += .01;
  EXPECT_FALSE(has_table(scene, table));
  const auto original = make_table("cylinder", {.05, .5}, {0, 0, .725}, 0);
  scene.world.collision_objects = {original};
  EXPECT_TRUE(has_table(scene, original));
}
TEST(TableGeometry, RejectsInvalidGeometry)
{
  EXPECT_THROW(make_table("box", {.05, .5}, {0, 0, .725}, 0), std::invalid_argument);
  EXPECT_THROW(make_table("cylinder", {-.05, .5}, {0, 0, .725}, 0), std::invalid_argument);
  EXPECT_THROW(make_table("cylinder", {0, .5}, {0, 0, .725}, 0), std::invalid_argument);
  EXPECT_THROW(make_table("box", {1, 1, 0}, {0, 0, .725}, 0), std::invalid_argument);
  EXPECT_THROW(make_table("box", {}, {0, 0, .725}, 0), std::invalid_argument);
}

TEST(TableGeometry, ZeroDimensionsRemoveTheObjectAndWaitForItsAbsence)
{
  for (const auto & shape : {"box", "cylinder"}) {
    const auto disabled = make_table(
      shape, std::vector<double>(std::string(shape) == "box" ? 3U : 2U, 0.0),
      {0, 0, .725}, 0);
    EXPECT_EQ(disabled.operation, moveit_msgs::msg::CollisionObject::REMOVE);
    EXPECT_TRUE(disabled.primitives.empty());
    EXPECT_TRUE(disabled.primitive_poses.empty());
    moveit_msgs::msg::PlanningScene scene;
    EXPECT_TRUE(has_table(scene, disabled));
    scene.world.collision_objects.push_back(make_table("cylinder", {.05, .5}, {0, 0, .725}, 0));
    EXPECT_FALSE(has_table(scene, disabled));
    scene.world.collision_objects[0].id = "another_obstacle";
    EXPECT_TRUE(has_table(scene, disabled));
  }
}
}  // namespace face_tracking_arm::tracking
