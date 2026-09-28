# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""Close approaches and lateral movement of one rendered person, without ROS."""

from .stress_motion import StressMotion


class CloseStressMotion(StressMotion):
    """
    Approach in two stages, move around the screen and leave time to recover.

    Radii describe the model origin relative to the robot base, not the measured
    distance between face and screen. The second approach is closer than the
    preferred face distance to a screen settled at the first close stop.
    """

    duration_sec = 120.0

    # Time, model radius, bearing in degrees, approximate face height, phase, visibility.
    _keyframes = (
        (0.0, 2.1, 0.0, 1.75, 'START_HOLD', True),
        (10.0, 2.1, 0.0, 1.75, 'APPROACH', True),
        (22.0, 1.0, 0.0, 1.55, 'NEAR_HOLD', True),
        (28.0, 1.0, 0.0, 1.55, 'CLOSE_APPROACH', True),
        (32.0, .75, 0.0, 1.55, 'CLOSE_HOLD', True),
        (42.0, .75, 0.0, 1.55, 'CLOSE_LEFT', True),
        (50.0, .75, -30.0, 1.55, 'CLOSE_CROSS', True),
        (60.0, .75, 30.0, 1.70, 'CLOSE_LOWER', True),
        (68.0, .75, 30.0, 1.40, 'CLOSE_CENTER', True),
        (76.0, .75, 0.0, 1.40, 'LOW_CLOSE_HOLD', True),
        (84.0, .75, 0.0, 1.40, 'BACK_OFF', True),
        (94.0, 1.0, 0.0, 1.55, 'RECOVERY_HOLD', True),
        (100.0, 1.0, 0.0, 1.55, 'RETURN_START', True),
        (110.0, 2.1, 0.0, 1.75, 'FINAL_HOLD', True),
        (120.0, 2.1, 0.0, 1.75, 'COMPLETE', True),
    )
