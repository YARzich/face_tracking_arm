# Copyright 2026 YARzich
# SPDX-License-Identifier: MIT

"""OpenCV hand-eye calibration from a fixed square marker and stamped poses."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CalibrationSample:
    """One synchronized pose pair; transforms are parent_from_child matrices."""

    joint_positions_rad: tuple
    base_from_flange: tuple
    camera_from_marker: tuple
    marker_corners_px: tuple = ()
    image_stamp_sec: float = 0.0
    joint_state_stamp_sec: float = 0.0
    camera_frame: str = 'camera_optical_frame'
    flange_frame: str = 'link6'
    camera_matrix: tuple = ()
    distortion: tuple = ()
    image_size: tuple = ()


def planned_joint_offsets_deg():
    """Return training then held-out J5/J6 offsets from the measured start pose."""
    training = ((0.0, 0.0), (10.0, 0.0), (-10.0, 0.0), (0.0, 15.0),
                (0.0, -15.0), (10.0, 15.0), (-10.0, -15.0))
    held_out = ((7.5, 0.0), (-7.5, 0.0), (0.0, 7.5), (0.0, -7.5))
    return training, held_out


def _rotation_vector(matrix):
    """Extract a stable axis-angle vector from a 3x3 rotation matrix."""
    cosine = max(-1.0, min(1.0, (sum(matrix[i][i] for i in range(3)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < 1e-7:
        return (0.0, 0.0, 0.0)
    scale = 2.0 * math.sin(angle)
    if abs(scale) < 1e-7:
        # Near pi, the largest diagonal component determines the stable axis.
        axis = [math.sqrt(max(0.0, (matrix[i][i] + 1.0) / 2.0)) for i in range(3)]
        pivot = max(range(3), key=lambda i: axis[i])
        if axis[pivot] < 1e-7:
            raise ValueError('Cannot determine rotation axis near pi')
        for i in range(3):
            if i != pivot:
                axis[i] = (matrix[pivot][i] + matrix[i][pivot]) / (4.0 * axis[pivot])
    else:
        axis = [(matrix[2][1] - matrix[1][2]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale,
                (matrix[1][0] - matrix[0][1]) / scale]
    return tuple(value * angle for value in axis)


def _rotation_between(left, right):
    return [[sum(left[k][row] * right[k][column] for k in range(3))
             for column in range(3)] for row in range(3)]


def validate_rigid_transform(transform, name='transform'):
    """Require a finite homogeneous matrix and a proper SO(3) rotation."""
    import numpy as np

    value = np.asarray(transform, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected a finite 4x4 transform')
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f'{name}: invalid homogeneous bottom row')
    rotation = value[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
            not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-4)):
        raise ValueError(f'{name}: rotation must be a proper SO(3) matrix')
    return value


def validate_training_samples(samples, expected_marker_side_m):
    """Reject unsynchronized, insufficient, or non-observable training data."""
    if not math.isfinite(expected_marker_side_m) or expected_marker_side_m <= 0:
        raise ValueError('Measure the printed marker square side in meters; it must be > 0')
    if len(samples) < 6:
        raise ValueError('Need at least six training poses; capture the seven planned poses')
    camera_frame = samples[0].camera_frame
    flange_frame = samples[0].flange_frame
    calibration = (samples[0].camera_matrix, samples[0].distortion,
                   samples[0].image_size)
    for sample in samples:
        if (len(sample.joint_positions_rad) != 6 or
                not all(math.isfinite(value) for value in sample.joint_positions_rad)):
            raise ValueError('Each sample needs six finite measured joint positions')
        if len(sample.base_from_flange) != 4 or len(sample.camera_from_marker) != 4:
            raise ValueError('Each sample needs stamped flange TF and marker pose')
        if (not math.isfinite(sample.image_stamp_sec) or
                not math.isfinite(sample.joint_state_stamp_sec) or
                abs(sample.image_stamp_sec - sample.joint_state_stamp_sec) > .05):
            raise ValueError('Image and joint state stamps differ by more than 0.05 s')
        if not sample.camera_frame or not sample.flange_frame:
            raise ValueError('Camera and flange frame ids are required')
        if sample.camera_frame != camera_frame or sample.flange_frame != flange_frame:
            raise ValueError('Training camera/flange frame ids must remain constant')
        if (sample.camera_matrix, sample.distortion, sample.image_size) != calibration:
            raise ValueError('Camera intrinsics, distortion, and image size must remain constant')
        if sample.marker_corners_px and not all(
                math.isfinite(value) for point in sample.marker_corners_px for value in point):
            raise ValueError('Marker corner coordinates must be finite')
        validate_rigid_transform(sample.base_from_flange, 'base_from_flange')
        validate_rigid_transform(sample.camera_from_marker, 'camera_from_marker')
        if sample.camera_from_marker[2][3] <= 0:
            raise ValueError('The marker pose is behind the camera; check detection/frame id')
    start = samples[0].joint_positions_rad
    for sample in samples[1:]:
        if any(abs(sample.joint_positions_rad[index] - start[index]) > math.radians(.1)
               for index in range(4)):
            raise ValueError('Calibration may move only J5 and J6; J1-J4 changed')
        if (abs(sample.joint_positions_rad[4] - start[4]) > math.radians(15.5) or
                abs(sample.joint_positions_rad[5] - start[5]) > math.radians(15.5)):
            raise ValueError('Calibration J5/J6 motion must stay within 15 degrees')
    joint5 = [round(math.degrees(sample.joint_positions_rad[4]), 2) for sample in samples]
    joint6 = [round(math.degrees(sample.joint_positions_rad[5]), 2) for sample in samples]
    if len(set(joint5)) < 2 or len(set(joint6)) < 2:
        raise ValueError('Training data must include motion of both J5 and J6')
    axes = []
    anchor = samples[0].base_from_flange
    anchor_rotation = [row[:3] for row in anchor[:3]]
    for sample in samples[1:]:
        rotation = [row[:3] for row in sample.base_from_flange[:3]]
        relative = _rotation_between(anchor_rotation, rotation)
        vector = _rotation_vector(relative)
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude > math.radians(2.0):
            axes.append(tuple(value / magnitude for value in vector))
    cross_products = [
        math.sqrt(sum((axes[i][a] * axes[j][b] - axes[i][b] * axes[j][a]) ** 2
                      for a, b in ((0, 1), (1, 2), (2, 0))))
        for i in range(len(axes)) for j in range(i + 1, len(axes))]
    if len(axes) < 3 or (cross_products and max(cross_products) < 0.10):
        raise ValueError('Training motion is degenerate; include varied J5 and J6 rotations')
    return True


def solve_hand_eye(samples, expected_marker_side_m):
    """Return T_flange_camera from OpenCV's eye-in-hand hand-eye calibration."""
    validate_training_samples(samples, expected_marker_side_m)
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError('OpenCV and NumPy are required for mount calibration') from error
    gripper_rotations = []
    gripper_translations = []
    target_rotations = []
    target_translations = []
    for sample in samples:
        base_from_flange = np.asarray(sample.base_from_flange, dtype=float)
        camera_from_marker = np.asarray(sample.camera_from_marker, dtype=float)
        gripper_rotations.append(base_from_flange[:3, :3])
        gripper_translations.append(base_from_flange[:3, 3].reshape(3, 1))
        target_rotations.append(camera_from_marker[:3, :3])
        target_translations.append(camera_from_marker[:3, 3].reshape(3, 1))
    rotation, translation = cv2.calibrateHandEye(
        gripper_rotations, gripper_translations, target_rotations, target_translations,
        method=cv2.CALIB_HAND_EYE_PARK)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    validate_rigid_transform(transform, 'OpenCV result')
    return transform.tolist()


def fixed_marker_consistency(samples, flange_from_camera):
    """Measure held-out fixed-marker pose spread in the robot base frame."""
    import numpy as np

    flange_from_camera = np.asarray(flange_from_camera, dtype=float)
    targets = [np.asarray(sample.base_from_flange) @ flange_from_camera @
               np.asarray(sample.camera_from_marker) for sample in samples]
    positions = np.asarray([target[:3, 3] for target in targets])
    center = positions.mean(axis=0)
    position_errors = np.linalg.norm(positions - center, axis=1)
    reference_rotation = targets[0][:3, :3]
    angle_errors = []
    for target in targets:
        delta = reference_rotation.T @ target[:3, :3]
        angle_errors.append(math.degrees(math.acos(max(
            -1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0)))))
    return {
        'target_base_position_m': center.tolist(),
        'position_rms_m': float(np.sqrt(np.mean(position_errors ** 2))),
        'position_max_m': float(position_errors.max()),
        'orientation_rms_deg': float(np.sqrt(np.mean(np.square(angle_errors)))),
        'orientation_max_deg': float(max(angle_errors)),
    }


def validate_held_out(training_samples, samples, flange_from_camera,
                      camera_matrix, distortion, marker_side_m,
                      *, max_position_rms_m=.01, max_position_m=.02,
                      max_orientation_rms_deg=2.0, max_reprojection_rms_px=2.0):
    """Check independent poses, positive PnP depth and image reprojection."""
    import cv2
    import numpy as np

    validate_rigid_transform(flange_from_camera, 'flange_from_camera')
    if len(samples) < 2:
        raise ValueError('Capture at least two independent held-out poses')
    if len(training_samples) < 6:
        raise ValueError('Held-out validation requires the accepted training set')
    validate_training_samples(training_samples, marker_side_m)
    start = training_samples[0].joint_positions_rad
    camera_frame = training_samples[0].camera_frame
    flange_frame = training_samples[0].flange_frame
    camera_calibration = (training_samples[0].camera_matrix,
                          training_samples[0].distortion,
                          training_samples[0].image_size)
    for sample in samples:
        if (len(sample.joint_positions_rad) != 6 or
                not all(math.isfinite(value) for value in sample.joint_positions_rad)):
            raise ValueError('Held-out sample needs six finite measured joint positions')
        if (not math.isfinite(sample.image_stamp_sec) or
                not math.isfinite(sample.joint_state_stamp_sec) or
                abs(sample.image_stamp_sec - sample.joint_state_stamp_sec) > .05):
            raise ValueError('Held-out image and joint state stamps differ by more than 0.05 s')
        if sample.camera_frame != camera_frame or sample.flange_frame != flange_frame:
            raise ValueError('Held-out camera/flange frame ids differ from training frames')
        if (sample.camera_matrix, sample.distortion, sample.image_size) != camera_calibration:
            raise ValueError('Camera calibration or image size changed during held-out capture')
        validate_rigid_transform(sample.base_from_flange, 'held-out base_from_flange')
        validate_rigid_transform(sample.camera_from_marker, 'held-out camera_from_marker')
        if sample.camera_from_marker[2][3] <= 0:
            raise ValueError('Held-out marker pose is behind the camera')
    training_angles = [(sample.joint_positions_rad[4], sample.joint_positions_rad[5])
                       for sample in training_samples]
    held_out_angles = []
    for sample in samples:
        if (len(sample.joint_positions_rad) != 6 or
                any(abs(sample.joint_positions_rad[index] - start[index]) > math.radians(.1)
                    for index in range(4))):
            raise ValueError('Held-out poses may move only J5 and J6')
        if (abs(sample.joint_positions_rad[4] - start[4]) > math.radians(15.5) or
                abs(sample.joint_positions_rad[5] - start[5]) > math.radians(15.5)):
            raise ValueError('Held-out J5/J6 motion must stay within 15 degrees')
        pair = (sample.joint_positions_rad[4], sample.joint_positions_rad[5])
        if any(math.hypot(pair[0] - known[0], pair[1] - known[1]) < math.radians(1.0)
               for known in training_angles + held_out_angles):
            raise ValueError('Held-out poses must be distinct from training and each other')
        held_out_angles.append(pair)
    # Freeze the marker pose from training only. Held-out samples cannot move
    # that reference to hide a systematic extrinsic bias.
    training_marker_poses = [np.asarray(sample.base_from_flange) @
                             np.asarray(flange_from_camera) @
                             np.asarray(sample.camera_from_marker)
                             for sample in training_samples]
    reference_position = np.mean([pose[:3, 3] for pose in training_marker_poses], axis=0)
    reference_rotation = _mean_rotation([pose[:3, :3] for pose in training_marker_poses])
    base_from_marker_reference = np.eye(4)
    base_from_marker_reference[:3, :3] = reference_rotation
    base_from_marker_reference[:3, 3] = reference_position
    held_out_marker_poses = [np.asarray(sample.base_from_flange) @
                             np.asarray(flange_from_camera) @
                             np.asarray(sample.camera_from_marker) for sample in samples]
    position_errors = [float(np.linalg.norm(pose[:3, 3] - reference_position))
                       for pose in held_out_marker_poses]
    orientation_errors = []
    for pose in held_out_marker_poses:
        delta = reference_rotation.T @ pose[:3, :3]
        orientation_errors.append(math.degrees(math.acos(max(
            -1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0)))))
    report = {
        'position_rms_m': float(np.sqrt(np.mean(np.square(position_errors)))),
        'position_max_m': float(max(position_errors)),
        'orientation_rms_deg': float(np.sqrt(np.mean(np.square(orientation_errors)))),
        'orientation_max_deg': float(max(orientation_errors)),
    }
    intrinsics = np.asarray(camera_matrix, dtype=float)
    coefficients = np.asarray(distortion, dtype=float)
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or
            intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or
            not np.isfinite(coefficients).all()):
        raise ValueError('CameraInfo intrinsics are not finite 3x3 values')
    if camera_calibration[0]:
        if (not np.allclose(intrinsics.reshape(-1), camera_calibration[0], atol=1e-9) or
                not np.allclose(coefficients.reshape(-1), camera_calibration[1], atol=1e-9)):
            raise ValueError('Validation intrinsics differ from the frozen training CameraInfo')
    if not math.isfinite(marker_side_m) or marker_side_m <= 0:
        raise ValueError('Measure the printed marker square side in meters; it must be > 0')
    half_side = marker_side_m / 2
    object_points = np.asarray([
        [-half_side, half_side, 0], [half_side, half_side, 0],
        [half_side, -half_side, 0], [-half_side, -half_side, 0]], dtype=float)
    errors = []
    for sample in samples:
        if len(sample.marker_corners_px) != 4:
            raise ValueError('Held-out image is missing the four marker corners')
        if not all(math.isfinite(value) for point in sample.marker_corners_px for value in point):
            raise ValueError('Held-out image corner coordinates must be finite')
        # Predict the held-out camera view from the marker pose frozen using
        # training captures. Do not use held-out PnP pose to reconstruct it;
        # that would cancel the candidate extrinsic and make reprojection tautological.
        camera_from_marker = (
            np.linalg.inv(np.asarray(flange_from_camera)) @
            np.linalg.inv(np.asarray(sample.base_from_flange)) @ base_from_marker_reference)
        if camera_from_marker[2, 3] <= 0:
            raise ValueError('Predicted held-out marker pose is behind the camera')
        rotation, _ = cv2.Rodrigues(camera_from_marker[:3, :3])
        projected, _ = cv2.projectPoints(
            object_points, rotation,
            camera_from_marker[:3, 3],
            intrinsics, coefficients)
        measured = np.asarray(sample.marker_corners_px, dtype=float).reshape(4, 2)
        errors.extend(np.linalg.norm(measured - projected.reshape(4, 2), axis=1).tolist())
    report['reprojection_rms_px'] = float(np.sqrt(np.mean(np.square(errors))))
    report['reprojection_max_px'] = float(max(errors))
    failures = []
    if report['position_rms_m'] > max_position_rms_m:
        failures.append(f"position RMS {report['position_rms_m']:.4f} m")
    if report['position_max_m'] > max_position_m:
        failures.append(f"position max {report['position_max_m']:.4f} m")
    if report['orientation_rms_deg'] > max_orientation_rms_deg:
        failures.append(f"orientation RMS {report['orientation_rms_deg']:.2f} deg")
    if report['reprojection_rms_px'] > max_reprojection_rms_px:
        failures.append(f"reprojection RMS {report['reprojection_rms_px']:.2f} px")
    if failures:
        raise ValueError('Held-out validation failed: ' + ', '.join(failures))
    report['accepted'] = True
    return report


def _mean_rotation(rotations):
    """Project the summed training rotations back onto SO(3)."""
    import numpy as np

    left, _, right = np.linalg.svd(np.sum(rotations, axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def detect_square_marker(image, camera_matrix, distortion, side_m, marker_id=0):
    """Detect the generated OpenCV 4x4 marker and estimate T_camera_marker."""
    import cv2
    import numpy as np

    camera_matrix = np.ascontiguousarray(camera_matrix, dtype=np.float64)
    distortion = np.ascontiguousarray(distortion, dtype=np.float64).reshape(-1, 1)
    if (camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all() or
            not np.isfinite(distortion).all()):
        raise ValueError('CameraInfo must contain finite 3x3 intrinsics and distortion values')
    aruco = getattr(cv2, 'aruco', None)
    if aruco is None:
        raise RuntimeError('This OpenCV build has no aruco module; use manual measurement')
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = None
    if hasattr(aruco, 'ArucoDetector') and hasattr(aruco, 'DetectorParameters'):
        parameters = aruco.DetectorParameters()
    elif hasattr(aruco, 'DetectorParameters_create'):
        parameters = aruco.DetectorParameters_create()
    if parameters is not None and hasattr(aruco, 'CORNER_REFINE_APRILTAG'):
        # The rendered calibration marker can sit on a gray plane, where the
        # default contour corners expand into the surrounding quiet zone.
        parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    if hasattr(aruco, 'ArucoDetector'):
        detector = (aruco.ArucoDetector(dictionary, parameters) if parameters is not None
                    else aruco.ArucoDetector(dictionary))
        corners, ids, _ = detector.detectMarkers(image)
    else:
        if parameters is None:
            corners, ids, _ = aruco.detectMarkers(image, dictionary)
        else:
            corners, ids, _ = aruco.detectMarkers(
                image, dictionary, parameters=parameters)
    if ids is None:
        return None
    found = next((index for index, value in enumerate(ids.flatten()) if value == marker_id), None)
    if found is None:
        return None
    points = np.ascontiguousarray(corners[found].reshape(1, 4, 2), dtype=np.float32)
    if parameters is not None and hasattr(aruco, 'CORNER_REFINE_APRILTAG'):
        # AprilTag's line fit identifies the marker boundary; subpixel fitting
        # then uses the rendered edge intensities to refine those four points.
        gray = (image if len(image.shape) == 2 else
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), criteria)
    points = points.reshape(4, 2)
    if hasattr(aruco, 'estimatePoseSingleMarkers'):
        rotation_vectors, translations, _ = aruco.estimatePoseSingleMarkers(
            [points], side_m, camera_matrix, distortion)
        rotation, _ = cv2.Rodrigues(rotation_vectors[0, 0])
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = translations[0, 0]
        return transform.tolist(), points.tolist()
    raise RuntimeError('OpenCV aruco pose estimation is unavailable; use manual measurement')


def generate_marker(path, pixels=1200, marker_id=0):
    """Generate the exact printable/viewable marker with a white quiet zone."""
    import cv2
    import numpy as np

    aruco = getattr(cv2, 'aruco', None)
    if aruco is None:
        raise RuntimeError('This OpenCV build has no aruco module')
    border = max(32, pixels // 8)
    marker_size = pixels - 2 * border
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, 'generateImageMarker'):
        marker = aruco.generateImageMarker(dictionary, marker_id, marker_size)
    else:
        marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
        aruco.drawMarker(dictionary, marker_id, marker_size, marker, 1)
    image = np.full((pixels, pixels), 255, dtype=np.uint8)
    image[border:border + marker_size, border:border + marker_size] = marker
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write marker image to {path}')
