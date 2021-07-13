import numpy as np
import numpy as np
import vg

from typing import Callable, Dict, List, Optional, Tuple

from smg.skeletons import Keypoint, KeypointOrienter, KeypointUtil, Skeleton3D

from .vicon_interface import ViconInterface


class ViconSkeletonDetector:
    """A 3D skeleton detector based on a Vicon system."""

    # CONSTRUCTOR

    def __init__(self, vicon: ViconInterface, *, is_person: Callable[[str, ViconInterface], bool],
                 use_vicon_poses: bool = False):
        """
        Construct a 3D skeleton detector based on a Vicon system.

        :param vicon:           The Vicon interface.
        :param is_person:       A function that determines whether or not the specified subject is a person.
        :param use_vicon_poses: Whether to use the joint poses produced by the Vicon system.
        """
        self.__vicon: ViconInterface = vicon
        self.__is_person: Callable[[str, ViconInterface], bool] = is_person
        self.__use_vicon_poses: bool = use_vicon_poses

        # Specify which keypoints are joined to form bones.
        self.__keypoint_pairs: List[Tuple[str, str]] = [
            ("Head", "Neck"),
            ("LAnkle", "LKnee"),
            ("LAnkle", "LToe"),
            ("LElbow", "LShoulder"),
            ("LElbow", "LWrist"),
            ("LHip", "MidHip"),
            ("LKnee", "LHip"),
            ("LShoulder", "Neck"),
            ("MidHip", "Neck"),
            ("MidHip", "RHip"),
            ("Neck", "RShoulder"),
            ("RAnkle", "RKnee"),
            ("RAnkle", "RToe"),
            ("RElbow", "RShoulder"),
            ("RElbow", "RWrist"),
            ("RHip", "RKnee")
        ]

        # A mapping from Vicon marker names to keypoints. Note that whilst these markers directly correspond to
        # useful keypoints, some other keypoints (e.g. MidHip) have to be computed based on the positions of
        # multiple markers (see the detect_skeletons function).
        self.__marker_to_keypoint: Dict[str, str] = {
            "LANK": "LAnkle",
            "LELB": "LElbow",
            "LKNE": "LKnee",
            "LSHO": "LShoulder",
            "LTOE": "LToe",
            "RANK": "RAnkle",
            "RELB": "RElbow",
            "RKNE": "RKnee",
            "RSHO": "RShoulder",
            "RTOE": "RToe"
        }

        # A mapping specifying the midhip-from-rest transforms for the keypoints.
        # FIXME: These need to be properly checked next time I have access to the Vicon system.
        lm: np.ndarray = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        rm: np.ndarray = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

        self.__midhip_from_rests: Dict[str, np.ndarray] = {
            "LElbow": lm,
            "LHip": np.eye(3),
            "LKnee": np.eye(3),
            "LShoulder": lm,
            "MidHip": np.eye(3),
            "Neck": np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
            "RElbow": rm,
            "RHip": np.eye(3),
            "RKnee": np.eye(3),
            "RShoulder": rm
        }

        # A mapping specifying the child to parent relationships between the keypoints.
        self.__parent_keypoints: Dict[str, str] = {
            "LElbow": "LShoulder",
            "LHip": "MidHip",
            "LKnee": "LHip",
            "LShoulder": "Neck",
            "Neck": "MidHip",
            "RElbow": "RShoulder",
            "RHip": "MidHip",
            "RKnee": "RHip",
            "RShoulder": "Neck"
        }

        # A mapping from Vicon segment names to the keypoints that control the poses of the corresponding bones.
        self.__segment_to_keypoint: Dict[str, str] = {
            "L_Elbow": "LElbow",
            "R_Elbow": "RElbow",
            "L_Femur": "LHip",
            "R_Femur": "RHip",
            "L_Humerus": "LShoulder",
            "R_Humerus": "RShoulder",
            "L_Tibia": "LKnee",
            "R_Tibia": "RKnee"
        }

    # PUBLIC METHODS

    def detect_skeletons(self) -> Dict[str, Skeleton3D]:
        """
        Detect 3D skeletons in the scene using the Vicon system.

        :return:    The detected 3D skeletons.
        """
        skeletons: Dict[str, Skeleton3D] = {}

        # For each relevant Vicon subject:
        for subject in self.__vicon.get_subject_names():
            # If the subject is not a person, skip it.
            if not self.__is_person(subject, self.__vicon):
                continue

            # Otherwise, get its marker positions.
            marker_positions: Dict[str, np.ndarray] = self.__vicon.get_marker_positions(subject)

            # Try to hallucinate some of the missing markers (where feasible).
            ViconSkeletonDetector.__try_hallucinate_missing_markers(marker_positions)

            # Construct the keypoints for the skeleton based on the available marker positions.
            keypoints: Dict[str, Keypoint] = {}

            # First add keypoints whose positions can be derived from a single marker.
            for marker_name, keypoint in self.__marker_to_keypoint.items():
                marker_position: Optional[np.ndarray] = marker_positions.get(marker_name)
                if marker_position is not None:
                    keypoints[keypoint] = Keypoint(keypoint, marker_position)

            # Then add keypoints whose positions are the result of averaging the positions of several Vicon markers.
            ViconSkeletonDetector.__try_add_keypoint(
                "Head", [["LBHD", "LFHD", "RBHD", "RFHD"]], marker_positions, keypoints
            )
            ViconSkeletonDetector.__try_add_keypoint("LHip", [["LASI", "LPSI"]], marker_positions, keypoints)
            ViconSkeletonDetector.__try_add_keypoint(
                "LWrist", [["LWRA", "LWRB"], ["LWRA"], ["LWRB"], ["LFIN"]], marker_positions, keypoints
            )
            ViconSkeletonDetector.__try_add_keypoint(
                "MidHip", [["LASI", "LPSI", "RASI", "RPSI"], ["LASI", "RPSI"], ["RASI", "LPSI"]],
                marker_positions, keypoints
            )
            ViconSkeletonDetector.__try_add_keypoint("Neck", [["LSHO", "RSHO"]], marker_positions, keypoints)
            ViconSkeletonDetector.__try_add_keypoint("RHip", [["RASI", "RPSI"]], marker_positions, keypoints)
            ViconSkeletonDetector.__try_add_keypoint(
                "RWrist", [["RWRA", "RWRB"], ["RWRA"], ["RWRB"], ["RFIN"]], marker_positions, keypoints
            )

            # If we're using the joint poses from the Vicon system:
            if self.__use_vicon_poses:
                global_keypoint_poses: Dict[str, np.ndarray] = {}

                # Look up the poses of those keypoints that have a corresponding Vicon segment.
                for segment, keypoint in self.__segment_to_keypoint.items():
                    # TODO: Consider making get_segment_global_pose return w_t_c poses instead of c_t_w ones.
                    keypoint_from_world: Optional[np.ndarray] = self.__vicon.get_segment_global_pose(subject, segment)
                    if keypoint_from_world is not None:
                        global_keypoint_poses[keypoint] = np.linalg.inv(keypoint_from_world)

                # Compute the poses for the other relevant keypoints that don't have a corresponding Vicon segment.
                midhip_orienter: Optional[KeypointOrienter] = KeypointOrienter.try_make(
                    keypoints, "MidHip", "Neck", ("RHip", "LHip", "Neck")
                )
                if midhip_orienter is not None:
                    global_keypoint_poses["MidHip"] = midhip_orienter.global_pose

                neck_orienter: Optional[KeypointOrienter] = KeypointOrienter.try_make(
                    keypoints, "Neck", "MidHip", ("LShoulder", "RShoulder", "MidHip")
                )
                if neck_orienter is not None:
                    global_keypoint_poses["Neck"] = neck_orienter.global_pose

                # Compute local rotations for the relevant keypoints based on the global poses.
                local_keypoint_rotations: Dict[str, np.ndarray] = KeypointUtil.compute_local_keypoint_rotations(
                    global_keypoint_poses=global_keypoint_poses,
                    midhip_from_rests=self.__midhip_from_rests,
                    parent_keypoints=self.__parent_keypoints
                )

                # Add the skeleton to the dictionary.
                skeletons[subject] = Skeleton3D(
                    keypoints, self.__keypoint_pairs, global_keypoint_poses, local_keypoint_rotations
                )

            # Otherwise, if we're computing our own joint poses:
            else:
                # Simply add the skeleton to the dictionary, and let the joint poses be computed internally.
                skeletons[subject] = Skeleton3D(keypoints, self.__keypoint_pairs)

        return skeletons

    # PRIVATE STATIC METHODS

    @staticmethod
    def __try_add_keypoint(keypoint_name: str, base_marker_sets: List[List[str]],
                           marker_positions: Dict[str, np.ndarray],
                           keypoints: Dict[str, Keypoint]) -> None:
        """
        Try to add a keypoint whose position is the result of averaging the positions of several Vicon markers.

        :param keypoint_name:       The name of the keypoint to try to add.
        :param base_marker_sets:    A list containing different sets of markers whose positions can be averaged to
                                    compute the position of the keypoint (these will be tried in order).
        :param marker_positions:    The positions of the Vicon markers that have been detected.
        :param keypoints:           The list of keypoints for the skeletion (any new keypoint will be added to this).
        """
        # For each possible set of base markers:
        for base_marker_set in base_marker_sets:
            # Try to get the markers' positions.
            base_marker_positions: List[Optional[np.ndarray]] = [marker_positions.get(m) for m in base_marker_set]

            # If all of them are available:
            if all([p is not None for p in base_marker_positions]):
                # Average them to get the position of the keypoint, then add the keypoint to the list and return.
                keypoints[keypoint_name] = Keypoint(keypoint_name, np.mean(base_marker_positions, axis=0))
                return

    @staticmethod
    def __try_hallucinate_missing_markers(marker_positions: Dict[str, np.ndarray]) -> None:
        """
        Try to hallucinate some of the missing markers.

        :param marker_positions:    The positions of the markers detected by the Vicon system.
        """
        ViconSkeletonDetector.__try_hallucinate_trapezium_marker("LASI", marker_positions, "LPSI", "RPSI", "RASI")
        ViconSkeletonDetector.__try_hallucinate_trapezium_marker("LPSI", marker_positions, "LASI", "RASI", "RPSI")
        ViconSkeletonDetector.__try_hallucinate_trapezium_marker("RASI", marker_positions, "RPSI", "LPSI", "LASI")
        ViconSkeletonDetector.__try_hallucinate_trapezium_marker("RPSI", marker_positions, "RASI", "LASI", "LPSI")

    @staticmethod
    def __try_hallucinate_trapezium_marker(target_name: str, marker_positions: Dict[str, np.ndarray],
                                           origin_name: str, base_name: str, diagonal_name: str) -> None:
        """
        Try to hallucinate a missing marker using a trapezium approach.

        .. note::
            The approach assumes that the missing marker is part of a planar trapezium of markers, and that the
            positions of the other three markers in the trapezium are known. This is particularly relevant to
            the Vicon skeleton, which has configurations of four markers around the waist and the head that are
            both roughly planar trapeziums.
        .. note::
            The configuration of markers is (cue ASCII art):

                origin -- base
                 /         \
              target -- diagonal

        :param target_name:         The name of the target marker (the one we're trying to hallucinate).
        :param marker_positions:    The positions of all detected (or previously hallucinated) markers.
        :param origin_name:         The name of the origin marker.
        :param base_name:           The name of the base marker.
        :param diagonal_name:       The name of the diagonal marker.
        """
        origin_pos: Optional[np.ndarray] = marker_positions.get(origin_name)
        base_pos: Optional[np.ndarray] = marker_positions.get(base_name)
        diagonal_pos: Optional[np.ndarray] = marker_positions.get(diagonal_name)

        # If the positions of the origin, base and diagonal markers are all known, but that of the target isn't:
        if all([p is not None for p in [origin_pos, base_pos, diagonal_pos]]) \
                and marker_positions.get(target_name) is None:
            # Calculate a position for the target and add it to the map.
            a: np.ndarray = diagonal_pos - origin_pos
            b: np.ndarray = base_pos - origin_pos
            a_par: np.ndarray = vg.project(a, b)
            a_perp: np.ndarray = a - a_par
            marker_positions[target_name] = origin_pos + b + a_perp - a_par
