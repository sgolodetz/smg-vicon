import numpy as np

from typing import Callable, Dict, List, Optional, Tuple

from smg.skeletons import Keypoint, KeypointOrienter, KeypointUtil, Skeleton3D

from .vicon_interface import ViconInterface


class ViconSkeletonDetector:
    """A 3D skeleton detector based on a Vicon system."""

    # CONSTRUCTOR

    def __init__(self, vicon: ViconInterface, *, is_person: Callable[[str], bool]):
        """
        Construct a 3D skeleton detector based on a Vicon system.

        :param vicon:       The Vicon interface.
        :param is_person:   A function that determines whether or not the specified subject is a person.
        """
        self.__vicon: ViconInterface = vicon
        self.__is_person: Callable[[str], bool] = is_person

        # Specify which keypoints are joined to form bones.
        self.__keypoint_pairs: List[Tuple[str, str]] = [
            ("Head", "Neck"),
            ("LAnkle", "LKnee"),
            ("LElbow", "LShoulder"),
            ("LElbow", "LWrist"),
            ("LHip", "MidHip"),
            ("LKnee", "LHip"),
            ("LShoulder", "Neck"),
            ("MidHip", "Neck"),
            ("MidHip", "RHip"),
            ("Neck", "RShoulder"),
            ("RAnkle", "RKnee"),
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
            "LTHI": "LThigh",
            "LTIB": "LTibula",
            "LTOE": "LToe",
            "RANK": "RAnkle",
            "RELB": "RElbow",
            "RKNE": "RKnee",
            "RSHO": "RShoulder",
            "RTHI": "RThigh",
            "RTIB": "RTibula",
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

    def detect_skeletons(self) -> List[Skeleton3D]:
        """
        Detect 3D skeletons in the scene using the Vicon system.

        :return:    The detected 3D skeletons.
        """
        skeletons: List[Skeleton3D] = []

        # For each relevant Vicon subject:
        for subject in self.__vicon.get_subject_names():
            # If the subject is not a person, skip it.
            if not self.__is_person(subject):
                continue

            # Otherwise, get its marker positions.
            marker_positions: Dict[str, np.ndarray] = self.__vicon.get_marker_positions(subject)

            # Construct the keypoints for a skeleton based on the available marker positions.
            keypoints: Dict[str, Keypoint] = {}

            for marker_name, keypoint_name in self.__marker_to_keypoint.items():
                marker_position: Optional[np.ndarray] = marker_positions.get(marker_name)
                if marker_position is not None:
                    keypoints[keypoint_name] = Keypoint(keypoint_name, marker_position)

            ViconSkeletonDetector.__try_add_keypoint(
                "Head", [["LBHD", "LFHD", "RBHD", "RFHD"]], keypoints, marker_positions
            )
            ViconSkeletonDetector.__try_add_keypoint("LHip", [["LASI", "LPSI"]], keypoints, marker_positions)
            ViconSkeletonDetector.__try_add_keypoint(
                "LWrist", [["LWRA", "LWRB"], ["LWRA"], ["LWRB"], ["LFIN"]], keypoints, marker_positions
            )
            ViconSkeletonDetector.__try_add_keypoint(
                "MidHip", [["LASI", "LPSI", "RASI", "RPSI"], ["LASI", "RPSI"], ["RASI", "LPSI"]],
                keypoints, marker_positions
            )
            ViconSkeletonDetector.__try_add_keypoint("Neck", [["LSHO", "RSHO"]], keypoints, marker_positions)
            ViconSkeletonDetector.__try_add_keypoint("RHip", [["RASI", "RPSI"]], keypoints, marker_positions)
            ViconSkeletonDetector.__try_add_keypoint(
                "RWrist", [["RWRA", "RWRB"], ["RWRA"], ["RWRB"], ["RFIN"]], keypoints, marker_positions
            )

            global_keypoint_poses: Dict[str, np.ndarray] = {}

            for segment, keypoint_name in self.__segment_to_keypoint.items():
                # TODO: Consider making get_segment_pose return w_t_c poses instead of c_t_w ones.
                keypoint_from_world: Optional[np.ndarray] = self.__vicon.get_segment_global_pose(subject, segment)
                if keypoint_from_world is not None:
                    global_keypoint_poses[keypoint_name] = np.linalg.inv(keypoint_from_world)

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

            local_keypoint_rotations: Dict[str, np.ndarray] = KeypointUtil.compute_local_keypoint_rotations(
                global_keypoint_poses=global_keypoint_poses,
                midhip_from_rests=self.__midhip_from_rests,
                parent_keypoints=self.__parent_keypoints
            )

            # Add the skeleton to the list.
            # skeletons.append(Skeleton3D(keypoints, self.__keypoint_pairs))
            skeletons.append(Skeleton3D(
                keypoints, self.__keypoint_pairs, global_keypoint_poses, local_keypoint_rotations
            ))

        return skeletons

    # PRIVATE STATIC METHODS

    @staticmethod
    def __try_add_keypoint(keypoint_name: str, base_marker_sets: List[List[str]],
                           keypoints: Dict[str, Keypoint], marker_positions: Dict[str, np.ndarray]) -> None:
        for base_marker_set in base_marker_sets:
            base_marker_positions: List[Optional[np.ndarray]] = [marker_positions.get(m) for m in base_marker_set]
            if all([p is not None for p in base_marker_positions]):
                keypoints[keypoint_name] = Keypoint(keypoint_name, np.mean(base_marker_positions, axis=0))
                return
