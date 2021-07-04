import numpy as np

from typing import Callable, Dict, List, Optional, Tuple

from smg.skeletons import Keypoint, Skeleton3D

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

        # Note: These markers directly correspond to useful keypoints, whereas some other keypoints (e.g. MidHip)
        #       have to be computed based on the positions of multiple markers (see the detect_skeletons function).
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

        self.__segment_to_keypoint: Dict[str, str] = {
            "L_Elbow": "LElbow",
            "R_Elbow": "RElbow",
            # "L_Femur": "LHip",
            # "R_Femur": "RHip"
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

            global_keypoint_poses: Dict[str, np.ndarray] = {"MidHip": np.eye(4)}
            local_keypoint_rotations: Dict[str, np.ndarray] = {}

            for segment, keypoint_name in self.__segment_to_keypoint.items():
                local_keypoint_rotation: Optional[np.ndarray] = self.__vicon.get_segment_local_rotation(subject, segment)
                if local_keypoint_rotation is not None:
                    local_keypoint_rotations[keypoint_name] = local_keypoint_rotation

            # Add the skeleton to the list.
            skeletons.append(Skeleton3D(keypoints, self.__keypoint_pairs))
            # skeletons.append(Skeleton3D(
            #     keypoints, self.__keypoint_pairs, global_keypoint_poses, local_keypoint_rotations
            # ))

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
