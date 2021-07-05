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
            "L_Humerus": "LShoulder",
            "R_Humerus": "RShoulder",
            # "L_Femur": "LHip",
            # "R_Femur": "RHip"
        }

        rm = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        lm = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

        self.__midhip_from_rests: Dict[str, np.ndarray] = {
            "LElbow": lm,  # FIXME
            "LShoulder": lm,  # FIXME
            "MidHip": np.eye(3),
            "Neck": np.eye(3),  # FIXME
            "RElbow": rm,  # FIXME
            "RShoulder": rm  # FIXME
        }

        self.__keypoint_parents: Dict[str, str] = {
            "LElbow": "LShoulder",
            "LShoulder": "Neck",
            "Neck": "MidHip",
            "RElbow": "RShoulder",
            "RShoulder": "Neck"
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
                keypoint_from_world: Optional[np.ndarray] = self.__vicon.get_segment_pose(subject, segment)
                if keypoint_from_world is not None:
                    global_keypoint_poses[keypoint_name] = np.linalg.inv(keypoint_from_world)

            global_keypoint_poses["MidHip"] = np.eye(4)
            global_keypoint_poses["MidHip"][0:3, 3] = keypoints["MidHip"].position
            # global_keypoint_poses["MidHip"] = np.linalg.inv(self.__vicon.get_segment_pose(subject, "Root"))
            global_keypoint_poses["Neck"] = np.eye(4)
            global_keypoint_poses["Neck"][0:3, 3] = keypoints["Neck"].position

            local_keypoint_rotations: Dict[str, np.ndarray] = Skeleton3D.compute_local_keypoint_rotations(
                global_keypoint_poses=global_keypoint_poses,
                keypoint_parents=self.__keypoint_parents,
                midhip_from_rests=self.__midhip_from_rests
            )

            # Add the skeleton to the list.
            skeletons.append(Skeleton3D(keypoints, self.__keypoint_pairs))
            # skeletons.append(Skeleton3D(
            #     keypoints, self.__keypoint_pairs, global_keypoint_poses, local_keypoint_rotations
            # ))

        return skeletons

    # PRIVATE STATIC METHODS

    # @staticmethod
    # def __compute_global_keypoint_pose(keypoints: Dict[str, Keypoint], keypoint_name: str, other_keypoint_name: str,
    #                                    triangle_keypoint_names: List[str]) -> Optional[np.ndarray]:
    #     other_keypoint: Optional[Keypoint] = keypoints.get(other_keypoint_name)
    #     if other_keypoint is None:
    #         return None
    #
    #     vs: List[np.ndarray] = []
    #     for i in range(3):
    #         pose: Optional[np.ndarray] = keypoints.get(triangle_keypoint_names[i])
    #         if pose is not None:
    #             vs.append(pose[0:3, 3])
    #
    #     import vg
    #     y = vg.normalize(other_keypoint_pose[0:3, 3] - )

    @staticmethod
    def __try_add_keypoint(keypoint_name: str, base_marker_sets: List[List[str]],
                           keypoints: Dict[str, Keypoint], marker_positions: Dict[str, np.ndarray]) -> None:
        for base_marker_set in base_marker_sets:
            base_marker_positions: List[Optional[np.ndarray]] = [marker_positions.get(m) for m in base_marker_set]
            if all([p is not None for p in base_marker_positions]):
                keypoints[keypoint_name] = Keypoint(keypoint_name, np.mean(base_marker_positions, axis=0))
                return
