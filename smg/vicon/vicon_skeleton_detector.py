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

            lasi_pos: Optional[np.ndarray] = marker_positions.get("LASI")
            lpsi_pos: Optional[np.ndarray] = marker_positions.get("LPSI")
            rasi_pos: Optional[np.ndarray] = marker_positions.get("RASI")
            rpsi_pos: Optional[np.ndarray] = marker_positions.get("RPSI")

            if lasi_pos is not None and lpsi_pos is not None:
                keypoints["LHip"] = Keypoint("LHip", (lasi_pos + lpsi_pos) / 2)
            if rasi_pos is not None and rpsi_pos is not None:
                keypoints["RHip"] = Keypoint("RHip", (rasi_pos + rpsi_pos) / 2)

            if lasi_pos is not None and lpsi_pos is not None and rasi_pos is not None and rpsi_pos is not None:
                keypoints["MidHip"] = Keypoint("MidHip", (lasi_pos + lpsi_pos + rasi_pos + rpsi_pos) / 4)
            elif lasi_pos is not None and rpsi_pos is not None:
                keypoints["MidHip"] = Keypoint("MidHip", (lasi_pos + rpsi_pos) / 2)
            elif rasi_pos is not None and lpsi_pos is not None:
                keypoints["MidHip"] = Keypoint("MidHip", (rasi_pos + lpsi_pos) / 2)

            lwra_pos: Optional[np.ndarray] = marker_positions.get("LWRA")
            lwrb_pos: Optional[np.ndarray] = marker_positions.get("LWRB")
            if lwra_pos is not None and lwrb_pos is not None:
                keypoints["LWrist"] = Keypoint("LWrist", (lwra_pos + lwrb_pos) / 2)

            rwra_pos: Optional[np.ndarray] = marker_positions.get("RWRA")
            rwrb_pos: Optional[np.ndarray] = marker_positions.get("RWRB")
            if rwra_pos is not None and rwrb_pos is not None:
                keypoints["RWrist"] = Keypoint("RWrist", (rwra_pos + rwrb_pos) / 2)

            lsho_pos: Optional[np.ndarray] = marker_positions.get("LSHO")
            rsho_pos: Optional[np.ndarray] = marker_positions.get("RSHO")
            if lsho_pos is not None and rsho_pos is not None:
                keypoints["Neck"] = Keypoint("Neck", (lsho_pos + rsho_pos) / 2)

            # global_keypoint_poses: Dict[str, np.ndarray] = {"MidHip": np.eye(4)}
            # local_keypoint_rotations: Dict[str, np.ndarray] = {}
            #
            # from smg.vicon import LiveViconInterface
            # from typing import cast
            # live_vicon: LiveViconInterface = cast(LiveViconInterface, self.__vicon)
            # for segment, keypoint_name in self.__segment_to_keypoint.items():
            #     local_keypoint_rotation: Optional[np.ndarray] = live_vicon.get_segment_local_rotation(subject, segment)
            #     if local_keypoint_rotation is not None:
            #         local_keypoint_rotations[keypoint_name] = local_keypoint_rotation

            # Add the skeleton to the list.
            skeletons.append(Skeleton3D(keypoints, self.__keypoint_pairs))
            # skeletons.append(Skeleton3D(
            #     keypoints, self.__keypoint_pairs, global_keypoint_poses, local_keypoint_rotations
            # ))

        return skeletons
