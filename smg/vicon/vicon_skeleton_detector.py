import numpy as np

from typing import Dict, List, Optional, Tuple

from smg.skeletons import Keypoint, Skeleton3D

from .vicon_interface import ViconInterface


class ViconSkeletonDetector:
    """A 3D skeleton detector based on a Vicon system."""

    # CONSTRUCTOR

    def __init__(self, vicon: ViconInterface):
        """
        Construct a 3D skeleton detector based on a Vicon system.

        :param vicon:   The Vicon interface.
        """
        self.__vicon: ViconInterface = vicon

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

    # PUBLIC METHODS

    def detect_skeletons(self) -> List[Skeleton3D]:
        """
        Detect 3D skeletons in the scene using the Vicon system.

        :return:    The detected 3D skeletons.
        """
        skeletons: List[Skeleton3D] = []

        # For each relevant Vicon subject:
        for subject in self.__vicon.get_subject_names():
            # FIXME: We need a proper way of identifying the relevant Vicon subjects.
            if subject == "Madhu":
                # Get the marker positions for the subject.
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
                if lasi_pos is not None and lpsi_pos is not None and rasi_pos is not None and rpsi_pos is not None:
                    keypoints["LHip"] = Keypoint("LHip", (lasi_pos + lpsi_pos) / 2)
                    keypoints["MidHip"] = Keypoint("MidHip", (lasi_pos + lpsi_pos + rasi_pos + rpsi_pos) / 4)
                    keypoints["RHip"] = Keypoint("RHip", (rasi_pos + rpsi_pos) / 2)

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

                # Add the skeleton to the list.
                skeletons.append(Skeleton3D(keypoints, self.__keypoint_pairs))

        return skeletons
