import numpy as np
import os

from typing import Dict, List, Optional

from .vicon_interface import ViconInterface


class ViconFrameSaver:
    """Used to save frames of Vicon data to a folder on disk."""

    # CONSTRUCTOR

    def __init__(self, *, folder: str, vicon: ViconInterface):
        """
        Construct a Vicon frame saver.

        :param folder:  The folder on disk to which to save Vicon frames.
        :param vicon:   The Vicon interface.
        """
        self.__folder: str = folder
        self.__vicon: ViconInterface = vicon

        # Make sure the folder exists.
        os.makedirs(folder, exist_ok=True)

    # PUBLIC METHODS

    def save_frame(self) -> None:
        """Save the current frame of Vicon data to disk."""
        output: str = ""

        for subject_name in self.__vicon.get_subject_names():
            output += f"Subject: {subject_name}\n"

            output += "Marker Positions: "
            output += repr(self.__vicon.get_marker_positions(subject_name))
            output += "\n"

            segment_names: List[str] = self.__vicon.get_segment_names(subject_name)
            segment_global_poses: Dict[str, Optional[np.ndarray]] = {}
            segment_local_rotations: Dict[str, Optional[np.ndarray]] = {}

            for segment_name in segment_names:
                segment_global_pose: Optional[np.ndarray] = self.__vicon.get_segment_global_pose(
                    subject_name, segment_name
                )
                if segment_global_pose is not None:
                    segment_global_pose = segment_global_pose.ravel()
                segment_global_poses[segment_name] = segment_global_pose

                segment_local_rotation: Optional[np.ndarray] = self.__vicon.get_segment_local_rotation(
                    subject_name, segment_name
                )
                if segment_local_rotation is not None:
                    segment_local_rotation = segment_local_rotation.ravel()
                segment_local_rotations[segment_name] = segment_local_rotation

            # FIXME: Change the string to "Segment Global Poses: ".
            output += "Segment Poses: "
            with np.printoptions(linewidth=np.inf):
                output += repr(segment_global_poses)
            output += "\n"

            output += "Segment Local Rotations: "
            with np.printoptions(linewidth=np.inf):
                output += repr(segment_local_rotations)
            output += "\n\n"

        filename: str = os.path.join(self.__folder, f"{self.__vicon.get_frame_number()}.txt")

        with open(filename, "w") as f:
            f.write(output)
