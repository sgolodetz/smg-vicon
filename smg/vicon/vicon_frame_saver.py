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
            segments: Dict[str, Optional[np.ndarray]] = {}
            for segment_name in segment_names:
                segment_pose: Optional[np.ndarray] = self.__vicon.get_segment_pose(subject_name, segment_name)
                if segment_pose is not None:
                    segment_pose = segment_pose.ravel()
                segments[segment_name] = segment_pose

            output += "Segment Poses: "
            with np.printoptions(linewidth=np.inf):
                output += repr(segments)
            output += "\n\n"

        filename: str = os.path.join(self.__folder, f"{self.__vicon.get_frame_number()}.txt")
        print(filename)

        with open(filename, "w") as f:
            f.write(output)
