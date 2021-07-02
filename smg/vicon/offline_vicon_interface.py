import numpy as np
import os

from typing import Dict, List, Optional

from .vicon_interface import ViconInterface


class OfflineViconInterface(ViconInterface):
    """
    The interface to an offline Vicon system.

    An offline Vicon system simulates a live Vicon system by using saved state that was originally captured live.
    """

    # NESTED TYPES

    class Subject:
        """The offline Vicon system's representation of a Vicon subject."""

        # CONSTRUCTOR

        def __init__(self, marker_positions: Dict[str, np.ndarray], segment_poses: Dict[str, Optional[np.ndarray]],
                     segment_local_rotations: Dict[str, Optional[np.ndarray]]):
            """
            Construct a Vicon subject.

            :param marker_positions:        The positions of the subject's markers.
            :param segment_poses:           The 6D poses of the subject's segments (if known).
            :param segment_local_rotations: TODO
            """
            self.__marker_positions: Dict[str, np.ndarray] = marker_positions
            self.__segment_local_rotations: Dict[str, Optional[np.ndarray]] = segment_local_rotations
            self.__segment_poses: Dict[str, Optional[np.ndarray]] = segment_poses

        # PROPERTIES

        @property
        def marker_positions(self) -> Dict[str, np.ndarray]:
            """
            Get the positions of the subject's markers.

            :return:    The positions of the subject's markers.
            """
            return self.__marker_positions

        @property
        def segment_local_rotations(self) -> Dict[str, Optional[np.ndarray]]:
            return self.__segment_local_rotations

        @property
        def segment_poses(self) -> Dict[str, Optional[np.ndarray]]:
            """
            Get the 6D poses of the subject's segments (if known).

            .. note::
                Some or all of these can be None if they're unknown. However, the dictionary will in any case
                contain an entry for each segment the subject has.

            :return:    The 6D poses of the subject's segments.
            """
            return self.__segment_poses

    # CONSTRUCTOR

    def __init__(self, *, folder: str):
        """
        Construct an offline Vicon system.

        :param folder:  A folder on disk that contains saved state from a live Vicon system.
        """
        self.__folder: str = folder

        # The names of the files in the folder on disk that contain saved Vicon frame data.
        self.__frame_filenames: List[str] = sorted(os.listdir(self.__folder))

        # The number originally assigned to the current frame by the live Vicon system.
        self.__frame_number: Optional[int] = None

        # The index in the frame filenames array of the next frame to load.
        self.__next_frame_idx: int = 0

        # The Vicon subjects present in the current frame.
        self.__subjects: Dict[str, OfflineViconInterface.Subject] = {}

    # DESTRUCTOR

    def __del__(self):
        """Destroy the Vicon interface."""
        self.terminate()

    # SPECIAL METHODS

    def __enter__(self):
        """No-op (needed to allow the interface's lifetime to be managed by a with statement)."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Destroy the interface at the end of the with statement that's used to manage its lifetime."""
        self.terminate()

    # PUBLIC METHODS

    def get_frame(self) -> bool:
        """
        Try to get the latest frame of data from the system.

        .. note::
            For the offline Vicon system, this means loading the next frame from disk.

        :return:    True, if the latest frame of data was successfully obtained, or False otherwise.
        """
        # Clear the existing dictionary of subjects ready for the new frame (if any).
        self.__subjects = {}

        # If there are still frames on disk that we haven't looked at:
        if self.__next_frame_idx < len(self.__frame_filenames):
            # Get the name of the next frame in the sequence.
            frame_filename: str = self.__frame_filenames[self.__next_frame_idx]

            # Get the number of the new frame from the filename.
            self.__frame_number = int(frame_filename[:-4])

            # Load the new frame.
            with open(os.path.join(self.__folder, frame_filename)) as f:
                lines: List[str] = f.readlines()

                # Note: The file consists of four content lines and one blank line per subject, hence the "5" here.
                for i in range(0, len(lines), 5):
                    subject_name: str = lines[i][len("Subject: "):-1]
                    marker_positions: Dict[str, np.ndarray] = eval(
                        lines[i+1][len("Marker Positions: "):-1], {'array': np.array}
                    )
                    segment_poses: Dict[str, Optional[np.ndarray]] = eval(
                        lines[i+2][len("Segment Poses: "):-1], {'array': OfflineViconInterface.__make_pose_matrix}
                    )
                    segment_local_rotations: Dict[str, Optional[np.ndarray]] = eval(
                        lines[i+3][len("Segment Local Rotations: "):-1], {'array': OfflineViconInterface.__make_rotation_matrix}
                    )

                    self.__subjects[subject_name] = OfflineViconInterface.Subject(
                        marker_positions, segment_poses, segment_local_rotations
                    )

            # Advance the frame index.
            self.__next_frame_idx += 1

            return True

        # Otherwise, clear the frame number and signal to the caller that there are no more frames.
        else:
            self.__frame_number = None
            return False

    def get_frame_number(self) -> Optional[int]:
        """
        Try to get the frame number of the latest frame of data from the system.

        :return:    The frame number of the latest frame of data from the system, if possible, or None otherwise.
        """
        return self.__frame_number

    def get_marker_positions(self, subject_name: str) -> Dict[str, np.ndarray]:
        """
        Try to get the latest positions of the markers for the subject with the specified name.

        :param subject_name:    The name of the subject.
        :return:                The latest positions of the markers for the subject (indexed by name), if possible,
                                or the empty dictionary otherwise.
        """
        subject: Optional[OfflineViconInterface.Subject] = self.__subjects.get(subject_name)
        return subject.marker_positions if subject is not None else {}

    def get_segment_local_rotation(self, subject_name: str, segment_name: str) -> Optional[np.ndarray]:
        subject: Optional[OfflineViconInterface.Subject] = self.__subjects.get(subject_name)
        return subject.segment_local_rotations.get(segment_name) if subject is not None else None

    def get_segment_names(self, subject_name: str) -> List[str]:
        """
        Try to get the names of all of the segments for the specified subject.

        :param subject_name:    The name of the subject.
        :return:                The names of all of the segments for the specified subject, if possible, or the
                                empty list otherwise.
        """
        subject: Optional[OfflineViconInterface.Subject] = self.__subjects.get(subject_name)
        return list(subject.segment_poses.keys()) if subject is not None else None

    def get_segment_pose(self, subject_name: str, segment_name: str) -> Optional[np.ndarray]:
        """
        Try to get the current 6D pose of the specified segment for the specified subject.

        :param subject_name:    The name of the subject.
        :param segment_name:    The name of the segment.
        :return:                The current 6D pose of the segment, if possible, or None otherwise.
        """
        subject: Optional[OfflineViconInterface.Subject] = self.__subjects.get(subject_name)
        return subject.segment_poses.get(segment_name) if subject is not None else None

    def get_subject_names(self) -> List[str]:
        """
        Try to get the names of all of the subjects that are present in the data stream from the system.

        :return:    The names of all of the subjects that are present in the data stream from the system, if possible,
                    or the empty list otherwise.
        """
        return list(self.__subjects.keys())

    def terminate(self) -> None:
        """Destroy the Vicon interface."""
        # Note: No cleanup is needed for the offline Vicon system, so this is a no-op.
        pass

    # PRIVATE STATIC METHODS

    @staticmethod
    def __make_pose_matrix(flat_pose: List[float]) -> np.ndarray:
        """
        Convert a flat array of 16 floats in row-major order into a 4*4 pose matrix.

        :param flat_pose:   A flat array of 16 floats in row-major order.
        :return:            The corresponding 4*4 pose matrix.
        """
        return np.array(flat_pose).reshape(4, 4)

    @staticmethod
    def __make_rotation_matrix(flat_rot: List[float]) -> np.ndarray:
        return np.array(flat_rot).reshape(3, 3)
