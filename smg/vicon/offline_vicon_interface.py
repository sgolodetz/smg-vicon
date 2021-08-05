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

        def __init__(self, marker_positions: Dict[str, np.ndarray],
                     segment_global_poses: Dict[str, Optional[np.ndarray]],
                     segment_local_rotations: Dict[str, Optional[np.ndarray]]):
            """
            Construct a Vicon subject.

            :param marker_positions:        The positions of the subject's markers.
            :param segment_global_poses:    The global 6D poses of the subject's segments (if known).
            :param segment_local_rotations: The local rotation matrices of the subject's segments (if known).
            """
            self.__marker_positions: Dict[str, np.ndarray] = marker_positions
            self.__segment_global_poses: Dict[str, Optional[np.ndarray]] = segment_global_poses
            self.__segment_local_rotations: Dict[str, Optional[np.ndarray]] = segment_local_rotations

        # PROPERTIES

        @property
        def marker_positions(self) -> Dict[str, np.ndarray]:
            """
            Get the positions of the subject's markers.

            :return:    The positions of the subject's markers.
            """
            return self.__marker_positions

        @property
        def segment_global_poses(self) -> Dict[str, Optional[np.ndarray]]:
            """
            Get the global 6D poses of the subject's segments (if known).

            .. note::
                Some or all of these can be None if they're unknown. However, the dictionary will in any case
                contain an entry for each segment the subject has.

            :return:    The global 6D poses of the subject's segments.
            """
            return self.__segment_global_poses

        @property
        def segment_local_rotations(self) -> Dict[str, Optional[np.ndarray]]:
            """
            Get the local rotation matrices of the subject's segments (if known).

            .. note::
                Some or all of these can be None if they're unknown. However, the dictionary will in any case
                contain an entry for each segment the subject has.

            :return:    The local rotation matrices of the subject's segments.
            """
            return self.__segment_local_rotations

    # CONSTRUCTOR

    def __init__(self, *, folder: str):
        """
        Construct an offline Vicon system.

        :param folder:  A folder on disk that contains saved state from a live Vicon system.
        """
        self.__folder: str = folder

        # The names of the files in the folder on disk that contain saved Vicon frame data, in frame number order.
        frame_filenames: List[str] = [f for f in os.listdir(self.__folder) if f.endswith(".vicon.txt")]
        self.__frame_filenames: List[str] = sorted(
            frame_filenames, key=OfflineViconInterface.__get_frame_number
        )

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
            self.__frame_number = int(frame_filename[:-len(".vicon.txt")])

            # Load the new frame.
            with open(os.path.join(self.__folder, frame_filename)) as f:
                lines: List[str] = f.readlines()

                # Note: The new version of the file format uses four content lines and one blank line per subject,
                #       hence the "5" here. There's also an older version of the file format that didn't save the
                #       local rotations and so has one fewer content line per subject - we handle that below.
                for i in range(0, len(lines), 5):
                    subject_name: str = OfflineViconInterface.__get_line_contents(lines[i])
                    marker_positions: Dict[str, np.ndarray] = eval(
                        OfflineViconInterface.__get_line_contents(lines[i+1]), {'array': np.array}
                    )
                    segment_global_poses: Dict[str, Optional[np.ndarray]] = eval(
                        OfflineViconInterface.__get_line_contents(lines[i+2]),
                        {'array': OfflineViconInterface.__make_pose_matrix}
                    )

                    # If the file contains four content lines for this subject, read in its local rotations.
                    if lines[i+3] != "\n":
                        segment_local_rotations: Dict[str, Optional[np.ndarray]] = eval(
                            OfflineViconInterface.__get_line_contents(lines[i+3]),
                            {'array': OfflineViconInterface.__make_rotation_matrix}
                        )

                    # Otherwise, use an empty map of local rotations, and decrement the line counter to compensate
                    # for the fact that this subject only has three content lines. Yes, this is a nasty hack :)
                    else:
                        segment_local_rotations: Dict[str, Optional[np.ndarray]] = {}
                        i -= 1

                    self.__subjects[subject_name] = OfflineViconInterface.Subject(
                        marker_positions, segment_global_poses, segment_local_rotations
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

    def get_segment_global_pose(self, subject_name: str, segment_name: str) -> Optional[np.ndarray]:
        """
        Try to get the current global 6D pose of the specified segment for the specified subject.

        :param subject_name:    The name of the subject.
        :param segment_name:    The name of the segment.
        :return:                The current global 6D pose of the segment, if possible, or None otherwise.
        """
        subject: Optional[OfflineViconInterface.Subject] = self.__subjects.get(subject_name)
        return subject.segment_global_poses.get(segment_name) if subject is not None else None

    def get_segment_local_rotation(self, subject_name: str, segment_name: str) -> Optional[np.ndarray]:
        """
        Try to get the current local rotation matrix of the specified segment for the specified subject.

        :param subject_name:    The name of the subject.
        :param segment_name:    The name of the segment.
        :return:                The current local rotation matrix of the segment, if possible, or None otherwise.
        """
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
        return list(subject.segment_global_poses.keys()) if subject is not None else None

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
    def __get_frame_number(filename: str) -> int:
        """
        Get the frame number corresponding to a file containing Vicon frame data.

        .. note::
            The files are named <frame number>.txt, so we can get the frame numbers directly from the file names.

        :param filename:    The name of a file containing Vicon frame data.
        :return:            The corresponding frame number.
        """
        frame_number, _, _ = filename.split(".")
        return int(frame_number)

    @staticmethod
    def __get_line_contents(line: str) -> str:
        """
        Get the contents part of a line in one of the files that contains Vicon frame data.

        .. note::
            The files contain multiple lines of the form "specifier: contents\n". This function simply gets the
            contents part of such a line.
        .. note::
            The contents parts of some of the lines themselves contain ": ", which is why we need to specify a
            maxsplit of 1 here.
        .. note::
            The line passed in ends with a "\n", which we strip from the end of the contents before returning.

        :param line:    The line whose contents part we want to get.
        :return:        The contents part of the line.
        """
        _, contents = line.split(": ", maxsplit=1)
        return contents[:-1]

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
        """
        Convert a flat array ot 9 floats in row-major order into a 3*3 rotation matrix.

        :param flat_rot:    A flat array of 9 floats in row-major order.
        :return:            The corresponding 3*3 rotation matrix.
        """
        return np.array(flat_rot).reshape(3, 3)
