import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class ViconInterface(ABC):
    """The interface to a Vicon system."""

    # PUBLIC ABSTRACT METHODS

    @abstractmethod
    def get_frame(self) -> bool:
        """
        Try to get the latest frame of data from the system.

        :return:    True, if the latest frame of data was successfully obtained, or False otherwise.
        """
        pass

    @abstractmethod
    def get_frame_number(self) -> Optional[int]:
        """
        Try to get the frame number of the latest frame of data from the system.

        :return:    The frame number of the latest frame of data from the system, if possible, or None otherwise.
        """
        pass

    @abstractmethod
    def get_marker_positions(self, subject_name: str) -> Dict[str, np.ndarray]:
        """
        Try to get the latest positions of the markers for the subject with the specified name.

        :param subject_name:    The name of the subject.
        :return:                The latest positions of the markers for the subject (indexed by name), if possible,
                                or the empty dictionary otherwise.
        """
        pass

    @abstractmethod
    def get_segment_names(self, subject_name: str) -> List[str]:
        """
        Try to get the names of all of the segments for the specified subject.

        :param subject_name:    The name of the subject.
        :return:                The names of all of the segments for the specified subject, if possible, or the
                                empty list otherwise.
        """
        pass

    @abstractmethod
    def get_segment_pose(self, subject_name: str, segment_name) -> Optional[np.ndarray]:
        """
        Try to get the current 6D pose of the specified segment for the specified subject.

        :param subject_name:    The name of the subject.
        :param segment_name:    The name of the segment.
        :return:                The current 6D pose of the segment, if possible, or None otherwise.
        """
        pass

    @abstractmethod
    def get_subject_names(self) -> List[str]:
        """
        Try to get the names of all of the subjects that are present in the data stream from the system.

        :return:    The names of all of the subjects that are present in the data stream from the system, if possible,
                    or the empty list otherwise.
        """
        pass

    @abstractmethod
    def terminate(self) -> None:
        """Destroy the Vicon interface."""
        pass
