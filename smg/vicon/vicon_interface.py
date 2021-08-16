import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .subject_from_source_cache import SubjectFromSourceCache


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
    def get_segment_global_pose(self, subject_name: str, segment_name: str) -> Optional[np.ndarray]:
        """
        Try to get the current global 6D pose of the specified segment for the specified subject.

        :param subject_name:    The name of the subject.
        :param segment_name:    The name of the segment.
        :return:                The current global 6D pose of the segment, if possible, or None otherwise.
        """
        pass

    @abstractmethod
    def get_segment_local_rotation(self, subject_name: str, segment_name: str) -> Optional[np.ndarray]:
        """
        Try to get the current local rotation matrix of the specified segment for the specified subject.

        :param subject_name:    The name of the subject.
        :param segment_name:    The name of the segment.
        :return:                The current local rotation matrix of the segment, if possible, or None otherwise.
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

    # PUBLIC METHODS

    def get_image_source_pose(self, subject_name: str, subject_from_source_cache: SubjectFromSourceCache) \
            -> Optional[np.ndarray]:
        """
        Try to get the current 6D pose of the image source associated with the specified subject.

        .. note::
            This will be a transformation from image source space to Vicon space.

        :param subject_name:                The name of the subject.
        :param subject_from_source_cache:   A cache of the transformations from image sources to their Vicon subjects.
        :return:                            The current 6D pose of the image source, if possible, or None otherwise.
        """
        subject_from_source: Optional[np.ndarray] = subject_from_source_cache.get(subject_name)
        subject_from_vicon: Optional[np.ndarray] = self.get_segment_global_pose(subject_name, subject_name)
        if subject_from_source is not None and subject_from_vicon is not None:
            return np.linalg.inv(subject_from_vicon) @ subject_from_source
        else:
            return None
