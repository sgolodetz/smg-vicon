import numpy as np
import os

from typing import Dict, Optional

from smg.utility import PoseUtil


class SubjectFromSourceCache:
    """A cache of the transformations from image sources to their Vicon subjects."""

    # CONSTRUCTOR

    def __init__(self, directory: str):
        """
        Construct a subject-from-source cache.

        :param directory:   A directory containing the files from which to originally load the transformations.
        """
        self.__directory: str = directory
        self.__subjects_from_sources: Dict[str, np.ndarray] = {}

    # PUBLIC METHODS

    def get(self, subject_name: str) -> Optional[np.ndarray]:
        """
        Try to get the subject-from-source transformation for the specified Vicon subject.

        :param subject_name:    The name of a Vicon subject.
        :return:                The subject-from-source transformation for the Vicon subject, if available,
                                or None otherwise.
        """
        subject_from_source: Optional[np.ndarray] = self.__subjects_from_sources.get(subject_name)
        if subject_from_source is not None:
            return subject_from_source
        else:
            filename: str = os.path.join(self.__directory, f"subject_from_source-{subject_name}.txt")
            if os.path.exists(filename):
                subject_from_source = PoseUtil.load_pose(filename)
                self.__subjects_from_sources[subject_name] = subject_from_source
            else:
                return None
