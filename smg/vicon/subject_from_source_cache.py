import numpy as np
import os

from typing import Dict, Optional

from smg.utility import PoseUtil


class SubjectFromSourceCache:
    """A cache of the transformations from image sources to their Vicon subjects."""

    # CONSTRUCTOR

    def __init__(self, folder: str):
        """
        Construct a subject-from-source cache.

        .. note::
            The transformations are originally calculated offline by a separate script and saved to disk.

        :param folder:  A folder containing the files from which to load the transformations.
        """
        self.__folder: str = folder
        self.__subjects_from_sources: Dict[str, np.ndarray] = {}

    # PUBLIC METHODS

    def get(self, subject_name: str) -> Optional[np.ndarray]:
        """
        Try to get the subject-from-source transformation for the specified Vicon subject.

        :param subject_name:    The name of a Vicon subject.
        :return:                The subject-from-source transformation for the Vicon subject, if available,
                                or None otherwise.
        """
        # Try to find the transformation for the subject in the cache.
        subject_from_source: Optional[np.ndarray] = self.__subjects_from_sources.get(subject_name)

        # If the transformation's present in the cache, return it.
        if subject_from_source is not None:
            return subject_from_source

        # Otherwise, if the transformation is not in the cache:
        else:
            # Look for the file containing the transformation on disk.
            filename: str = os.path.join(self.__folder, f"subject_from_source-{subject_name}.txt")

            # If the file exists:
            if os.path.exists(filename):
                # Load in the transformation.
                subject_from_source = PoseUtil.load_pose(filename)

                # Store it in the cache for future reference, and then return it.
                self.__subjects_from_sources[subject_name] = subject_from_source
                return subject_from_source

            # Otherwise, if the file containing the transformation doesn't exist, return None.
            else:
                return None
