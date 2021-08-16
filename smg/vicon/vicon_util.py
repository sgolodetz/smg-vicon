import numpy as np

from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Optional, Tuple

from smg.opengl import OpenGLLightingContext
from smg.rigging.cameras import SimpleCamera
from smg.rigging.helpers import CameraPoseConverter
from smg.skeletons import Keypoint, Skeleton3D
from smg.utility import GeometryUtil

from .vicon_interface import ViconInterface


class ViconUtil:
    """Utility functions related to the Vicon system."""

    # PUBLIC STATIC METHODS

    @staticmethod
    def compute_subject_designations(vicon: ViconInterface, skeletons: Dict[str, Skeleton3D]) \
            -> Dict[str, List[Tuple[str, float]]]:
        """
        Compute the subject designations for the current Vicon frame.

        .. note::
            The subject designations for the frame consist of a subject -> [(skeleton, distance)] table quantifying
            how closely different skeletons are pointing towards different designatable subjects. The table contains
            a list of (skeleton, distance) pairs for each subject that might be being designated, which is sorted in
            non-decreasing order of distance. By "might be being designated", I mean that the subject's position must
            be known, and there must be at least one skeleton in the scene whose right shoulder and right elbow joint
            positions are known. The distance for a given subject and skeleton is that between the position of the
            subject and the position of the closest point to the subject on a half-ray starting at the skeleton's
            right shoulder and directed through its right elbow.

        :param vicon:       The Vicon interface.
        :param skeletons:   The skeletons that have been detected in the current frame.
        :return:            The subject designations for the current frame.
        """
        designations: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # For each Vicon subject:
        for subject_name in vicon.get_subject_names():
            # If the subject is not designatable, skip it.
            if not ViconUtil.is_designatable(subject_name):
                continue

            # Try to get the subject's position. If that fails, skip the subject.
            subject_from_world: Optional[np.ndarray] = vicon.get_segment_global_pose(subject_name, subject_name)
            if subject_from_world is None:
                continue

            subject_cam: SimpleCamera = CameraPoseConverter.pose_to_camera(subject_from_world)
            subject_pos: np.ndarray = subject_cam.p()

            # For each skeleton that has been detected in the current frame:
            for skeleton_name, skeleton in skeletons.items():
                # Try to get its right shoulder and right elbow keypoints.
                right_shoulder: Optional[Keypoint] = skeleton.keypoints.get("RShoulder")
                right_elbow: Optional[Keypoint] = skeleton.keypoints.get("RElbow")

                # If that's possible (i.e. if they've been successfully detected by the Vicon system):
                if right_shoulder is not None and right_elbow is not None:
                    # Compute and record the designation distance for the current subject and skeleton.
                    right_shoulder_pos: np.ndarray = right_shoulder.position
                    right_elbow_pos: np.ndarray = right_elbow.position
                    closest_point: np.ndarray = GeometryUtil.find_closest_point_on_half_ray(
                        subject_pos, right_shoulder_pos, right_elbow_pos - right_shoulder_pos
                    )
                    designations[subject_name].append((skeleton_name, np.linalg.norm(subject_pos - closest_point)))

            # If there any designations for the current subject, sort them in non-decreasing order of distance:
            if designations.get(subject_name) is not None:
                designations[subject_name] = sorted(designations[subject_name], key=itemgetter(1))

        return designations

    @staticmethod
    def default_lighting_context() -> OpenGLLightingContext:
        """
        Get the default lighting context to use when rendering Vicon scenes.

        :return:    The default lighting context to use when rendering Vicon scenes.
        """
        direction: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0])
        return OpenGLLightingContext({
            0: OpenGLLightingContext.DirectionalLight(direction),
            1: OpenGLLightingContext.DirectionalLight(-direction),
        })

    @staticmethod
    def is_designatable(subject_name: str) -> bool:
        """
        Determine whether or not the specified Vicon subject is designatable.

        :param subject_name:    The name of the subject.
        :return:                True, if the specified Vicon subject is designatable, or False otherwise.
        """
        return subject_name.startswith("Object")

    @staticmethod
    def is_person(subject_name: str, vicon: ViconInterface) -> bool:
        """
        Determine whether or not the specified Vicon subject is a person.

        :param subject_name:    The name of the subject.
        :param vicon:           The Vicon interface.
        :return:                True, if the specified Vicon subject is a person, or False otherwise.
        """
        return "Root" in vicon.get_segment_names(subject_name)
