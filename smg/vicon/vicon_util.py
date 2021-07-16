import numpy as np

from smg.opengl import OpenGLLightingContext

from .vicon_interface import ViconInterface


class ViconUtil:
    """Utility functions related to the Vicon system."""

    # PUBLIC STATIC METHODS

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
    def is_person(subject_name: str, vicon: ViconInterface) -> bool:
        """
        Determine whether or not the specified Vicon subject is a person.

        :param subject_name:    The name of the subject.
        :param vicon:           The Vicon interface.
        :return:                True, if the specified Vicon subject is a person, or False otherwise.
        """
        return "Root" in vicon.get_segment_names(subject_name)
