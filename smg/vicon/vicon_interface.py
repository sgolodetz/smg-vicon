import numpy as np

from typing import Dict, Optional
from vicon_dssdk import ViconDataStream


class ViconInterface:
    """The interface to a Vicon system."""

    # CONSTRUCTOR

    def __init__(self, host: str):
        """
        Construct a Vicon interface.

        :param host:    The host (IP address:port) on which the Vicon software is running.
        """
        self.__alive: bool = False

        # Construct the Vicon client.
        self.__client: ViconDataStream.Client = ViconDataStream.Client()

        # Try to connect to the Vicon system (this will raise an exception if it fails).
        self.__client.Connect(host)
        print("Connected to the Vicon system")

        # Set up the Vicon client.
        self.__client.EnableMarkerData()
        self.__client.EnableSegmentData()
        self.__client.EnableUnlabeledMarkerData()
        self.__client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)

        self.__alive = True

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

    def terminate(self) -> None:
        """Destroy the Vicon interface."""
        if self.__alive:
            self.__client.DisableMarkerData()
            self.__client.DisableSegmentData()
            self.__client.DisableUnlabeledMarkerData()
            self.__client.Disconnect()
            self.__alive = False

    def try_get_marker_positions(self, subject_name: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Try to get the positions of the markers for the subject with the specified name.

        .. note::
            This may fail if we move out of the range of the cameras or some of the markers are occluded.

        :param subject_name:    The name of the subject.
        :return:                The positions of the markers for the subject, indexed by name, or None if they are
                                temporarily unavailable.
        """
        try:
            result: Dict[str, np.ndarray] = {}

            # If there's no frame currently available, early out.
            if not self.__client.GetFrame():
                return None

            # For each marker that the subject has:
            for marker_name, parent_segment in self.__client.GetMarkerNames(subject_name):
                # Get its position in the Vicon coordinate system (if known), together with its occlusion status.
                trans, occluded = self.__client.GetMarkerGlobalTranslation(subject_name, marker_name)

                # If we can't currently get the position of the marker, skip it.
                if occluded:
                    continue

                # Transform the marker position from the Vicon coordinate system to our one (the Vicon coordinate
                # system is in mm, whereas ours is in metres).
                pos: np.ndarray = np.array([trans[0] / 1000, trans[1] / 1000, trans[2] / 1000])

                # Record the position in the dictionary.
                result[marker_name] = pos

            return result
        except ViconDataStream.DataStreamException as e:
            # If any exceptions are raised, print out what happened, but otherwise suppress them and keep running.
            print(e)
            return None
