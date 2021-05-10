import numpy as np

from typing import Dict, Optional
from vicon_dssdk import ViconDataStream


class ViconInterface:
    """The interface to a Vicon system."""

    # CONSTRUCTOR

    def __init__(self, host: str):
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
        if self.__alive:
            self.__client.DisableMarkerData()
            self.__client.DisableSegmentData()
            self.__client.DisableUnlabeledMarkerData()
            self.__client.Disconnect()
            self.__alive = False

    def try_get_marker_positions(self, subject_name: str) -> Optional[Dict[str, np.ndarray]]:
        try:
            if not self.__client.GetFrame():
                return None

            result: Dict[str, np.ndarray] = {}

            for marker_name, parent_segment in self.__client.GetMarkerNames(subject_name):
                # TODO
                trans, occluded = self.__client.GetMarkerGlobalTranslation(subject_name, marker_name)

                # If we can't currently get the position of the marker, skip it.
                if occluded:
                    continue

                # TODO
                pos: np.ndarray = np.array([trans[0] / 1000, trans[1] / 1000, trans[2] / 1000])

                # TODO
                result[marker_name] = pos

            return result
        except ViconDataStream.DataStreamException as e:
            print(e)
            return None
