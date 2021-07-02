import numpy as np

from typing import Dict, List, Optional
from vicon_dssdk import ViconDataStream

from .vicon_interface import ViconInterface


class LiveViconInterface(ViconInterface):
    """The interface to a live Vicon system."""

    # CONSTRUCTOR

    def __init__(self, host: str = "192.168.137.1:801"):
        """
        Construct a live Vicon interface.

        .. note::
            In the Wytham Flight Lab, the host seems to be "169.254.185.150:801" when connecting via Ethernet,
            and "192.168.137.1:801" when connecting via the WiFi hotspot. I've set the default to the WiFi one,
            since that's most useful on a day-to-day basis.

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

    def get_frame(self) -> bool:
        """
        Try to get the latest frame of data from the system.

        :return:    True, if the latest frame of data was successfully obtained, or False otherwise.
        """
        try:
            return self.__client.GetFrame()
        except ViconDataStream.DataStreamException as e:
            # If any exceptions are raised, print out what happened, but otherwise suppress them and keep running.
            print(e)
            return False

    def get_frame_number(self) -> Optional[int]:
        """
        Try to get the frame number of the latest frame of data from the system.

        :return:    The frame number of the latest frame of data from the system, if possible, or None otherwise.
        """
        try:
            return self.__client.GetFrameNumber()
        except ViconDataStream.DataStreamException as e:
            # If any exceptions are raised, print out what happened, but otherwise suppress them and keep running.
            print(e)
            return None

    def get_marker_positions(self, subject_name: str) -> Dict[str, np.ndarray]:
        """
        Try to get the latest positions of the markers for the subject with the specified name.

        :param subject_name:    The name of the subject.
        :return:                The latest positions of the markers for the subject (indexed by name), if possible,
                                or the empty dictionary otherwise.
        """
        try:
            result: Dict[str, np.ndarray] = {}

            # For each marker that the subject has:
            for marker_name, parent_segment in self.__client.GetMarkerNames(subject_name):
                # Get its position in the Vicon coordinate system (if known), together with its occlusion status.
                trans, occluded = self.__client.GetMarkerGlobalTranslation(subject_name, marker_name)

                # If we can't currently get the position of the marker, skip it.
                if occluded:
                    continue

                # Record the position in the dictionary.
                result[marker_name] = LiveViconInterface.__from_vicon_position(trans)

            return result
        except ViconDataStream.DataStreamException as e:
            # If any exceptions are raised, print out what happened, but otherwise suppress them and keep running.
            print(e)
            return {}

    def get_segment_local_rotation(self, subject_name: str, segment_name: str) -> Optional[np.ndarray]:
        try:
            rot, occluded = self.__client.GetSegmentLocalRotationMatrix(subject_name, segment_name)
            return np.array(rot) if not occluded else None
        except ViconDataStream.DataStreamException as e:
            # If any exceptions are raised, print out what happened, but otherwise suppress them and keep running.
            print(e)
            return None

    def get_segment_names(self, subject_name: str) -> List[str]:
        """
        Try to get the names of all of the segments for the specified subject.

        :param subject_name:    The name of the subject.
        :return:                The names of all of the segments for the specified subject, if possible, or the
                                empty list otherwise.
        """
        try:
            return self.__client.GetSegmentNames(subject_name)
        except ViconDataStream.DataStreamException as e:
            # If any exceptions are raised, print out what happened, but otherwise suppress them and keep running.
            print(e)
            return []

    def get_segment_pose(self, subject_name: str, segment_name: str) -> Optional[np.ndarray]:
        """
        Try to get the current 6D pose of the specified segment for the specified subject.

        :param subject_name:    The name of the subject.
        :param segment_name:    The name of the segment.
        :return:                The current 6D pose of the segment, if possible, or None otherwise.
        """
        try:
            world_from_camera: np.ndarray = np.eye(4)

            trans, occluded = self.__client.GetSegmentGlobalTranslation(subject_name, segment_name)
            if occluded:
                return None
            else:
                world_from_camera[0:3, 3] = LiveViconInterface.__from_vicon_position(trans)

            rot, occluded = self.__client.GetSegmentGlobalRotationMatrix(subject_name, segment_name)
            if occluded:
                return None
            else:
                world_from_camera[0:3, 0:3] = rot

            return np.linalg.inv(world_from_camera)
        except ViconDataStream.DataStreamException as e:
            # If any exceptions are raised, print out what happened, but otherwise suppress them and keep running.
            print(e)
            return None

    def get_subject_names(self) -> List[str]:
        """
        Try to get the names of all of the subjects that are present in the data stream from the Vicon.

        :return:    The names of all of the subjects that are present in the data stream from the Vicon, if possible,
                    or the empty list otherwise.
        """
        try:
            return self.__client.GetSubjectNames()
        except ViconDataStream.DataStreamException as e:
            # If any exceptions are raised, print out what happened, but otherwise suppress them and keep running.
            print(e)
            return []

    def terminate(self) -> None:
        """Destroy the Vicon interface."""
        if self.__alive:
            self.__client.DisableMarkerData()
            self.__client.DisableSegmentData()
            self.__client.DisableUnlabeledMarkerData()
            self.__client.Disconnect()
            self.__alive = False

    # PRIVATE STATIC METHODS

    @staticmethod
    def __from_vicon_position(pos) -> np.ndarray:
        """
        Transform a position from the Vicon coordinate system to our one.

        .. note::
            The Vicon coordinate system is in mm, whereas ours is in metres.

        :param pos: The position in the Vicon coordinate system.
        :return:    The equivalent position in our coordinate system.
        """
        return np.array(pos) / 1000
