import numpy as np
import open3d as o3d
import os

from typing import Dict

from smg.meshing import MeshUtil
from smg.opengl import OpenGLLightingContext, OpenGLTriMesh
from smg.utility import FiducialUtil, GeometryUtil

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

    @staticmethod
    def load_scene_mesh(scenes_folder: str, scene_timestamp: str, vicon: ViconInterface) -> OpenGLTriMesh:
        """
        Load in a scene mesh, transforming it into the Vicon coordinate system in the process.

        :param scenes_folder:   The folder from which to load the scene mesh.
        :param scene_timestamp: A timestamp indicating which scene mesh to load.
        :param vicon:           The Vicon interface.
        :return:                The scene mesh.
        """
        # Specify the file paths.
        mesh_filename: str = os.path.join(scenes_folder, f"TangoCapture-{scene_timestamp}-cleaned.ply")
        fiducials_filename: str = os.path.join(scenes_folder, f"TangoCapture-{scene_timestamp}-fiducials.txt")

        # Load in the positions of the four ArUco marker corners as estimated during the reconstruction process.
        fiducials: Dict[str, np.ndarray] = FiducialUtil.load_fiducials(fiducials_filename)

        # Stack these positions into a 3x4 matrix.
        p: np.ndarray = np.column_stack([
            fiducials["0_0"],
            fiducials["0_1"],
            fiducials["0_2"],
            fiducials["0_3"]
        ])

        # Look up the Vicon coordinate system positions of the all of the Vicon markers that can currently be seen
        # by the Vicon system, hopefully including ones for the ArUco marker corners.
        marker_positions: Dict[str, np.ndarray] = vicon.get_marker_positions("Registrar")

        # Again, stack the relevant positions into a 3x4 matrix.
        q: np.ndarray = np.column_stack([
            marker_positions["0_0"],
            marker_positions["0_1"],
            marker_positions["0_2"],
            marker_positions["0_3"]
        ])

        # Estimate the rigid transformation between the two sets of points.
        transform: np.ndarray = GeometryUtil.estimate_rigid_transform(p, q)

        # Load in the scene mesh and transform it into the Vicon coordinate system.
        scene_mesh_o3d: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(mesh_filename)
        scene_mesh_o3d.transform(transform)

        # Convert the scene mesh to OpenGL format and return it.
        return MeshUtil.convert_trimesh_to_opengl(scene_mesh_o3d)
