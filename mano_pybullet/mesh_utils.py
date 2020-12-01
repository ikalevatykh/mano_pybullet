"""Mesh manipulating uility functions."""

import numpy as np

__all__ = ('filter_mesh', 'save_mesh_obj')


def filter_mesh(vertices, faces, vertex_mask):
    """Get a submesh from a mesh using vertex mask.

    Arguments:
        vertices {array} -- whole mesh vertices
        faces {array} -- whole mesh faces
        vertex_mask {boolean array} -- vertex filter
    """
    index_map = np.cumsum(vertex_mask) - 1
    faces_mask = np.all(vertex_mask[faces], axis=1)
    vertices_sub = vertices[vertex_mask]
    faces_sub = index_map[faces[faces_mask]]
    return vertices_sub, faces_sub


def save_mesh_obj(filename, vertices, faces):
    """Save a mesh as an obj file.

    Arguments:
        filename {str} -- output file name
        vertices {array} -- mesh vertices
        faces {array} -- mesh faces
    """
    with open(filename, 'w') as obj:
        for vert in vertices:
            obj.write('v {:f} {:f} {:f}\n'.format(*vert))
        for face in faces + 1:  # Faces are 1-based, not 0-based in obj files
            obj.write('f {:d} {:d} {:d}\n'.format(*face))
