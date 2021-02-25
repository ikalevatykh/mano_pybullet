"""This module describes the ManoModel."""

import functools
import os
import pickle
import warnings

import numpy as np

from .math_utils import rvec2mat

__all__ = ('ManoModel')


class ManoModel:
    """The helper class to work with a MANO hand model."""

    def __init__(self, left_hand=False):
        """Load the hand model from a pickled file.

        Keyword Arguments:
            left_hand {bool} -- create a left hand myodel (default: {False})
        """
        path = f'$MANO_MODELS_DIR/MANO_{["RIGHT", "LEFT"][left_hand]}.pkl'
        self._model = self._load(os.path.expandvars(path))
        self._is_left_hand = left_hand

    @staticmethod
    @functools.lru_cache(maxsize=2)
    def _load(path):
        """Load the model from disk.

        Arguments:
            path {str} -- path to the pickled model file

        Returns:
            chumpy array -- MANO model
        """
        with open(path, 'rb') as pick_file:
            with warnings.catch_warnings():  # suppress chumpy warnings
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                return pickle.load(pick_file, encoding='latin1')

    @property
    def is_left_hand(self):
        """This is the model of a left hand.

        Returns:
            bool -- left hand flag
        """
        return self._is_left_hand

    @property
    def faces(self):
        """Hand mesh faces indices.

        Returns:
            np.ndarray -- matrix Nf x 3, where Nf - number of faces
        """
        return self._model.get('f')

    @property
    def weights(self):
        """Vertex weights.

        Returns:
            array -- matrix Nv x Nl, where Nv - number of vertices, Nl - number of links
        """
        return self._model.get('weights')

    @property
    def kintree_table(self):
        """Kinematic tree.

        Returns:
            array -- matrix 2 x Nl, where Nl - number of links
        """
        return np.int32(self._model.get('kintree_table'))

    @property
    def shapedirs(self):
        """Shape mapping matrix.

        Returns:
            array -- matrix Nv x 3 x Nb, where Nv - vertices number, Nb - shape coeffs number
        """
        return self._model.get('shapedirs')

    @property
    def posedirs(self):
        """Pose mapping matrix.

        Returns:
            array -- matrix Nv x 3 x ((Nl-1)*9), where Nv - vertices number, Nl - links number
        """
        return self._model.get('posedirs')

    @property
    def link_names(self):
        """Human readable link names.

        Returns:
            list -- list of link names of size Nl, where Nl - number of links
        """
        fingers = ('index', 'middle', 'pinky', 'ring', 'thumb')
        return ['palm'] + ['{}{}'.format(f, i) for f in fingers for i in range(1, 4)]

    @property
    def tip_links(self):
        """Tip link indices.

        Returns:
            list -- list of tip link indices
        """
        return [3, 6, 12, 9, 15]

    def origins(self, betas=None, pose=None, trans=None):
        """Joint origins.

        Keyword Arguments:
            betas {array} -- shape coefficients, vector 1 x 10 (default: {None})
            pose {array} -- hand pose, matrix Nl x 3 (default: {None})
            trans {array} -- translation, vector 1 x 3 (default: {None})

        Returns:
            array -- matrix Nl x 3, where Nl - number of links
        """
        origins = self._model.get('J')
        if betas is not None:
            regressor = self._model.get('J_regressor')
            origins = regressor.dot(self.vertices(betas=betas))
        if pose is not None:
            raise NotImplementedError
        if trans is not None:
            origins = origins + trans
        return origins

    def vertices(self, betas=None, pose=None, trans=None):
        """Hand mesh verticies.

        Keyword Arguments:
            betas {array} -- shape coefficients, vector 1 x 10 (default: {None})
            pose {array} -- hand pose, matrix Nl x 3 (default: {None})
            trans {array} -- translation, vector 1 x 3 (default: {None})

        Returns:
            array -- matrix Nv x 3, where Nv - number of vertices
        """
        vertices = self._model.get('v_template')
        if betas is not None:
            vertices = vertices + np.dot(self.shapedirs, betas)
        if pose is not None:
            pose = np.ravel([rvec2mat(rvec) - np.eye(3) for rvec in pose[1:]])
            vertices = vertices + np.dot(self.posedirs, pose)
            raise NotImplementedError
        if trans is not None:
            vertices = vertices + trans
        return vertices
