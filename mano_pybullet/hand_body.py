"""Rigid hand body."""
import contextlib
import tempfile

import numpy as np
import pybullet as pb

from .math_utils import mat2pb, pb2mat
from .mesh_utils import filter_mesh, save_mesh_obj

__all__ = ('HandBody')


class HandBody:
    """Rigid multi-link hand body."""

    FLAG_STATIC = 1  # create static (fixed) body
    FLAG_ENABLE_COLLISION_SHAPES = 2  # enable colllision shapes
    FLAG_ENABLE_VISUAL_SHAPES = 4  # enable visual shapes
    FLAG_JOINT_LIMITS = 8  # apply joint limits
    FLAG_DYNAMICS = 16  # overide default dynamics parameters
    FLAG_USE_SELF_COLLISION = 32  # enable self collision
    FLAG_DEFAULT = sum([FLAG_ENABLE_COLLISION_SHAPES,
                        FLAG_ENABLE_VISUAL_SHAPES,
                        FLAG_JOINT_LIMITS,
                        FLAG_DYNAMICS,
                        FLAG_USE_SELF_COLLISION])

    def __init__(self, client, hand_model, flags=FLAG_DEFAULT, shape_betas=None):
        """HandBody constructor.

        Arguments:
            client {PybulletClient} -- pybullet client
            hand_model {HandModel} -- rigid hand model

        Keyword Arguments:
            flags {int} -- configuration flags (default: {FLAG_DEFAULT})
            color {list} -- color RGBA (default: {None})
            shape_betas {array} -- MANO shape beta parameters  (default: {None})
        """
        self._client = client
        self._model = hand_model
        self._flags = flags
        self._vertices = hand_model.vertices(betas=shape_betas)
        self._origin = self._model.origins()[0]
        self._joint_indices = []
        self._joint_limits = []
        self._link_mapping = {}
        self._body_id = self._make_body()
        self._constraint_id = self._make_constraint()
        self._apply_joint_limits()
        self._apply_dynamics()

    @ property
    def body_id(self):
        """Body unique id in the simulator.

        Returns:
            int -- body unique id in PyBullet
        """
        return self._body_id

    @ property
    def joint_indices(self):
        """Articulated joint indices.

        Returns:
            list -- list of joint indices
        """
        return self._joint_indices

    @ property
    def joint_limits(self):
        """Articulated joints angle bounds.

        Returns:
            list -- list of tuples (lower limit, upper limit)
        """
        return self._joint_limits

    def get_state(self):
        """Get current hand state.

        Returns:
            tuple -- base position, orientation, forces, joint positions, velocities, torques
        """
        base_pos, base_orn = self._client.getBasePositionAndOrientation(self._body_id)
        if self._constraint_id != -1:
            constraint_forces = self._client.getConstraintState(self._constraint_id)
        else:
            constraint_forces = [0.0] * 6
        joint_states = self._client.getJointStates(self._body_id, self._joint_indices)
        joints_pos, joints_vel, _, joints_torque = zip(*joint_states)
        return base_pos, base_orn, constraint_forces, joints_pos, joints_vel, joints_torque

    def reset(self, position, orientation, joint_angles=None):
        """Reset base pose and joint angles.

        Arguments:
            position {vec3} -- position
            orientation {vec4} -- quaternion x,y,z,w

        Keyword Arguments:
            joint_angles {list} -- new angles for all articulated joints (default: {None})
        """
        self._client.resetBasePositionAndOrientation(self._body_id, position, orientation)
        if joint_angles is not None:
            for i, angle in zip(self._joint_indices, joint_angles):
                self._client.resetJointState(self._body_id, i, angle)

    def set_target(self, position, orientation, joint_angles=None):
        """Set target base pose and joint angles.

        Arguments:
            position {vec3} -- position
            orientation {vec4} -- quaternion x,y,z,w

        Keyword Arguments:
            joint_angles {list} -- new angles for all articulated joints (default: {None})
        """
        if self._constraint_id != -1:
            self._client.changeConstraint(
                self._constraint_id,
                jointChildPivot=position,
                jointChildFrameOrientation=orientation,
                maxForce=10.0)
        if joint_angles is not None:
            self._client.setJointMotorControlArray(
                bodyUniqueId=self._body_id,
                jointIndices=self._joint_indices,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=joint_angles,
                forces=[0.5]*len(joint_angles))

    def get_mano_state(self):
        """Get current hand state as a MANO model state.

        Returns:
            tuple -- trans, pose
        """
        base_pos, base_orn, _, angles, _, _ = self.get_state()
        basis = pb2mat(base_orn)
        trans = base_pos - self._origin + basis @ self._origin
        mano_pose = self._model.angles_to_mano(angles, basis)
        return trans, mano_pose

    def reset_from_mano(self, trans, mano_pose):
        """Reset hand state from a Mano pose.

        Arguments:
            mano_pose {array} -- pose of the Mano model
            trans {vec3} -- hand translation
        """
        angles, basis = self._model.mano_to_angles(mano_pose)
        trans = trans + self._origin - basis @ self._origin
        self.reset(trans, mat2pb(basis), angles)

    def set_target_from_mano(self, trans, mano_pose):
        """Set target hand state from a Mano pose.

        Arguments:
            mano_pose {array} -- pose of the Mano model
            trans {vec3} -- hand translation
        """
        angles, basis = self._model.mano_to_angles(mano_pose)
        trans = trans + self._origin - basis @ self._origin
        self.set_target(trans, mat2pb(basis), angles)

    def _make_body(self):
        joints = self._model.joints
        link_masses = [0.2]
        link_collision_indices = [self._make_collision_shape(0, joints[0].basis.T)]
        link_visual_indices = [self._make_visual_shape(0, joints[0].basis.T)]
        link_positions = [joints[0].origin]
        link_orientations = [mat2pb(joints[0].basis)]
        link_parent_indices = [0]
        link_joint_types = [pb.JOINT_FIXED]
        link_joint_axis = [[0.0, 0.0, 0.0]]
        self._link_mapping[0] = 0

        for i, j in self._model.kintree_table.T[1:]:
            parent_index = self._link_mapping[i]
            origin_rel = joints[i].basis.T @ (joints[j].origin - joints[i].origin)
            basis_rel = joints[i].basis.T @ joints[j].basis

            for axis, limits in zip(joints[j].axes, joints[j].limits):
                link_masses.append(0.0)
                link_collision_indices.append(-1)
                link_visual_indices.append(-1)
                link_positions.append(origin_rel)
                link_orientations.append(mat2pb(basis_rel))
                link_parent_indices.append(parent_index+1)
                link_joint_types.append(pb.JOINT_REVOLUTE)
                link_joint_axis.append(np.eye(3)[ord(axis) - ord('x')])
                origin_rel, basis_rel = [0.0, 0.0, 0.0], np.eye(3)
                parent_index = len(link_masses) - 1
                self._link_mapping[j] = parent_index
                self._joint_indices.append(parent_index)
                self._joint_limits.append(limits)

            link_masses[-1] = 0.02
            link_visual_indices[-1] = self._make_visual_shape(j, joints[j].basis.T)
            link_collision_indices[-1] = self._make_collision_shape(j, joints[j].basis.T)

        flags = pb.URDF_INITIALIZE_SAT_FEATURES
        if self.FLAG_USE_SELF_COLLISION & self._flags:
            flags |= pb.URDF_USE_SELF_COLLISION
            flags |= pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS

        base_mass = 0.01
        if self.FLAG_STATIC & self._flags:
            base_mass = 0.0

        return self._client.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_indices,
            linkVisualShapeIndices=link_visual_indices,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * len(link_masses),
            linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * len(link_masses),
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axis,
            flags=flags)

    def _make_collision_shape(self, link_index, basis):
        if self.FLAG_ENABLE_COLLISION_SHAPES & self._flags:
            with self._temp_link_mesh(link_index, True) as filename:
                return self._client.createCollisionShape(
                    pb.GEOM_MESH,
                    fileName=filename,
                    meshScale=[1.0, 1.0, 1.0],
                    collisionFramePosition=[0, 0, 0],
                    collisionFrameOrientation=mat2pb(basis))
        return -1

    def _make_visual_shape(self, link_index, basis):
        if self.FLAG_ENABLE_VISUAL_SHAPES & self._flags:
            with self._temp_link_mesh(link_index, False) as filename:
                return self._client.createVisualShape(
                    pb.GEOM_MESH,
                    fileName=filename,
                    meshScale=[1.0, 1.0, 1.0],
                    rgbaColor=[0.0, 1.0, 0.0, 1.0],
                    specularColor=[1.0, 1.0, 1.0],
                    visualFramePosition=[0.0, 0.0, 0.0],
                    visualFrameOrientation=mat2pb(basis))
        return -1

    def _make_constraint(self):
        if not self.FLAG_STATIC & self._flags:
            return self._client.createConstraint(
                parentBodyUniqueId=self._body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=pb.JOINT_FIXED,
                jointAxis=[0.0, 0.0, 0.0],
                parentFramePosition=[0.0, 0.0, 0.0],
                childFramePosition=[0.0, 0.0, 0.0])
        return -1

    def _apply_joint_limits(self):
        if self.FLAG_JOINT_LIMITS & self._flags:
            for i, limits in zip(self._joint_indices, self._joint_limits):
                self._client.changeDynamics(
                    bodyUniqueId=self._body_id,
                    linkIndex=i,
                    jointLowerLimit=limits[0],
                    jointUpperLimit=limits[1])

    def _apply_dynamics(self):
        if self.FLAG_DYNAMICS & self._flags:
            self._client.changeDynamics(
                bodyUniqueId=self._body_id,
                linkIndex=-1,
                linearDamping=10.0,
                angularDamping=10.0)

            for i in [0] + self._joint_indices:
                self._client.changeDynamics(
                    bodyUniqueId=self._body_id,
                    linkIndex=i,
                    restitution=0.5,
                    lateralFriction=5.0,
                    spinningFriction=5.0)

    @contextlib.contextmanager
    def _temp_link_mesh(self, link_index, collision):
        with tempfile.NamedTemporaryFile('w', suffix='.obj') as temp_file:
            threshold = 0.2
            if collision and link_index in [4, 7, 10]:
                threshold = 0.7
            vertex_mask = self._model.weights[:, link_index] > threshold
            vertices, faces = filter_mesh(self._vertices, self._model.faces, vertex_mask)
            vertices -= self._model.joints[link_index].origin
            save_mesh_obj(temp_file.name, vertices, faces)
            yield temp_file.name
