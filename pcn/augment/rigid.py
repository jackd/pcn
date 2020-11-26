from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf

import wtftf


class RotateScheme:
    NONE = "none"
    RANDOM = "random"
    PCA_XY = "pca-xy"

    @classmethod
    def all(cls):
        return (RotateScheme.NONE, RotateScheme.RANDOM, RotateScheme.PCA_XY)

    @classmethod
    def validate(cls, value: str):
        if value not in cls.all():
            raise ValueError(
                f"Invalid {cls.__name__} value {value} - must be in {cls.all()}"
            )


def _get_pc_tf(xy: tf.Tensor) -> tf.Tensor:
    s, u, v = tf.linalg.svd(xy)
    del s, u
    return tf.gather(v, 0, axis=-1)


def get_pca_xy_angle(positions: tf.Tensor) -> tf.Tensor:
    xy, _ = tf.split(positions, [2, 1], axis=-1)
    pca_vec = _get_pc_tf(xy)
    x, y = tf.unstack(pca_vec, axis=0)
    angle = tf.atan2(y, x)
    return angle


def from_axis_angle(axis: tf.Tensor, angle: tf.Tensor, name=None) -> tf.Tensor:
    """Convert an axis-angle representation to a rotation matrix.

    Straight from tensorflow_graphics. Copied here because importing
    graphics takes FOREVER, and this is the only use.

    Note:
      In the following, A1 to An are optional batch dimensions, which must be
      broadcast compatible.

    Args:
      axis: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
        represents a normalized axis.
      angle: A tensor of shape `[A1, ..., An, 1]`, where the last dimension
        represents a normalized axis.
      name: A name for this op that defaults to
        "rotation_matrix_3d_from_axis_angle".

    Returns:
      A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
      represents a 3d rotation matrix.

    Raises:
      ValueError: If the shape of `axis` or `angle` is not supported.
    """
    with tf.name_scope(name or "rotation_matrix_3d_from_axis_angle"):
        axis = tf.convert_to_tensor(value=axis)
        angle = tf.convert_to_tensor(value=angle)

        # shape.check_static(tensor=axis,
        #                    tensor_name="axis",
        #                    has_dim_equals=(-1, 3))
        # shape.check_static(tensor=angle,
        #                    tensor_name="angle",
        #                    has_dim_equals=(-1, 1))
        # shape.compare_batch_dimensions(tensors=(axis, angle),
        #                                tensor_names=("axis", "angle"),
        #                                last_axes=-2,
        #                                broadcast_compatible=True)
        # axis = asserts.assert_normalized(axis)

        sin_axis = tf.sin(angle) * axis
        cos_angle = tf.cos(angle)
        cos1_axis = (1.0 - cos_angle) * axis
        _, axis_y, axis_z = tf.unstack(axis, axis=-1)
        cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
        sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1)
        tmp = cos1_axis_x * axis_y
        m01 = tmp - sin_axis_z
        m10 = tmp + sin_axis_z
        tmp = cos1_axis_x * axis_z
        m02 = tmp + sin_axis_y
        m20 = tmp - sin_axis_y
        tmp = cos1_axis_y * axis_z
        m12 = tmp - sin_axis_x
        m21 = tmp + sin_axis_x
        diag = cos1_axis * axis + cos_angle
        diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
        matrix = tf.stack(
            (diag_x, m01, m02, m10, diag_y, m12, m20, m21, diag_z), axis=-1
        )  # pyformat: disable
        output_shape = tf.concat((tf.shape(input=axis)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)


def _pack_rotation_matrix(c, s, rotation_dim=2):
    # https://en.wikipedia.org/wiki/Rotation_matrix
    if rotation_dim == 0:
        return [1, 0, 0, 0, c, -s, 0, s, c]
    elif rotation_dim == 1:
        return [c, 0, s, 0, 1, 0, -s, 0, c]
    elif rotation_dim == 2:
        return [c, -s, 0, s, c, 0, 0, 0, 1]
    else:
        raise ValueError("rotation_dim must be 0, 1 or 2, got {}".format(rotation_dim))


def _rotate(positions, normals=None, angle=None, rotation_dim=2, impl=tf):
    """
    Randomly rotate the point cloud about the z-axis.

    Args:
        positions: (n, 3) float array
        normals (optional): (n, 3) float array
        angle: float scalar. If None, a uniform random angle in [0, 2pi) is
            used.
        rotation_dim: int denoting x (0), y (1), or z (2) axis about which to
            rotate
        impl: tf or np

    Returns:
        rotated (`positions`, `normals`). `normals` will be None if not
        provided. shape and dtype is the same as provided.
    """
    dtype = positions.dtype
    if angle is None:
        angle = wtftf.random.uniform((), dtype=dtype) * (2 * np.pi)

    if normals is not None:
        assert normals.dtype == dtype
    c = impl.cos(angle)
    s = impl.sin(angle)
    # multiply on right, use non-standard rotation matrix (-s and s swapped)
    rotation_matrix = impl.reshape(
        impl.stack(_pack_rotation_matrix(c, s, rotation_dim=rotation_dim)), (3, 3)
    )

    positions = impl.matmul(positions, rotation_matrix)
    if normals is not None:
        normals = impl.matmul(normals, rotation_matrix)
    return positions, normals


def rotate(
    positions: tf.Tensor,
    normals: Optional[tf.Tensor] = None,
    angle=None,
    rotation_dim=2,
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    if tf.is_tensor(angle) or angle != 0:
        positions, normals = _rotate(
            positions, normals, angle, rotation_dim=rotation_dim, impl=tf
        )
    return positions, normals


def rotate_np(positions, normals=None, angle=None, rotation_dim=2):
    return _rotate(positions, normals, angle, rotation_dim=rotation_dim, impl=np)


def reflect(xyz: tf.Tensor, dim: int = 0, axis: int = -1) -> tf.Tensor:
    values = tf.unstack(xyz, axis=axis)
    values[dim] *= -1
    return tf.stack(values, axis=axis)


def random_rigid_transform_matrix(stddev=0.02, clip=None, dim=3) -> tf.Tensor:
    offset = wtftf.random.normal(shape=(dim, dim), stddev=stddev)
    if clip:
        offset = tf.clip_by_value(
            offset, -clip, clip
        )  # pylint: disable=invalid-unary-operand-type
    return tf.eye(dim) + offset


def rotate_by_scheme(
    positions, normals=None, scheme: str = RotateScheme.RANDOM
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    """scheme should be in ("random", "pca-xy", "none")."""
    RotateScheme.validate(scheme)
    if scheme == RotateScheme.NONE:
        return positions, normals

    if scheme == RotateScheme.PCA_XY:
        angle = get_pca_xy_angle(positions)
    elif scheme == RotateScheme.RANDOM:
        angle = wtftf.random.uniform(shape=(), dtype=positions.dtype) * (2 * np.pi)
    else:
        raise ValueError('Unrecognized scheme "%s"' % scheme)
    return rotate(positions, normals, angle)


def random_rigid_transform(
    positions: tf.Tensor, normals: Optional[tf.Tensor] = None, stddev=0.02, clip=None
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    transform = random_rigid_transform_matrix(stddev, clip, positions.shape[-1])
    positions = tf.matmul(positions, transform)
    if normals is not None:
        raise NotImplementedError("Normal rigid transform not implemented")
    return positions, normals


def _maybe_reflect(positions, axis=-1, dim=0, prob=0.5):
    should_reflect = wtftf.random.uniform(shape=(), dtype=tf.float32) > prob
    return tf.cond(
        should_reflect,
        lambda: tuple(reflect(p, dim=dim, axis=axis) for p in positions),
        lambda: tuple(positions),
    )


def maybe_reflect(
    positions, normals=None, **kwargs
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    if normals is None:
        return _maybe_reflect((positions,), **kwargs)[0], normals
    return _maybe_reflect((positions, normals), **kwargs)


def random_scale(positions: tf.Tensor, stddev=None, uniform_range=None) -> tf.Tensor:
    if stddev is not None:
        scale = wtftf.random.truncated_normal(shape=(), mean=1.0, stddev=stddev)
    elif uniform_range is not None:
        minval, maxval = uniform_range
        scale = wtftf.random.uniform(shape=(), minval=minval, maxval=maxval)
    else:
        raise NotImplementedError("One of stddev or uniform_range must be defined")
    return positions * scale


def random_rotation_matrix(
    batch_shape=(), angle_stddev=0.06, angle_clip=0.18
) -> tf.Tensor:
    # slightly different to the one used in pointnet2
    # we use from_axis_angle rather than from_euler_angles
    # from tensorflow_graphics.geometry.transformation.rotation_matrix_3d \
    #   import from_axis_angle
    batch_shape = tuple(batch_shape)
    axis = wtftf.random.normal(shape=batch_shape + (3,))
    axis = axis / tf.linalg.norm(axis, axis=-1, keepdims=True)
    angle = wtftf.random.normal(shape=batch_shape + (1,), stddev=angle_stddev)
    if angle_clip:
        angle = tf.clip_by_value(angle, -angle_clip, angle_clip)
    return from_axis_angle(axis, angle)


def random_rotation(
    positions: tf.Tensor,
    normals: Optional[tf.Tensor] = None,
    angle_stddev: Union[tf.Tensor, float] = 0.06,
    angle_clip: Union[tf.Tensor, float] = 0.18,
) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    batch_shape = positions.shape[:-2].as_list()
    matrix = random_rotation_matrix(batch_shape, angle_stddev, angle_clip)
    positions = tf.matmul(positions, matrix)
    if normals is not None:
        normals = tf.matmul(normals, matrix)
    return positions, normals
