from typing import Optional, Tuple

import tensorflow as tf

import wtftf
from pcn.augment import jitter, rigid


def augment(
    coords,
    num_points: int = 1024,
    shuffle: bool = True,
    rotate_scheme: str = rigid.RotateScheme.NONE,
    angle_stddev: Optional[float] = None,
    angle_clip: Optional[float] = None,
    jitter_stddev: float = None,
    jitter_clip: float = None,
    maybe_reflect_x: bool = False,
    uniform_scale_range: Optional[Tuple[float, float]] = None,
    drop_prob_limits: Optional[Tuple[float, float]] = None,
    up_dim: int = 2,
    shuffle_first: bool = False,
):
    coords.shape.assert_has_rank(2)
    assert coords.shape[1] == 3

    if shuffle_first and shuffle:
        coords = wtftf.random.shuffle(coords)

    if num_points is not None:
        coords = coords[:num_points]

    if not shuffle_first and shuffle:
        coords = wtftf.random.shuffle(coords)

    if drop_prob_limits is not None:
        lower, upper = drop_prob_limits
        drop_prob = wtftf.random.uniform((), lower, upper)
        if shuffle:
            size = tf.cast(
                tf.cast(tf.shape(coords)[0], tf.float32) * (1 - drop_prob), tf.int64
            )
            coords = coords[:size]
        else:
            mask = wtftf.random.uniform((tf.shape(coords)[0],)) > drop_prob
            coords = tf.boolean_mask(coords, mask)

    if up_dim != 2:
        roll = (2 - up_dim) % 3
        coords = tf.roll(coords, roll, axis=-1)
    if jitter_stddev is not None:
        coords = jitter.jitter_positions(coords, jitter_stddev, jitter_clip)
    if angle_stddev is not None:
        coords, _ = rigid.random_rotation(
            coords, angle_stddev=angle_stddev, angle_clip=angle_clip
        )
    if rotate_scheme is not None:
        coords, _ = rigid.rotate_by_scheme(coords, None, scheme=rotate_scheme)

    if uniform_scale_range is not None:
        coords = rigid.random_scale(coords, uniform_range=uniform_scale_range)

    if maybe_reflect_x:
        prob = wtftf.random.uniform(()) < 0.5
        coords = tf.cond(prob, lambda: coords, lambda: rigid.reflect(coords))

    return coords


def augment_segmentation(
    coords,
    normals,
    labels,
    num_points=None,
    shuffle=True,
    rotate_scheme="none",
    angle_stddev: Optional[float] = None,
    angle_clip: Optional[float] = None,
    jitter_stddev: float = None,
    jitter_clip: float = None,
    maybe_reflect_x=False,
    uniform_scale_range: Optional[Tuple[float, float]] = None,
    drop_prob_limits: Optional[Tuple[float, float]] = None,
    up_dim: int = 2,
):
    coords.shape.assert_has_rank(2)
    assert coords.shape[1] == 3
    normals.shape.assert_has_rank(2)
    assert normals.shape[1] == 3

    if num_points is not None:
        coords = coords[:num_points]
        normals = normals[:num_points]
        labels = labels[:num_points]

    size = tf.size(labels)
    if shuffle:
        order = wtftf.random.shuffle(tf.range(size))
        coords, normals, labels = (
            tf.gather(x, order) for x in (coords, normals, labels)
        )

    if drop_prob_limits is not None:
        lower, upper = drop_prob_limits
        drop_prob = wtftf.random.uniform((), lower, upper)
        if shuffle:
            size = tf.cast(
                tf.cast(tf.shape(coords)[0], tf.float32) * (1 - drop_prob), tf.int64
            )

            def f(x):
                return x[:size]

        else:
            mask = wtftf.random.uniform((tf.shape(coords)[0],)) > drop_prob

            def f(x):
                return tf.boolean_mask(x, mask)

        coords, normals, labels = (f(x) for x in (coords, normals, labels))

    if up_dim != 2:
        roll = (2 - up_dim) % 3
        coords, normals = (tf.roll(x, roll, axis=-1) for x in (coords, normals))
    if jitter_stddev is not None:
        coords = jitter.jitter_positions(coords, jitter_stddev, jitter_clip)
    if angle_stddev is not None:
        coords, normals = rigid.random_rotation(
            coords, normals, angle_stddev=angle_stddev, angle_clip=angle_clip
        )
    if rotate_scheme is not None:
        coords, normals = rigid.rotate_by_scheme(coords, normals, scheme=rotate_scheme)

    if uniform_scale_range is not None:
        coords = rigid.random_scale(coords, uniform_range=uniform_scale_range)

    if maybe_reflect_x:
        prob = wtftf.random.uniform(()) < 0.5
        coords, normals = tf.cond(
            prob,
            lambda: (coords, normals),
            lambda: (rigid.reflect(coords), rigid.reflect(normals)),
        )

    return coords, normals, labels
