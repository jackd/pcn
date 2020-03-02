from typing import Optional, Tuple
import tensorflow as tf
import gin
from . import jitter
from . import rigid


@gin.configurable(module='pcn')
def augment(coords,
            num_points=1024,
            shuffle=True,
            rotate_scheme='none',
            angle_stddev: Optional[float] = None,
            angle_clip: Optional[float] = None,
            jitter_stddev: float = None,
            jitter_clip: float = None,
            maybe_reflect_x=False,
            uniform_scale_range: Optional[Tuple[float, float]] = None,
            drop_prob_limits: Optional[Tuple[float, float]] = None,
            up_dim: int = 2):
    coords.shape.assert_has_rank(2)
    assert (coords.shape[1] == 3)

    if num_points is not None:
        coords = coords[:num_points]

    if shuffle:
        coords = tf.random.shuffle(coords)

    if drop_prob_limits is not None:
        lower, upper = drop_prob_limits
        drop_prob = tf.random.uniform((), lower, upper)
        if shuffle:
            size = tf.cast(
                tf.cast(tf.shape(coords)[0], tf.float32) * (1 - drop_prob),
                tf.int64)
            coords = coords[:size]
        else:
            mask = tf.random.uniform((tf.shape(coords)[0],)) > drop_prob
            coords = tf.boolean_mask(coords, mask)

    if up_dim != 2:
        roll = (2 - up_dim) % 3
        coords = tf.roll(coords, roll, axis=-1)
    if jitter_stddev is not None:
        coords = jitter.jitter_positions(coords, jitter_stddev, jitter_clip)
    if angle_stddev is not None:
        coords, _ = rigid.random_rotation(coords,
                                          angle_stddev=angle_stddev,
                                          angle_clip=angle_clip)
    if rotate_scheme is not None:
        coords, _ = rigid.rotate_by_scheme(coords, None, scheme=rotate_scheme)

    if uniform_scale_range is not None:
        coords = rigid.random_scale(coords, uniform_range=uniform_scale_range)

    return coords
