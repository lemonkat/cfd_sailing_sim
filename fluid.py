import numpy as np
import pygame as pg
import pymunk as pm

import tensorflow as tf
from tensorflow import convert_to_tensor as ctt

from skimage import draw

import util
import draw_util
from draw_util import draw_vec_field

floor = tf.math.floor
ceil = tf.math.ceil

FLOAT = tf.float32
INT = tf.int32

R90 = tf.constant([[0, -1], [1, 0]], dtype=INT)

"""
all ops used:
0. tf.math.floor
1. tf.math.ceil
2. subtraction
3. tf.range
4. tf.cast
5. tf.stack
6. tf.scatter_nd
7. addition
8. tf.roll
9. tf.tensor_scatter_nd_add
10. tf.reduce_sum
11. tf.concat
12. tf.fill
13. tf.slice
"""


@tf.function
def frac(x: tf.Tensor) -> tf.Tensor:
    return x - floor(x)


def np_grid(arr: np.ndarray[np.float32], *tgt_shape: list[int]) -> np.ndarray:
    n_spec = len(tgt_shape)
    shape = []
    for i in range(n_spec):
        shape.append(tgt_shape[i])
        shape.append(arr.shape[i] // tgt_shape[i])
        arr = arr.swapaxes(0, i)[
            : (arr.shape[i] // tgt_shape[i]) * tgt_shape[i]
        ].swapaxes(0, i)

    shape.extend(arr.shape[i] for i in range(n_spec, len(arr.shape)))
    # shape = shape[0] * (arr.shape[0] // shape[0]), shape[0], shape[1] * (arr.shape[1] // shape[1]), shape[1]
    # ax_order = [i * 2 for i in range(n_spec)] + [i * 2 + 1 for i in range(n_spec)] + list(range(n_spec, len(arr.shape)))
    ax_order = (
        list(range(0, n_spec * 2, 2))
        + list(range(1, n_spec * 2, 2))
        + list(range(n_spec * 2, n_spec + len(arr.shape)))
    )
    return arr.reshape(shape).transpose(ax_order)


def grid(arr: tf.Tensor, *tgt_shape: list[int]) -> tf.Tensor:
    n_spec = len(tgt_shape)
    shape = []
    for i in range(n_spec):
        shape.append(tgt_shape[i])
        shape.append(arr.shape[i] // tgt_shape[i])
    arr = arr[tuple(slice((a // t) * t) for a, t in zip(arr.shape, tgt_shape))]
    shape.extend(arr.shape[i] for i in range(n_spec, len(arr.shape)))
    ax_order = (
        list(range(0, n_spec * 2, 2))
        + list(range(1, n_spec * 2, 2))
        + list(range(n_spec * 2, n_spec + len(arr.shape)))
    )
    return tf.transpose(tf.reshape(arr, shape), ax_order)


# @tf.function
# def blur(field: tf.Tensor, a: tf.Tensor = 0.6) -> tf.Tensor:
#     return a * field + 0.25 * (1 - a) * (tf.roll(field, 0, -1) + tf.roll(field, 1, -1) + tf.roll(field, 0, 1) + tf.roll(field, 1, 1))


@tf.function
def induce(
    field: tf.Tensor, val: tf.Tensor, depth: tf.Tensor, a: tf.Tensor
) -> tf.Tensor:
    inv_a = 1 - a

    horz_side = tf.fill((tf.shape(field)[0], depth), val * a)
    vert_side = tf.fill((depth, tf.shape(field)[1] - 2 * depth), val * a)
    
    return tf.concat(
        [
            horz_side + field[:, :depth] * inv_a,
            tf.concat(
                [
                    vert_side + field[:depth, depth:-depth] * inv_a,
                    field[depth:-depth, depth:-depth],
                    vert_side + field[-depth:, depth:-depth] * inv_a,
                ],
                axis=0,
            ),
            horz_side + field[:, -depth:] * inv_a,
        ],
        axis=1,
    )



@tf.function
def advect(
    field: tf.Tensor, vel_i: tf.Tensor, vel_j: tf.Tensor, dt: tf.Tensor
) -> tf.Tensor:
    field_shape = tf.shape(field, out_type=INT)

    tgt_i = (
        vel_i * dt
        + tf.cast(tf.range(0, field_shape[0], 1, dtype=INT), dtype=FLOAT)[:, None]
    )
    tgt_j = (
        vel_j * dt
        + tf.cast(tf.range(0, field_shape[1], 1, dtype=INT), dtype=FLOAT)[None, :]
    )

    f_i = tf.cast(tf.math.floor(tgt_i), INT) % field_shape[0]
    f_j = tf.cast(tf.math.floor(tgt_j), INT) % field_shape[1]
    c_i = tf.cast(tf.math.ceil(tgt_i), INT) % field_shape[0]
    c_j = tf.cast(tf.math.ceil(tgt_j), INT) % field_shape[1]

    frac_i = tgt_i - tf.math.floor(tgt_i)
    frac_j = tgt_j - tf.math.floor(tgt_j)

    ff = tf.scatter_nd(
        tf.stack([f_i, f_j], axis=2),
        ((1 - frac_i) * (1 - frac_j)) * field,
        shape=field_shape,
    )
    fc = tf.scatter_nd(
        tf.stack([f_i, c_j], axis=2),
        ((1 - frac_i) * frac_j) * field,
        shape=field_shape,
    )
    cf = tf.scatter_nd(
        tf.stack([c_i, f_j], axis=2),
        (frac_i * (1 - frac_j)) * field,
        shape=field_shape,
    )
    cc = tf.scatter_nd(
        tf.stack([c_i, c_j], axis=2),
        (frac_i * frac_j) * field,
        shape=field_shape,
    )

    return ff + fc + cf + cc


@tf.function
def divergence(vel_i: tf.Tensor, vel_j: tf.Tensor) -> tf.Tensor:
    grad_i = 0.5 * (tf.roll(vel_i, 1, axis=0) - tf.roll(vel_i, -1, axis=0))
    grad_j = 0.5 * (tf.roll(vel_j, 1, axis=1) - tf.roll(vel_j, -1, axis=1))
    return grad_i + grad_j


@tf.function
def project(
    dens: tf.Tensor, dt: tf.Tensor, a: int = 100
) -> tuple[tf.Tensor, tf.Tensor]:
    grad_i = 0.5 * (tf.roll(dens, 1, axis=0) - tf.roll(dens, -1, axis=0))
    grad_j = 0.5 * (tf.roll(dens, 1, axis=1) - tf.roll(dens, -1, axis=1))
    return dt * a * grad_i, dt * a * grad_j


@tf.function(reduce_retracing=True)
def get_forces(
    fvel_i: tf.Tensor,
    fvel_j: tf.Tensor,
    dens: tf.Tensor,
    pos_i: tf.Tensor,
    pos_j: tf.Tensor,
    vel_i: tf.Tensor,
    vel_j: tf.Tensor,
    avel: tf.Tensor,
    weight: tf.Tensor,
    rr: tf.Tensor,
    cc: tf.Tensor,
    scale: tf.Tensor,
    loc_i: tf.Tensor,
    loc_j: tf.Tensor,
) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]]:
    rr_float = tf.cast(rr, dtype=FLOAT)
    cc_float = tf.cast(cc, dtype=FLOAT)

    pos_i, pos_j = pos_i / scale - loc_i, pos_j / scale - loc_j
    vel_i, vel_j = vel_i / scale, vel_j / scale
    vel_i = -(cc_float - pos_i) * avel + vel_i
    vel_j = (rr_float - pos_j) * avel + vel_j

    idx = tf.stack([rr, cc], axis=1)

    rvel_i = vel_i - tf.gather_nd(fvel_i, idx)
    rvel_j = vel_j - tf.gather_nd(fvel_j, idx)

    fvel_i = tf.tensor_scatter_nd_add(fvel_i, idx, rvel_i * weight)
    fvel_j = tf.tensor_scatter_nd_add(fvel_j, idx, rvel_j * weight)

    dens = tf.gather_nd(dens, idx) * (scale**-2)
    force_i = -tf.reduce_sum(dens * rvel_i) * weight / scale
    force_j = -tf.reduce_sum(dens * rvel_j) * weight / scale

    cross = rvel_i * (cc_float - pos_j) - rvel_j * (rr_float - pos_i)
    torque = weight * tf.reduce_sum(dens * cross)

    return (fvel_i, fvel_j), ((force_i, force_j), torque)


@tf.function
def step(
    vel_i: tf.Tensor, vel_j: tf.Tensor, dens: tf.Tensor, dt: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # vel_i_1 = advect(vel_i, vel_i, vel_j, dt)
    # vel_j_1 = advect(vel_j, vel_i, vel_j, dt)

    # pv_i, pv_j = project(dens, dt, 10)
    # vel_i = vel_i_1 + pv_i
    # vel_j = vel_j_1 + pv_j

    # dens = advect(dens, vel_i, vel_j, dt)

    # return (vel_i, vel_j), dens

    # advect density and momentum - not density and velocity
    mtm_i, mtm_j = vel_i * dens, vel_j * dens

    mtm_i_1, mtm_j_1 = advect(mtm_i, vel_i, vel_j, dt), advect(mtm_j, vel_i, vel_j, dt)

    dens = advect(dens, vel_i, vel_j, dt)

    pv_i, pv_j = project(dens, dt, 100)

    vel_i = mtm_i_1 / dens + pv_i
    vel_j = mtm_j_1 / dens + pv_j

    return vel_i, vel_j, dens


# class StepModel(tf.keras.Model):
#     def call(self, vel_i, vel_j, dens, dt):
#         return step(vel_i, vel_j, dens, dt)

# def get_optimized_step_model(shape):
#     def representative_dataset():
#         zero = ctt(np.zeros(shape), dtype=FLOAT)
#         high = ctt(np.full(shape, 10), dtype=FLOAT)
#         low = ctt(np.full(shape, -10), dtype=FLOAT)
#         yield high, high, high
#         yield high, low, high
#         yield low, high, high
#         yield low, low, high
#         yield high, high, zero
#         yield high, low, zero
#         yield low, high, zero
#         yield low, low, zero
#     converter = tf.lite.TFLiteConverter.from_keras_model(StepModel())
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.representative_dataset = representative_dataset
#     return converter.convert()


def shape_to_rrcc(
    obj: pm.Body, shape: pm.Shape, scale: float, loc: pm.Vec2d
) -> tuple[tf.Tensor, tf.Tensor]:
    l2w = obj.local_to_world

    def pm_to_grid(pos: pm.Vec2d) -> pm.Vec2d:
        return pos / scale - loc

    if isinstance(shape, pm.Circle):
        rr, cc = draw.ellipse(
            *pm_to_grid(l2w(shape.offset)), shape.radius * scale, shape.radius * scale
        )

    elif isinstance(shape, pm.Segment):
        a, b = pm_to_grid(l2w(shape.a)), pm_to_grid(l2w(shape.b))
        rr, cc, _ = draw.line_aa(*util.to_int(a), *util.to_int(b))

    elif isinstance(shape, pm.Poly):
        verts = [pm_to_grid(l2w(vert)) for vert in shape.get_vertices()]
        rr, cc = draw.polygon(*zip(*verts))

    else:
        rr, cc = np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    return ctt(rr, dtype=INT), ctt(cc, dtype=INT)


# @tf.function
# def shift_old(inputs, shift, axes, pad_val=0):
#     assert len(shift) == len(axes)
#     axis_shift = zip(axes, shift)
#     axis2shift = dict(axis_shift)
#     old_shape = inputs.shape

#     for axis in axis2shift:
#         pad_shape = list(inputs.shape)
#         pad_shape[axis] = abs(axis2shift[axis])
#         input_pad = tf.fill(pad_shape, pad_val)
#         inputs = tf.concat((inputs, input_pad), axis=axis)

#     input_roll = tf.roll(inputs, shift, axes)
#     ret = tf.slice(input_roll, [0 for _ in range(len(old_shape))], old_shape)

#     return ret


@tf.function
def shift(field: tf.Tensor, i: tf.Tensor, j: tf.Tensor, fill: tf.Tensor) -> tf.Tensor:
    if i < 0:
        i = tf.abs(i)
        field = tf.concat(
            [
                field[i:, :],
                tf.fill((i, tf.shape(field)[1]), fill),
            ],
            axis=0,
        )
    elif i > 0:
        field = tf.concat(
            [
                tf.fill((i, tf.shape(field)[1]), fill),
                field[:-i, :],
            ],
            axis=0,
        )

    if j < 0:
        j = tf.abs(j)
        field = tf.concat(
            [
                field[:, j:],
                tf.fill((tf.shape(field)[0], j), fill),
            ],
            axis=1,
        )
    elif j > 0:
        field = tf.concat(
            [
                tf.fill((tf.shape(field)[0], j), fill),
                field[:, :-j],
            ],
            axis=1,
        )

    return field


class Fluid:
    def __init__(
        self,
        shape: tuple[int, int],
        current: pm.Vec2d,
        loc: tuple[int, int] = (0, 0),
        scale: float = 1.0,
    ):
        self.shape = shape
        self.current = current
        self.vel_i = tf.Variable(tf.cast(tf.fill(shape, current.x), dtype=FLOAT))
        self.vel_j = tf.Variable(tf.cast(tf.fill(shape, current.y), dtype=FLOAT))

        self.loc = loc  # in PM-coordinates, position of (0, 0)
        self.scale = scale  # 1 grid = how many PM units?

        self.dens = tf.Variable(
            tf.constant(np.full(shape, 1, dtype=np.float32), dtype=FLOAT)
        )

        ind_mask = np.full(shape, 1.0, dtype=np.float32)
        ind_mask[3: -3, 3: -3] = 0.0
        self._ind_mask = ctt(ind_mask, dtype=FLOAT)

    def run(self, steps: int, dt: float = 0.1) -> None:
        vel_i, vel_j, dens = self.vel_i, self.vel_j, self.dens

        for _ in range(steps):
            vel_i, vel_j, dens = step(vel_i, vel_j, dens, tf.constant(dt, dtype=FLOAT))

        self.vel_i.assign(vel_i)
        self.vel_j.assign(vel_j)
        self.dens.assign(dens)

    def step(self, dt: float = 0.1) -> None:
        vel_i, vel_j, dens = step(
            self.vel_i, self.vel_j, self.dens, tf.constant(dt, dtype=FLOAT)
        )

        self.vel_i.assign(induce(vel_i, ctt(self.current[0], dtype=FLOAT), ctt(3, dtype=INT), ctt(1.0, dtype=FLOAT)))
        self.vel_j.assign(induce(vel_j, ctt(self.current[1], dtype=FLOAT), ctt(3, dtype=INT), ctt(1.0, dtype=FLOAT)))
        self.dens.assign(induce(dens, ctt(1.0, dtype=FLOAT), ctt(3, dtype=INT), ctt(1.0, dtype=FLOAT)))

    def draw(self, surface: pg.Surface, num_arrows: int = 25) -> None:
        surface.fill([255, 255, 255])
        surf_arr = pg.surfarray.pixels3d(surface)
        surf_arr_scaled = np_grid(surf_arr, *self.shape)
        dens = self.dens.numpy()
        colors = (255 - 70 * dens[:, :, None, None, None]).astype(int)
        colors[colors < 0] = 0
        colors[colors > 255] = 255
        surf_arr_scaled[..., :-1] = colors

        vel_scaled = np.mean(
            np_grid(
                np.stack([self.vel_i.numpy(), self.vel_j.numpy()], axis=2)
                * self.dens.numpy()[:, :, None],
                num_arrows,
                num_arrows,
            ),
            axis=(2, 3),
        )
        draw_vec_field(surface, vel_scaled, 0.5, [255, 0, 0])

    # "move" this field to be centered around position
    # pos is in PM-coordinates
    def shift(self, i: int, j: int) -> None:
        i = ctt(i, dtype=INT)
        j = ctt(j, dtype=INT)

        self.vel_i.assign(shift(self.vel_i, i, j, ctt(self.current.x, dtype=FLOAT)))
        self.vel_j.assign(shift(self.vel_j, i, j, ctt(self.current.y, dtype=FLOAT)))
        self.dens.assign(shift(self.dens, i, j, ctt(1.0, dtype=FLOAT)))

        self.loc = self.loc[0] + i, self.loc[1] + j

    def pm_to_grid(self, pos: pm.Vec2d) -> pm.Vec2d:
        return (pos - self.loc) / self.scale

    def pm_to_idx(self, pos: pm.Vec2d) -> tuple[float, float]:
        return util.to_int(self.pm_to_grid(pos))

    def grid_to_pm(self, pos: pm.Vec2d) -> pm.Vec2d:
        return pos * self.scale + self.loc

    def apply_forces(self, obj: pm.Body, shape: pm.Shape, weight: float = 1.0) -> None:
        rr, cc = shape_to_rrcc(obj, shape, self.scale, self.loc)
        (vel_i, vel_j), ((force_i, force_j), torque) = get_forces(
            self.vel_i,
            self.vel_j,
            self.dens,
            ctt(obj.position.x, dtype=FLOAT),
            ctt(obj.position.y, dtype=FLOAT),
            ctt(obj.velocity.x, dtype=FLOAT),
            ctt(obj.velocity.y, dtype=FLOAT),
            ctt(obj.angular_velocity, dtype=FLOAT),
            ctt(weight, dtype=FLOAT),
            rr,
            cc,
            ctt(self.scale, dtype=FLOAT),
            ctt(self.loc[0], dtype=FLOAT),
            ctt(self.loc[1], dtype=FLOAT),
        )
        force = pm.Vec2d(force_i.numpy(), force_j.numpy())
        print(force, torque.numpy())
        obj.apply_force_at_world_point(force, obj.center_of_gravity)
        obj.torque += torque.numpy()
        self.vel_i.assign(vel_i)
        self.vel_j.assign(vel_j)



if __name__ == "__main__":
    space = pm.Space()
    space.damping = 0.95
    wing = pm.Body(body_type=pm.Body.DYNAMIC)

    wing_shape = pm.Segment(wing, pm.Vec2d(-30, 0), pm.Vec2d(30, 0), 0.1)

    wing_shape.density = 10.0
    wing.position = pm.Vec2d(0, 0)
    wing.angular_velocity = 1
    wing.angle = 0.5

    space.add(wing, wing_shape)

    fl = Fluid([500, 500], pm.Vec2d(0.5, 0), (-250, -250), 1.0)

    pg.init()
    surface = pg.display.set_mode((1000, 1000))

    clock = pg.time.Clock()
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        for _ in range(1):
            for _ in range(5):
                space.step(2 / 60)
                fl.run(1, 2 / 60)

                fl.apply_forces(wing, wing_shape, 0.3)

        fl.draw(surface, 40)

        draw_util.draw_shape(
            surface, wing, wing_shape, surface.get_width() / fl.shape[0]
        )
        pg.display.flip()

        # clock.tick(60)  # limits FPS to 60
    pg.quit()
