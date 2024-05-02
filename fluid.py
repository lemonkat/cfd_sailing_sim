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

@tf.function
def frac(x: tf.Tensor) -> tf.Tensor:
    return x - floor(x)

def np_grid(arr: np.ndarray[np.float32], *tgt_shape: list[int]) -> np.ndarray:
    n_spec = len(tgt_shape)
    shape = []
    for i in range(n_spec):
        shape.append(tgt_shape[i])
        shape.append(arr.shape[i] // tgt_shape[i])
        arr = arr.swapaxes(0, i)[:(arr.shape[i] // tgt_shape[i]) * tgt_shape[i]].swapaxes(0, i)
    
    shape.extend(arr.shape[i] for i in range(n_spec, len(arr.shape)))
    # shape = shape[0] * (arr.shape[0] // shape[0]), shape[0], shape[1] * (arr.shape[1] // shape[1]), shape[1]
    # ax_order = [i * 2 for i in range(n_spec)] + [i * 2 + 1 for i in range(n_spec)] + list(range(n_spec, len(arr.shape)))
    ax_order = list(range(0, n_spec * 2, 2)) + list(range(1, n_spec * 2, 2)) + list(range(n_spec * 2, n_spec + len(arr.shape)))
    return arr.reshape(shape).transpose(ax_order)

def grid(arr: tf.Tensor, *tgt_shape: list[int]) -> tf.Tensor:
    n_spec = len(tgt_shape)
    shape = []
    for i in range(n_spec):
        shape.append(tgt_shape[i])
        shape.append(arr.shape[i] // tgt_shape[i])
    arr = arr[tuple(slice((a // t) * t) for a, t in zip(arr.shape, tgt_shape))]
    shape.extend(arr.shape[i] for i in range(n_spec, len(arr.shape)))
    ax_order = list(range(0, n_spec * 2, 2)) + list(range(1, n_spec * 2, 2)) + list(range(n_spec * 2, n_spec + len(arr.shape)))
    return tf.transpose(tf.reshape(arr, shape), ax_order)

# @tf.function
# def blur(field: tf.Tensor, a: tf.Tensor = 0.6) -> tf.Tensor:
#     return a * field + 0.25 * (1 - a) * (tf.roll(field, 0, -1) + tf.roll(field, 1, -1) + tf.roll(field, 0, 1) + tf.roll(field, 1, 1))

@tf.function
def advect(field: tf.Tensor, vel_i: tf.Tensor, vel_j: tf.Tensor, dt: tf.Tensor) -> tf.Tensor:
    tgt_i = vel_i * dt + tf.cast(tf.range(0, field.shape[0], 1, dtype=INT), dtype=FLOAT)[:, None]
    tgt_j = vel_j * dt + tf.cast(tf.range(0, field.shape[1], 1, dtype=INT), dtype=FLOAT)[None, :]

    f_i = tf.cast(tf.math.floor(tgt_i), INT) % field.shape[0]
    f_j = tf.cast(tf.math.floor(tgt_j), INT) % field.shape[1]
    c_i = tf.cast(tf.math.ceil(tgt_i), INT) % field.shape[0]
    c_j = tf.cast(tf.math.ceil(tgt_j), INT) % field.shape[1]

    frac_i = tgt_i - tf.math.floor(tgt_i)
    frac_j = tgt_j - tf.math.floor(tgt_j)

    ff = tf.scatter_nd(tf.stack([f_i, f_j], axis=2), ((1 - frac_i) * (1 - frac_j)) * field, shape=field.shape)
    fc = tf.scatter_nd(tf.stack([f_i, c_j], axis=2), ((1 - frac_i) * frac_j) * field, shape=field.shape)
    cf = tf.scatter_nd(tf.stack([c_i, f_j], axis=2), (frac_i * (1 - frac_j)) * field, shape=field.shape)
    cc = tf.scatter_nd(tf.stack([c_i, c_j], axis=2), (frac_i * frac_j) * field, shape=field.shape)

    return ff + fc + cf + cc

@tf.function
def divergence(vel_i: tf.Tensor, vel_j: tf.Tensor) -> tf.Tensor:
    grad_i = 0.5 * (tf.roll(vel_i, 1, axis=0) - tf.roll(vel_i, -1, axis=0))
    grad_j = 0.5 * (tf.roll(vel_j, 1, axis=1) - tf.roll(vel_j, -1, axis=1))
    return grad_i + grad_j

@tf.function
def project(dens: tf.Tensor, dt: tf.Tensor, a: int = 100) -> tuple[tf.Tensor, tf.Tensor]:
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
    loc_j: tf.Tensor
) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]]:
    rr_float = tf.cast(rr, dtype=FLOAT)
    cc_float = tf.cast(cc, dtype=FLOAT)

    pos_i, pos_j = (pos_i - loc_i) / scale, (pos_j - loc_j) / scale
    vel_i, vel_j = vel_i / scale, vel_j / scale
    vel_i = -(cc_float - pos_i) * avel + vel_i
    vel_j = (rr_float - pos_j) * avel + vel_j

    idx = tf.stack([rr, cc], axis=1)

    rvel_i = vel_i - tf.gather_nd(fvel_i, idx)
    rvel_j = vel_j - tf.gather_nd(fvel_j, idx)

    fvel_i = tf.tensor_scatter_nd_add(fvel_i, idx, rvel_i * weight)
    fvel_j = tf.tensor_scatter_nd_add(fvel_j, idx, rvel_j * weight)

    dens = tf.gather_nd(dens, idx) * (scale ** -2)
    force_i = - tf.reduce_sum(dens * rvel_i) * weight / scale
    force_j = - tf.reduce_sum(dens * rvel_j) * weight / scale
    
    cross = rvel_i * (cc_float - pos_j) - rvel_j * (rr_float - pos_i)
    torque = weight * tf.reduce_sum(dens * cross)

    return (fvel_i, fvel_j), ((force_i, force_j), torque)

@tf.function
def step(vel_i: tf.Tensor, vel_j: tf.Tensor, dens: tf.Tensor, dt: tf.Tensor) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    vel_i_1 = advect(vel_i, vel_i, vel_j, dt)
    vel_j_1 = advect(vel_j, vel_i, vel_j, dt)

    pv_i, pv_j = project(dens, dt, 10)
    vel_i = vel_i_1 + pv_i
    vel_j = vel_j_1 + pv_j
    
    dens = advect(dens, vel_i, vel_j, dt)

    return (vel_i, vel_j), dens

def shape_to_rrcc(obj: pm.Body, shape: pm.Shape, scale: float, loc: pm.Vec2d) -> tuple[tf.Tensor, tf.Tensor]:
    l2w = obj.local_to_world

    def pm_to_grid(pos: pm.Vec2d) -> pm.Vec2d:
        return (pos - loc) / scale
    
    if isinstance(shape, pm.Circle):
        rr, cc = draw.ellipse(*pm_to_grid(l2w(shape.offset)), shape.radius * scale, shape.radius * scale)
    
    elif isinstance(shape, pm.Segment):
        a, b = pm_to_grid(l2w(shape.a)), pm_to_grid(l2w(shape.b))
        rr, cc, _ = draw.line_aa(*util.to_int(a), *util.to_int(b))

    elif isinstance(shape, pm.Poly):
        verts = [pm_to_grid(l2w(vert)) for vert in shape.get_vertices()]
        rr, cc = draw.polygon(*zip(*verts))

    else:
        rr, cc = np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    return ctt(rr, dtype=INT), ctt(cc, dtype=INT)

@tf.function
def shift(inputs, shift, axes, pad_val = 0):
    assert len(shift) == len(axes)
    axis_shift = zip(axes, shift)
    axis2shift = dict(axis_shift)  
    old_shape = inputs.shape

    for axis in axis2shift:
        pad_shape = list(inputs.shape)
        pad_shape[axis] = abs(axis2shift[axis])
        input_pad = tf.fill(pad_shape, pad_val)
        inputs = tf.concat((inputs, input_pad), axis = axis) 
    
    
    input_roll = tf.roll(inputs, shift, axes)
    ret = tf.slice(input_roll, [0 for _ in range(len(old_shape))], old_shape)

    return ret    

class Fluid:
    def __init__(self, shape: tuple[int, int], current: pm.Vec2d, loc: pm.Vec2d = pm.Vec2d(0, 0), scale: float = 1.0):
        self.shape = shape
        self.current = current
        self.vel_i = tf.Variable(tf.cast(tf.fill(shape, current.x), dtype=FLOAT))
        self.vel_j = tf.Variable(tf.cast(tf.fill(shape, current.y), dtype=FLOAT))

        self.loc = loc # in PM-coordinates, position of (0, 0)
        self.scale = scale # 1 grid = how many PM units?

        self.dens = tf.Variable(tf.constant(np.full(shape, 1, dtype=np.float32), dtype=FLOAT))

    def run(self, steps: int, dt: float = 0.1) -> None:
        vel_i, vel_j, dens = self.vel_i, self.vel_j, self.dens
        
        for _ in range(steps):
            (vel_i, vel_j), dens = step(vel_i, vel_j, dens, tf.constant(dt, dtype=FLOAT))

        self.vel_i.assign(vel_i)
        self.vel_j.assign(vel_j)
        self.dens.assign(dens)

    def step(self, dt: float = 0.1) -> None:
        (vel_i, vel_j), dens = step(self.vel_i, self.vel_j, self.dens, tf.constant(dt, dtype=FLOAT))

        self.vel_i.assign(vel_i)
        self.vel_j.assign(vel_j)
        self.dens.assign(dens)

    def draw(self, surface: pg.Surface, num_arrows: int = 25) -> None:
        surface.fill([255, 255, 255])
        surf_arr = pg.surfarray.pixels3d(surface)
        surf_arr_scaled = np_grid(surf_arr, *self.shape)
        dens = self.dens.numpy()
        colors = (255 - 70 * dens[:, :, None, None, None]).astype(int)
        colors[colors < 0] = 0
        colors[colors > 255] = 255
        surf_arr_scaled[..., :-1] = colors

        vel_scaled = np.mean(np_grid(np.stack([self.vel_i.numpy(), self.vel_j.numpy()], axis=2) * self.dens.numpy()[:, :, None], num_arrows, num_arrows), axis=(2, 3))
        draw_vec_field(surface, vel_scaled, 0.5, [255, 0, 0])

    # "move" this field to be centered around position
    # pos is in PM-coordinates
    def center_on(self, pos: pm.Vec2d) -> None:
        i, j = util.to_int(pos) - self.loc
        axes = [0, 1]
        shifts = [int(i), int(j)]

        self.vel_i.assign(shift(self.vel_i, shifts, axes, ctt(self.current.x, dtype=FLOAT)))
        self.vel_j.assign(shift(self.vel_j, shifts, axes, ctt(self.current.y, dtype=FLOAT)))
        self.dens.assign(shift(self.dens, shifts, axes, ctt(1.0, dtype=FLOAT)))

        self.loc = util.to_int(pos)

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
            ctt(self.loc.x, dtype=FLOAT),
            ctt(self.loc.y, dtype=FLOAT),
        )
        force = pm.Vec2d(force_i.numpy(), force_j.numpy())
        obj.apply_force_at_world_point(force, obj.center_of_gravity)
        obj.torque += torque.numpy()
        self.vel_i.assign(vel_i)
        self.vel_j.assign(vel_j)

if __name__ == "__main__":
    # for i in range(250):
    #     for j in range(250):
    #         f.vel[i, j] = np.array([125 - j, i - 125], dtype=np.float64) * 0.5#  if np.sqrt((i - 125) ** 2 + (j - 125) ** 2) < 100 else [0, 0]
    # # f.vel[30: 70, 30: 70] = [10, 10]
    # f.dens[75: 175, 75: 175] = 10
    space = pm.Space()
    space.damping = 0.95
    wing = pm.Body(body_type=pm.Body.DYNAMIC)
    
    wing_shape = pm.Segment(wing, pm.Vec2d(-20, 0), pm.Vec2d(20, 0), 0.1)
    
    wing_shape.density = 10.0
    wing.position = pm.Vec2d(0, 0)
    wing.angular_velocity = 1
    wing.angle = 0.5

    space.add(wing, wing_shape)

    fl = Fluid([500, 500], pm.Vec2d(0, 0), pm.Vec2d(-250, -250), 1.0)
    
    pg.init()
    surface = pg.display.set_mode((1000, 1000))

    clock = pg.time.Clock()
    running = True
    i = 0
    while running:
        # if i >= 10:
        #     break
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        
        for _ in range(1):
            for _ in range(5):
                space.step(2 / 60)
                fl.run(1, 2 / 60)

                fl.apply_forces(wing, wing_shape, 0.3)

        fl.draw(surface, 40)
        print(str(wing.position), str(wing.angular_velocity))

        display_scale = surface.get_width() / fl.shape[0]

        # print(f)

        draw_util.draw_shape(surface, wing, wing_shape, display_scale)
        # a = wing.position * scale * display_scale + pm.Vec2d(*surface.get_size()) / 2
        # b = a + fl * scale * display_scale * 0.5
        # draw_util.draw_arrow(surface, a, b, [255, 255, 0])

        # pm_scale = scale * surface.get_width() / fl.shape[0]
        # for a, b in util.poly_sides(map(wing.local_to_world, wing_poly)):
        #     pg.draw.line(surface, [0, 0, 0], util.to_int(a * pm_scale), util.to_int(b * pm_scale))
        # pg.draw.line(
        #     surface, 
        #     [0, 0, 0], 
        #     util.to_int(a * surface.get_width() / fl.shape[0]), 
        #     util.to_int(b * surface.get_width() / fl.shape[0])
        # )
        # draw_util.draw_arrow(
        #     surface, 
        #     # util.to_int( / 2 * surface.get_width() / fl.shape[0]),
        #     u# til.to_int(((a + b) / 2 + 0.5 * f) * surface.get_width() / fl.shape[0]),
        #     [255, 255, 0],
        # )
        
        # flip() the display to put your work on screen
        pg.display.flip()

        # clock.tick(60)  # limits FPS to 60
        i += 1
    pg.quit()