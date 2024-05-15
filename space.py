import numpy as np
import pymunk as pm
import pygame as pg
import tensorflow as tf
from tensorflow import convert_to_tensor as ctt

import util
import fluid
import draw_util

from fluid import INT, FLOAT


class FluidSpace(pm.Space):
    class Layer:
        def __init__(
            self,
            shape: tuple[int, int],
            parent: "FluidSpace",
            current: pm.Vec2d = pm.Vec2d.zero(),
            density: float = 1.0,
        ):
            # self.current, self.density = current, density
            self.vel_i = tf.Variable(
                tf.fill(shape, ctt(current.x, FLOAT))
                + tf.random.uniform(shape, -0.1, 0.1, dtype=FLOAT)
            )
            self.vel_j = tf.Variable(
                tf.fill(shape, ctt(current.y, FLOAT))
                + tf.random.uniform(shape, -0.1, 0.1, dtype=FLOAT)
            )
            self.dens = tf.Variable(tf.fill(shape, ctt(density, FLOAT)))
            self.bg_vel_i = tf.constant(current.x, dtype=FLOAT)
            self.bg_vel_j = tf.constant(current.y, dtype=FLOAT)
            self.bg_dens = tf.constant(density, dtype=FLOAT)
            self.parent = parent
            # self.step(10.0)

        @tf.function
        def step(self, dt: tf.Tensor) -> None:
            # tf.Assert(tf.math.is_nan(tf.reduce_sum(self.vel_i)), [ctt(0)])
            # tf.Assert(tf.math.is_nan(tf.reduce_sum(self.vel_j)), [ctt(1)])
            # tf.Assert(tf.math.is_nan(tf.reduce_sum(self.dens)), [ctt(2)])
            # tf.Assert(tf.math.is_nan((self.bg_vel_i)), [ctt(3)])
            # tf.Assert(tf.math.is_nan((self.bg_vel_j)), [ctt(4)])
            # tf.Assert(tf.math.is_nan(self.bg_dens), [ctt(5)])
            # tf.Assert(tf.math.is_nan(dt), [ctt(6)])
            vel_i, vel_j, dens = fluid.step(
                self.vel_i,
                self.vel_j,
                self.dens,
                self.bg_vel_i,
                self.bg_vel_j,
                self.bg_dens,
                dt,
            )
            self.vel_i.assign(fluid.hide_the_evidence(vel_i))
            self.vel_j.assign(fluid.hide_the_evidence(vel_j))
            self.dens.assign(fluid.hide_the_evidence(dens))
            # self.vel_i.assign(vel_i)
            # self.vel_j.assign(vel_j)
            # self.dens.assign(dens)

        @tf.function
        def shift(self, i: tf.Tensor, j: tf.Tensor) -> None:
            self.vel_i.assign(fluid.shift(self.vel_i, i, j, self.bg_vel_i))
            self.vel_j.assign(fluid.shift(self.vel_j, i, j, self.bg_vel_j))
            self.dens.assign(fluid.shift(self.dens, i, j, self.bg_dens))

        @tf.function(reduce_retracing=True)
        def get_forces(
            self,
            rr: tf.Tensor,
            cc: tf.Tensor,
            pos_i: tf.Tensor,
            pos_j: tf.Tensor,
            vel_i: tf.Tensor,
            vel_j: tf.Tensor,
            avel: tf.Tensor,
            weight: tf.Tensor,
        ) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
            (vel_i, vel_j), ((force_i, force_j), torque) = fluid.get_forces(
                self.vel_i,
                self.vel_j,
                self.dens,
                pos_i,
                pos_j,
                vel_i,
                vel_j,
                avel,
                weight,
                rr,
                cc,
                self.parent.scale_tf,
                self.parent.loc_tf_i,
                self.parent.loc_tf_j,
            )
            self.vel_i.assign(fluid.hide_the_evidence(vel_i))
            self.vel_j.assign(fluid.hide_the_evidence(vel_j))
            return (force_i, force_j), torque

        def draw(self, surface: pg.Surface, num_arrows: int = 25) -> None:
            surface.fill([255, 255, 255])
            surf_arr = pg.surfarray.pixels3d(surface)
            surf_arr_scaled = fluid.np_grid(surf_arr, *self.parent.shape)
            dens = self.dens.numpy()
            colors = (255 - 70 * dens[:, :, None, None, None]).astype(int)
            colors[colors < 0] = 0
            colors[colors > 255] = 255
            surf_arr_scaled[..., :-1] = colors

            vel_scaled = np.mean(
                fluid.np_grid(
                    np.stack([self.vel_i.numpy(), self.vel_j.numpy()], axis=2)
                    * self.dens.numpy()[:, :, None],
                    num_arrows,
                    num_arrows,
                ),
                axis=(2, 3),
            )
            draw_util.draw_vec_field(surface, vel_scaled, 0.25, [255, 0, 0])

    def __init__(
        self,
        shape: tuple[int, int],
        fluids: set[str],
        currents: dict[str, pm.Vec2d],
        densities: dict[str, float],
        scale: float = 1.0,
        loc: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.scale = scale
        self.scale_tf = tf.constant(scale, dtype=float)

        if loc is not None:
            self.loc = loc
        else:
            self.loc = -self.shape[0] // 2, -self.shape[1] // 2

        self.loc_tf_i = tf.constant(self.loc[0], dtype=FLOAT)
        self.loc_tf_j = tf.constant(self.loc[1], dtype=FLOAT)

        self.fluids = {
            name: self.Layer(
                shape,
                self,
                currents.get(name, pm.Vec2d.zero()),
                densities.get(name, 1.0),
            )
            for name in fluids
        }

        self.weights: dict[pm.Shape, dict[str, FluidSpace.Layer]] = {}

    def step(self, dt: float):
        super().step(dt)
        dt = ctt(dt, dtype=FLOAT)

        for layer in self.fluids.values():
            layer.step(dt)

        for shape, weights in self.weights.items():
            body: pm.Body = shape.body
            rr, cc = fluid.shape_to_rrcc(body, shape, self.scale, self.loc)
            pos_i, pos_j = (
                ctt(body.position.x, dtype=FLOAT),
                ctt(body.position.y, dtype=FLOAT),
            )
            vel_i, vel_j = (
                ctt(body.velocity.x, dtype=FLOAT),
                ctt(body.velocity.y, dtype=FLOAT),
            )
            avel = ctt(body.angular_velocity, dtype=FLOAT)
            frc = pm.Vec2d.zero()
            trq = 0.0
            for name, layer in self.fluids.items():
                weight = ctt(weights[name], dtype=FLOAT)
                (force_i, force_j), torque = layer.get_forces(
                    rr, cc, pos_i, pos_j, vel_i, vel_j, avel, weight
                )
                frc += pm.Vec2d(force_i.numpy(), force_j.numpy())
                trq += torque.numpy()

            body.torque += trq
            body.apply_force_at_world_point(frc, body.position)

    def draw(self, surface: pg.Surface) -> None:
        self.fluids["water"].draw(surface)
        # for shape in self.weights:
        #     draw_util.draw_shape(surface, shape.body, shape, self.scale)

    def pm_to_grid(self, pos: pm.Vec2d) -> pm.Vec2d:
        return (pos - self.loc) / self.scale

    def pm_to_idx(self, pos: pm.Vec2d) -> tuple[float, float]:
        return util.to_int(self.pm_to_grid(pos))

    def grid_to_pm(self, pos: pm.Vec2d) -> pm.Vec2d:
        return pos * self.scale + self.loc

    def add_shape(self, shape: pm.Shape, weights: dict[str, float]) -> None:
        self.weights[shape] = weights


if __name__ == "__main__":
    space = FluidSpace(
        (500, 500), {"water"}, {"water": pm.Vec2d(0.0, 0.0)}, {}, 1.0, (-250, -250)
    )
    # space.fluids["water"].dens[100: 200, 100: 200].assign(tf.fill((100, 100), ctt(1.0, dtype=FLOAT)))
    # space.fluids["water"].vel_i[100: 200, 100: 200].assign(tf.fill((100, 100), ctt(2.0, dtype=FLOAT)))
    # space.fluids["water"].vel_j[100: 200, 100: 200].assign(tf.fill((100, 100), ctt(2.0, dtype=FLOAT)))
    space.damping = 0.95
    wing = pm.Body(body_type=pm.Body.DYNAMIC)

    wing_shape = pm.Segment(wing, pm.Vec2d(-30, 0), pm.Vec2d(30, 0), 0.1)

    wing_shape.density = 1.0
    wing.position = pm.Vec2d(0, 0)
    wing.velocity = pm.Vec2d(-0.5, 0.03)
    wing.angular_velocity = 1
    wing.angle = 0.5

    space.add(wing, wing_shape)
    space.add_shape(wing_shape, {"water": 0.75})

    pg.init()
    surface = pg.display.set_mode((1000, 1000))

    clock = pg.time.Clock()
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        for _ in range(1):
            space.step(1 / 30)


        space.draw(surface)

        draw_util.draw_shape(
            surface, wing, wing_shape, surface.get_width() / space.shape[0]
        )

        pg.display.flip()

        # clock.tick(60)  # limits FPS to 60
    pg.quit()
