import numpy as np
import pymunk as pm
import pygame as pg

import util
import draw_util
import fluid

def reflect(points: list[pm.Vec2d], scale: float) -> list[pm.Vec2d]:
    points = [pm.Vec2d(a, b) * scale for a, b in points]
    for point in reversed(points[1:-1]):
        points.append(pm.Vec2d(-point.x, point.y))
    return points

# useful for making sails and things
def create_links(a: pm.Body, pos_a: pm.Vec2d, b: pm.Body, pos_b: pm.Vec2d, num: int, width: float = 1.0) -> tuple[list[pm.Body], list[pm.Segment], list[pm.constraints.PivotJoint]]:
    points = [pos_a * (1 - i) + pos_b * i for i in np.linspace(0, 1, num)]
    bodies = [pm.Body() for _ in range(num)]
    segments = [pm.Segment(bodies[i], points[i], points[i + 1], width) for i in range(num)]
    bodies_w_endpoints = [a] + bodies + [b]
    joints = [pm.constraints.PivotJoint(bodies_w_endpoints[i], bodies_w_endpoints[i + 1], points[i]) for i in range(num + 1)]
    
    for joint in joints:
        joint.collide_bodies = False
    
    if a.space is not None:
        a.space.add(*bodies, *segments, *joints)
    
    return bodies, segments, joints

# for the rudder
class Hinge:
    def __init__(self, a: pm.Body, b: pm.Body, pos: pm.Vec2d, k: float = -0.5) -> None:
        self.a, self.b = a, b
        self.pos = pos

        self.tgt = 0.0

        self.pivot = pm.constraints.PivotJoint(a, b, pos)
        self.motor = pm.constraints.SimpleMotor(a, b, 0.0)
        self.motor.max_force = 100

        self.k = k

        self.pivot.collide_bodies = self.motor.collide_bodies = False
        self.motor.pre_solve = self.adjust

        if a.space is not None:
            a.space.add(self.pivot, self.motor)

    def adjust(self, motor: pm.constraints.SimpleMotor, space: pm.Space) -> None:
        # simple proportional controller
        motor.rate = self.k * (self.b.angle - self.a.angle)

class Water(pm.Space):
    AIR = 0
    SURF = 1
    DEEP = 2
    def __init__(
        self, 
        shape: tuple[int, int], 
        scale: float = 1.0,
        wind: pm.Vec2d = pm.Vec2d.zero(), 
        current: pm.Vec2d = pm.Vec2d.zero(),
        loc: pm.Vec2d = pm.Vec2d.zero(),
    ):
        super().__init__()
        self.shape, self.scale = shape, scale
        self.wind, self.current = wind, current
        self.air = fluid.Fluid(shape, wind, loc, scale)
        self.surf = fluid.Fluid(shape, current, loc, scale)
        self.deep = fluid.Fluid(shape, current, loc, scale)
        self.loc = loc

        self.main_obj = pm.Body()
        self.objects: dict[pm.Body, tuple[list[tuple[pm.Shape, float]], list[tuple[pm.Shape, float]], list[tuple[pm.Shape, float]]]] = {}

    def add_obj(
        self, 
        obj: pm.Body, 
        air: list[tuple[pm.Shape, float]] = [], 
        surf: list[tuple[pm.Shape, float]] = [], 
        deep: list[tuple[pm.Shape, float]] = []
    ) -> None:
        self.objects[obj] = air, surf, deep

    def step(self, dt: float):
        # any way to multi-thread this bit here?
        self.air.step(dt)
        self.surf.step(dt)
        self.deep.step(dt)
        for obj, data in self.objects.items():
            for (shape, weight), fld in zip(data, [self.air, self.surf, self.deep]):
                fld.apply_forces(obj, shape, weight)

        super().step(dt)

        i_shift = 0 if abs(self.main_obj.position.x - self.shape[0] // 2) < 5 else int(self.main_obj.position.x - self.shape[0] // 2)
        j_shift = 0 if abs(self.main_obj.position.y - self.shape[1] // 2) < 5 else int(self.main_obj.position.y - self.shape[1] // 2)
        
        for fld in [self.air, self.surf, self.deep]:
            fld.roll(-i_shift, -j_shift)
            
        for body in self.bodies:
            body.position = body.position - pm.Vec2d(i_shift, j_shift)

        self.air.induce(self.wind)
        self.surf.induce(self.current)
        self.deep.induce(self.current)

if __name__ == "__main__":
    space = Water((500, 500), 10.0, pm.Vec2d(0, -3))
    hull = pm.Body()
    r, s = 3, 40
    hull_poly = [[r * np.cos(i * 2 * np.pi / s), r * np.sin(i * 2 * np.pi / s)] for i in range(s)]
    # hull_poly = [
    #     pm.Vec2d(0.0, 2.5),
    #     pm.Vec2d(0.5, 2.0),
    #     pm.Vec2d(1.0, 0.5),
    #     pm.Vec2d(1.0, -1.5),
    #     pm.Vec2d(0.0, -2.0),
    # ]
    # hull_poly = reflect(hull_poly, scale=space.scale)
    
    hull_shape = pm.Poly(hull, hull_poly)

    space.add(hull, hull_shape)

    # keel = pm.Vec2d(0.0, 1.0), pm.Vec2d(0.0, -1.0), 1.0
    # rudder = pm.Body()
    # rudder_a, rudder_b = pm.Vec2d(0.0, -2.0), pm.Vec2d(0.0, -2.75)
    # rudder_a, rudder_b = rudder_a * space.scale, rudder_b * space.scale
    # rudder_shape = pm.Segment(rudder, rudder_a, rudder_b, 0.1)
    # rudder_hinge = Hinge(hull, rudder, rudder_a)
    space.add_obj(
        hull, 
        surf=[(a, b, 0.75) for a, b in util.poly_sides(hull_poly)],
        # deep=[keel]
    )
    # space.add_obj(
    #     rudder,
    #     surf=[(rudder_a, rudder_b, 0.75)],
    #     deep=[(rudder_a, rudder_b, 1.0)],
    # )

    # space.add(rudder, rudder_shape)

    # for body in space.bodies:
    #     for shape in body.shapes:
    #         shape.filter = pm.ShapeFilter(group=0)

    pg.init()

    surface = pg.display.set_mode((1000, 1000))

    clock = pg.time.Clock()
    running = True

    space.surf.vel[...] = [5, 0] + util.RNG.uniform(-1, 1, space.surf.vel.shape)

    def draw():
        shift = hull.position
        scale = surface.get_width() / space.shape[0]

        def pm_to_pg(pos: pm.Vec2d) -> pm.Vec2d:
            return util.to_int((hull.local_to_world(pos) - shift) * scale + pm.Vec2d(*surface.get_size()) * 0.5)
        
        surface.fill([255, 255, 255])

        space.surf.draw(surface, num_arrows=20)

        # draw surface water
        draw_util.draw_field(surface, util.normalize(space.surf.dens), [-1, -1, 0])
        
        # draw hull
        pg.draw.aalines(
            surface, 
            [0, 0, 0], 
            True, 
            [pm_to_pg(vert) for vert in hull_poly]
        )

        # # draw rudder
        # pg.draw.aaline(
        #     surface, 
        #     [0, 0, 0], 
        #     pm_to_pg(rudder_a),
        #     pm_to_pg(rudder_a + pm.Vec2d.rotated(rudder_b - rudder_a, rudder.angle - hull.angle)),
        # )
    
        pg.display.flip()
    # rudder.angle = 0.25 * np.pi
    draw()
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        
        space.step(0.6)
        draw()
        
        clock.tick(10)