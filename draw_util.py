import pygame as pg
import pymunk as pm
import numpy as np

import util


def draw_arrow(
    surface: pg.Surface,
    start: pg.Vector2,
    end: pg.Vector2,
    color: pg.Color,
    body_width: int = 2,
    head_width: int = 4,
    head_height: int = 2,
):
    """Draw an arrow between start and end with the arrow head at the end.

    Args:
        surface (pygame.Surface): The surface to draw on
        start (pygame.Vector2): Start position
        end (pygame.Vector2): End position
        color (pygame.Color): Color of the arrow
        body_width (int, optional): Defaults to 2.
        head_width (int, optional): Defaults to 4.
        head_height (float, optional): Defaults to 2.
    """
    start = pg.Vector2(list(start))
    end = pg.Vector2(list(end))
    arrow = start - end
    angle = arrow.angle_to(pg.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # Create the triangle head around the origin
    head_verts = [
        pg.Vector2(0, head_height / 2),  # Center
        pg.Vector2(head_width / 2, -head_height / 2),  # Bottomright
        pg.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
    ]
    # Rotate and translate the head into place
    translation = pg.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
    for i in range(len(head_verts)):
        head_verts[i].rotate_ip(-angle)
        head_verts[i] += translation
        head_verts[i] += start

    pg.draw.polygon(surface, color, head_verts)

    # Stop weird shapes when the arrow is shorter than arrow head
    if arrow.length() >= head_height:
        # Calculate the body rect, rotate and translate into place
        body_verts = [
            pg.Vector2(-body_width / 2, body_length / 2),  # Topleft
            pg.Vector2(body_width / 2, body_length / 2),  # Topright
            pg.Vector2(body_width / 2, -body_length / 2),  # Bottomright
            pg.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
        ]
        translation = pg.Vector2(0, body_length / 2).rotate(-angle)
        for i in range(len(body_verts)):
            body_verts[i].rotate_ip(-angle)
            body_verts[i] += translation
            body_verts[i] += start

        pg.draw.polygon(surface, color, body_verts)


def draw_vec_field(
    surface: pg.Surface,
    field: np.ndarray[np.float32],
    arrow_scale: int = 10,
    color: tuple[float, float, float] = [0, 0, 0],
) -> None:
    i_scale = surface.get_width() / field.shape[0]
    j_scale = surface.get_height() / field.shape[1]

    center_shfit = pg.Vector2(i_scale / 2, j_scale / 2)

    for i, j in np.ndindex(field.shape[:2]):
        di, dj = field[i, j]
        draw_arrow(
            surface,
            center_shfit + pg.Vector2(np.floor(i * i_scale), np.floor(j * j_scale)),
            center_shfit
            + pg.Vector2(
                np.floor((i + di * arrow_scale) * i_scale),
                np.floor((j + dj * arrow_scale) * j_scale),
            ),
            color,
        )


def draw_field(
    surface: pg.Surface,
    field: np.ndarray[np.float32],
    color: tuple[float, float, float] = [0, 0, 1],
) -> None:
    arr = util.grid(pg.surfarray.pixels3d(surface), *field.shape)
    arr[...] += np.floor(
        255.0
        * field[:, :, None, None, None]
        * np.array(color, dtype=np.float32)[None, None, None, None, :]
    ).astype(np.uint8)


def draw_shape(
    surface: pg.Surface,
    body: pm.Body,
    shape: pm.Shape,
    scale: float = 1.0,
    color: tuple[int, int, int] = [0, 0, 0],
) -> None:
    def pm_to_pg(pos: pm.Vec2d) -> pm.Vec2d:
        return body.local_to_world(pos) * scale + pm.Vec2d(*surface.get_size()) / 2

    if isinstance(shape, pm.Circle):
        pg.draw.circle(
            surface, color, util.to_int(pm_to_pg(shape.offset)), shape.radius * scale
        )

    elif isinstance(shape, pm.Segment):
        pg.draw.line(surface, color, pm_to_pg(shape.a), pm_to_pg(shape.b))

    elif isinstance(shape, pm.Poly):
        verts = [pm_to_pg(vert) for vert in shape.get_vertices()]
        pg.draw.lines(surface, color, True, verts)
