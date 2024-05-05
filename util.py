import typing

import numpy as np
import pymunk as pm

floor = np.floor
ceil = np.ceil

RNG = np.random.default_rng()

R90 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.int32)


def frac(x: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    return x - floor(x)


def proj_mat(vec):
    vec /= np.hypot(*vec)
    i, j = vec
    return np.array(
        [
            [i**2, i * j],
            [i * j, j**2],
        ],
        dtype=np.float64,
    )


def apply_mat(
    mat: np.ndarray[np.float32], args: np.ndarray[np.float32]
) -> np.ndarray[np.float32]:
    return args @ np.transpose(mat)


def grid(arr: np.ndarray[np.float32], *tgt_shape: list[int]) -> np.ndarray:
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


def poly_sides(verts: list[pm.Vec2d]) -> typing.Iterable[tuple[pm.Vec2d, pm.Vec2d]]:
    verts = list(verts)
    if len(verts) >= 2:
        yield verts[-1], verts[0]
        for i in range(1, len(verts)):
            yield verts[i - 1], verts[i]


def to_int(pos: pm.Vec2d) -> pm.Vec2d:
    return pm.Vec2d(int(pos[0]), int(pos[1]))


def project(
    a: np.ndarray[np.float32], b: np.ndarray[np.float32]
) -> np.ndarray[np.float32]:
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return b * (np.dot(a, b) / np.dot(b, b))[..., None]


def normalize(arr: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    min_val, max_val = min(0, np.min(arr, axis=None)), max(1, np.max(arr, axis=None))
    return (arr - min_val) / (max_val - min_val)
