"""Geometry unit tests: projection round trip, lattice planning around an obstacle, failed-move memory."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embagent.geometry import OccupancyMap, fwd_right, pixel_to_world, plan_pose, world_to_pixel  # noqa: E402


def test_fwd_right_conventions():
    f, r = fwd_right(0)
    assert np.allclose(f, [0, 1]) and np.allclose(r, [1, 0])
    f, r = fwd_right(90)  # RotateRight from yaw 0 faces +x
    assert np.allclose(f, [1, 0], atol=1e-9) and np.allclose(r, [0, -1], atol=1e-9)


def test_projection_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        pose = {"x": rng.uniform(-3, 3), "z": rng.uniform(-3, 3), "yaw": float(rng.choice([0, 90, 180, 270])),
                "horizon": float(rng.choice([0, 30, -30]))}
        u, v, d = rng.uniform(10, 490), rng.uniform(10, 490), rng.uniform(0.5, 6)
        x, z, h = pixel_to_world(u, v, d, pose)
        u2, v2, d2 = world_to_pixel(x, z, h, pose)
        assert abs(u2 - u) < 1e-6 and abs(v2 - v) < 1e-6 and abs(d2 - d) < 1e-6


def _wall_map():
    m = OccupancyMap((0.0, 0.0), half_extent=4.0)
    m.free[:] = 1  # everything observed as floor
    i0, j0 = m.cell(-1.0, 1.0)
    i1, _ = m.cell(1.0, 1.0)
    m.occ[i0:i1, j0:j0 + 2] = 5  # wall from x=-1..1 at z=1
    return m


def test_plan_goes_around_wall():
    m = _wall_map()
    acts, states, resid = plan_pose(m, (0.0, 0.0), 0, (0.0, 2.0), max_cost=30)
    assert resid <= 0.55
    for p, _ in states:  # never crosses the wall band
        assert not (-1.2 < p[0] < 1.2 and 0.8 < p[1] < 1.3)


def test_failed_move_not_retried():
    m = OccupancyMap((0.0, 0.0), half_extent=4.0)
    m.free[:] = 1
    m.add_failed_move((0.0, 0.0), (0.0, 1.0))  # moving +z from the origin was rejected
    acts, states, _ = plan_pose(m, (0.0, 0.0), 0, (0.0, 1.5), max_cost=20)
    assert acts and acts[0] != 0  # first action is not "forward" (+z at yaw 0)


def test_projection_round_trip_non_square():
    rng = np.random.default_rng(1)
    for _ in range(200):
        pose = {"x": rng.uniform(-3, 3), "z": rng.uniform(-3, 3), "yaw": rng.uniform(0, 360), "horizon": 30.0}
        u, v, d = rng.uniform(10, 630), rng.uniform(10, 470), rng.uniform(0.5, 6)
        x, z, hgt = pixel_to_world(u, v, d, pose, fov=63.45, cam_h=0.88, img_w=640, img_h=480)
        u2, v2, d2 = world_to_pixel(x, z, hgt, pose, fov=63.45, cam_h=0.88, img_w=640, img_h=480)
        assert abs(u2 - u) < 1e-6 and abs(v2 - v) < 1e-6 and abs(d2 - d) < 1e-6


def test_heading_planner_30deg_around_wall():
    from embagent.geometry import plan_heading
    m = _wall_map()
    acts, states, resid = plan_heading(m, (0.0, 0.0), 0.0, (0.0, 2.0), rot=30, max_cost=60)
    assert resid <= 0.55 and set(acts) <= {"F", "R", "L"}
    for p, y in states:
        assert not (-1.2 < p[0] < 1.2 and 0.8 < p[1] < 1.3)
        assert abs((y / 30) - round(y / 30)) < 1e-6
