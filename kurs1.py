import os
import time

import numpy as np

G = 6.67e-11  # gravitation constant

max_size = 1.0  # max size of cube in any direction


class Vector:
    def __init__(self, x: float, y: float, z: float):
        self.x = x  # x direction
        self.y = y  # y direction
        self.z = z  # z direction
        pass

    def __add__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self


class Point:
    def __init__(self, x: float, y: float, z: float):
        self.x = x  # x coordinate
        self.y = y  # y coordinate
        self.z = z  # z coordinate
        pass


class PointDS(Point):
    def __init__(self, x: float, y: float, z: float, m: float):
        super().__init__(x, y, z)
        self.m = m  # mass
        self.f_gr = Vector(0.0, 0.0, 0.0)  # F gravitation
        pass


class Cell:
    size = 0  # size
    center = Point(0.0, 0.0, 0.0)  # coordinate of cell center
    point = None  # coordinate of point
    M = 0.0  # summary mass
    mass_c = None  # coordinate of center mass
    f_gr = Vector(0.0, 0.0, 0.0)  # F gravitation

    def __init__(self, size: float, s_c_shift: list[float]):
        self.size = size
        self.center = Point(abs(s_c_shift[0] + size * 0.5),
                            abs(s_c_shift[1] + size * 0.5),
                            abs(s_c_shift[2] + size * 0.5))
        pass


class Tree:
    cur_cell = None  # link to current cell
    root_cell = None  # link to parent cell
    rel_xl_yl_zl = None  # link to left x, left y, left z tree
    rel_xr_yl_zl = None  # link to right x, left y, left z tree
    rel_xl_yr_zl = None  # link to left x, right y, left z tree
    rel_xr_yr_zl = None  # link to right, right y, left z tree
    rel_xl_yl_zr = None  # link to left x, left y, right z tree
    rel_xr_yl_zr = None  # link to right x, left y, right z tree
    rel_xl_yr_zr = None  # link to left x, right y, right z tree
    rel_xr_yr_zr = None  # link to right, right y, right z tree

    def __init__(self, size: float, s_c_shift=None, rel_root=None):
        if s_c_shift is None:
            s_c_shift = [0.0, 0.0, 0.0]
        self.cur_cell = Cell(size=size, s_c_shift=s_c_shift)
        self.root_cell = rel_root
        pass

    def _split_cur_cell(self):
        self.rel_xl_yl_zl = Tree(self.cur_cell.size * 0.5,
                                 [-self.cur_cell.center.x, -self.cur_cell.center.y, -self.cur_cell.center.z],
                                 self.cur_cell)
        self.rel_xr_yl_zl = Tree(self.cur_cell.size * 0.5,
                                 [self.cur_cell.center.x, -self.cur_cell.center.y, -self.cur_cell.center.z],
                                 self.cur_cell)
        self.rel_xl_yr_zl = Tree(self.cur_cell.size * 0.5,
                                 [-self.cur_cell.center.x, self.cur_cell.center.y, -self.cur_cell.center.z],
                                 self.cur_cell)
        self.rel_xr_yr_zl = Tree(self.cur_cell.size * 0.5,
                                 [self.cur_cell.center.x, self.cur_cell.center.y, -self.cur_cell.center.z],
                                 self.cur_cell)
        self.rel_xl_yl_zr = Tree(self.cur_cell.size * 0.5,
                                 [-self.cur_cell.center.x, -self.cur_cell.center.y, self.cur_cell.center.z],
                                 self.cur_cell)
        self.rel_xr_yl_zr = Tree(self.cur_cell.size * 0.5,
                                 [self.cur_cell.center.x, -self.cur_cell.center.y, self.cur_cell.center.z],
                                 self.cur_cell)
        self.rel_xl_yr_zr = Tree(self.cur_cell.size * 0.5,
                                 [-self.cur_cell.center.x, self.cur_cell.center.y, self.cur_cell.center.z],
                                 self.cur_cell)
        self.rel_xr_yr_zr = Tree(self.cur_cell.size * 0.5,
                                 [self.cur_cell.center.x, self.cur_cell.center.y, self.cur_cell.center.z],
                                 self.cur_cell)
        pass

    def _put_particle(self, coords: list[float], m: float):
        x_l_r = 0 if coords[0] < self.cur_cell.center.x else 1
        y_l_r = 0 if coords[1] < self.cur_cell.center.y else 1
        z_l_r = 0 if coords[2] < self.cur_cell.center.z else 1
        if x_l_r == 0 and y_l_r == 0 and z_l_r == 0:
            self.rel_xl_yl_zl.put_particle(coords, m)
        elif x_l_r == 1 and y_l_r == 0 and z_l_r == 0:
            self.rel_xr_yl_zl.put_particle(coords, m)
        elif x_l_r == 0 and y_l_r == 1 and z_l_r == 0:
            self.rel_xl_yr_zl.put_particle(coords, m)
        elif x_l_r == 1 and y_l_r == 1 and z_l_r == 0:
            self.rel_xr_yr_zl.put_particle(coords, m)
        elif x_l_r == 0 and y_l_r == 0 and z_l_r == 1:
            self.rel_xl_yl_zr.put_particle(coords, m)
        elif x_l_r == 1 and y_l_r == 0 and z_l_r == 1:
            self.rel_xr_yl_zr.put_particle(coords, m)
        elif x_l_r == 0 and y_l_r == 1 and z_l_r == 1:
            self.rel_xl_yr_zr.put_particle(coords, m)
        elif x_l_r == 1 and y_l_r == 1 and z_l_r == 1:
            self.rel_xr_yr_zr.put_particle(coords, m)
        pass

    def put_particle(self, coords: list[float], m: float):
        if self.cur_cell.M == 0.0 and self.rel_xl_yl_zl is None:
            self.cur_cell.point = Point(coords[0], coords[1], coords[2])
            self.cur_cell.mass_c = Point(coords[0], coords[1], coords[2])
        else:
            # split current cell to 8 cells and move curr point in one of the new cells
            if self.rel_xl_yl_zl is None:
                self._split_cur_cell()
                self._put_particle([self.cur_cell.point.x,
                                    self.cur_cell.point.y,
                                    self.cur_cell.point.z],
                                   self.cur_cell.M)
                self.cur_cell.point = None
            # place new point in one of the cells
            self._put_particle(coords, m)
        self.cur_cell.mass_c.x = (self.cur_cell.mass_c.x * self.cur_cell.M + coords[0] * m) / (self.cur_cell.M + m)
        self.cur_cell.mass_c.y = (self.cur_cell.mass_c.y * self.cur_cell.M + coords[1] * m) / (self.cur_cell.M + m)
        self.cur_cell.mass_c.z = (self.cur_cell.mass_c.z * self.cur_cell.M + coords[2] * m) / (self.cur_cell.M + m)
        self.cur_cell.M += m
        pass

    def find_cell_by_point(self, coords: list[float]):
        if self.rel_xl_yl_zl is None:
            if self.cur_cell.point.x == coords[0] and \
                    self.cur_cell.point.y == coords[1] and \
                    self.cur_cell.point.z == coords[2]:
                return self
            else:
                raise Exception('Cell not found')
        else:
            x_l_r = 0 if coords[0] < self.cur_cell.center.x else 1
            y_l_r = 0 if coords[1] < self.cur_cell.center.y else 1
            z_l_r = 0 if coords[2] < self.cur_cell.center.z else 1
            if x_l_r == 0 and y_l_r == 0 and z_l_r == 0:
                return self.rel_xl_yl_zl.find_cell_by_point(coords)
            elif x_l_r == 1 and y_l_r == 0 and z_l_r == 0:
                return self.rel_xr_yl_zl.find_cell_by_point(coords)
            elif x_l_r == 0 and y_l_r == 1 and z_l_r == 0:
                return self.rel_xl_yr_zl.find_cell_by_point(coords)
            elif x_l_r == 1 and y_l_r == 1 and z_l_r == 0:
                return self.rel_xr_yr_zl.find_cell_by_point(coords)
            elif x_l_r == 0 and y_l_r == 0 and z_l_r == 1:
                return self.rel_xl_yl_zr.find_cell_by_point(coords)
            elif x_l_r == 1 and y_l_r == 0 and z_l_r == 1:
                return self.rel_xr_yl_zr.find_cell_by_point(coords)
            elif x_l_r == 0 and y_l_r == 1 and z_l_r == 1:
                return self.rel_xl_yr_zr.find_cell_by_point(coords)
            elif x_l_r == 1 and y_l_r == 1 and z_l_r == 1:
                return self.rel_xr_yr_zr.find_cell_by_point(coords)
        pass


def gravity(point1: list[float], point2: list[float]) -> Vector:
    r2 = (point1[0] - point2[0]) * (point1[0] - point2[0]) + \
         (point1[1] - point2[1]) * (point1[1] - point2[1]) + \
         (point1[2] - point2[2]) * (point1[2] - point2[2])
    if r2 == 0:
        return Vector(0.0, 0.0, 0.0)
    r3_inv = 1.0 / (r2 * np.sqrt(r2))

    f_gr = Vector(-G * (point1[0] - point2[0]) * r3_inv,
                  -G * (point1[1] - point2[1]) * r3_inv,
                  -G * (point1[2] - point2[2]) * r3_inv)
    return f_gr


def calculate_by_direct_sum(points: list[PointDS], point: list[float]):
    for i in range(len(points)):
        if points[i].x != point[0] or points[i].y != point[1] or points[i].z != point[2]:
            continue
        for j in range(i + 1, len(points)):
            points[i].f_gr += gravity([points[i].x, points[i].y, points[i].z],
                                      [points[j].x, points[j].y, points[j].z])
    pass


def dump_cell(cell: Cell, tabular='', cell_position='centre') -> str:
    formated_str = f"| size={cell.size:.4f} center=[{cell.center.x:.4f}, {cell.center.y:.4f}, {cell.center.z:.4f}]" \
                   f" m={cell.M:.4f}"
    if cell.point is not None:
        formated_str += f" point=[{cell.point.x:.4f}, {cell.point.y:.4f}, {cell.point.z:.4f}]"
    if cell.mass_c is not None:
        formated_str += f" mass_c=[{cell.mass_c.x:.4f}, {cell.mass_c.y:.4f}, {cell.mass_c.z:.4f}]"
    formated_str += " |"
    top_row = f"-----{cell_position}"
    top_row = top_row + "-" * (len(formated_str) - len(top_row))
    bottom_row = "-" * len(formated_str)
    return f"{tabular}{top_row}\n" \
           f"{tabular}{formated_str}\n" \
           f"{tabular}{bottom_row}\n"


def dump_to_file(output_f, tree_for_dump: Tree, tabular='', position='centre'):
    output_f.write(dump_cell(tree_for_dump.cur_cell, tabular, position))
    tabular += '    '
    if tree_for_dump.rel_xl_yl_zl is not None:
        dump_to_file(output_f, tree_for_dump.rel_xl_yl_zl, tabular, ' left left left ')
    if tree_for_dump.rel_xr_yl_zl is not None:
        dump_to_file(output_f, tree_for_dump.rel_xr_yl_zl, tabular, ' right left left')
    if tree_for_dump.rel_xl_yr_zl is not None:
        dump_to_file(output_f, tree_for_dump.rel_xl_yr_zl, tabular, ' left right left ')
    if tree_for_dump.rel_xr_yr_zl is not None:
        dump_to_file(output_f, tree_for_dump.rel_xr_yr_zl, tabular, ' right right left')
    if tree_for_dump.rel_xl_yl_zr is not None:
        dump_to_file(output_f, tree_for_dump.rel_xl_yl_zr, tabular, ' left left right ')
    if tree_for_dump.rel_xr_yl_zr is not None:
        dump_to_file(output_f, tree_for_dump.rel_xr_yl_zr, tabular, ' right left right')
    if tree_for_dump.rel_xl_yr_zr is not None:
        dump_to_file(output_f, tree_for_dump.rel_xl_yr_zr, tabular, ' left right right ')
    if tree_for_dump.rel_xr_yr_zr is not None:
        dump_to_file(output_f, tree_for_dump.rel_xr_yr_zr, tabular, ' right right right')
    pass


def convert_to_tree(points: list[PointDS]) -> Tree:
    tree = Tree(1.0)
    for i in range(len(points)):
        tree.put_particle([points[i].x, points[i].y, points[i].z], points[i].m)
    return tree


def check_lr(cell: Cell, point: list[float], lr: float) -> bool:
    L = cell.size
    D = np.sqrt((cell.mass_c.x - point[0]) * (cell.mass_c.x - point[0])
                + (cell.mass_c.y - point[1]) * (cell.mass_c.y - point[1])
                + (cell.mass_c.z - point[2]) * (cell.mass_c.z - point[2]))
    if D == 0:
        return True
    LD = L / D
    return LD < lr


def calculate_by_tree(tree: Tree, point: list[float], lr: float):
    if check_lr(tree.cur_cell, point, lr):
        # include whole cell mass
        return gravity(point, [tree.cur_cell.mass_c.x,
                               tree.cur_cell.mass_c.y,
                               tree.cur_cell.mass_c.z])
    else:
        # step in and resolve refs
        f_gr = Vector(0.0, 0.0, 0.0)
        if tree.rel_xl_yl_zl is not None:
            f_gr += calculate_by_tree(tree.rel_xl_yl_zl, point, lr)
            f_gr += calculate_by_tree(tree.rel_xr_yl_zl, point, lr)
            f_gr += calculate_by_tree(tree.rel_xl_yr_zl, point, lr)
            f_gr += calculate_by_tree(tree.rel_xr_yr_zl, point, lr)
            f_gr += calculate_by_tree(tree.rel_xl_yl_zr, point, lr)
            f_gr += calculate_by_tree(tree.rel_xr_yl_zr, point, lr)
            f_gr += calculate_by_tree(tree.rel_xl_yr_zr, point, lr)
            f_gr += calculate_by_tree(tree.rel_xr_yr_zr, point, lr)
        else:
            f_gr = gravity(point, [tree.cur_cell.mass_c.x,
                                   tree.cur_cell.mass_c.y,
                                   tree.cur_cell.mass_c.z])
        return f_gr
    pass


def print_ds_result(point: PointDS):
    print(f"Fgr by Direct sum method for point [{point.x}, {point.y}, {point.z}]"
          f" is [{point.f_gr.x}, {point.f_gr.y}, {point.f_gr.z}]")
    pass


def print_tree_result(point: list[float], f_gr: Vector):
    print(f"Fgr by Direct sum method for point [{point[0]}, {point[1]}, {point[2]}]"
          f" is [{f_gr.x}, {f_gr.y}, {f_gr.z}]")
    pass


if __name__ == '__main__':
    main_points = []
    for a in range(32):
        for b in range(32):
            for c in range(32):
                main_points.append(PointDS(a * max_size / 31,
                                           b * max_size / 31,
                                           c * max_size / 31,
                                           1))

    desired_point = [0.0, 0.0, 0.0]

    start_time = time.time()
    calculate_by_direct_sum(main_points, desired_point)
    print(f"Execution time of Direct sum method is {time.time() - start_time:.4f} seconds")

    main_tree = convert_to_tree(main_points)

    start_time = time.time()
    f_gravity = calculate_by_tree(main_tree, desired_point, 0.5)
    print(f"Execution time of Tree method is {time.time() - start_time:.4f} seconds")

    os.makedirs("temp", exist_ok=True)
    with open('temp/Tree_dump.txt', 'w') as output:
        dump_to_file(output, main_tree)

    print_ds_result(main_points[0])
    print_tree_result(desired_point, f_gravity)

    pass
