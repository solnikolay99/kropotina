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
        """
        Split current cell to 8 sub-cells
        """
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
        """
        Move particle to 1 of 8 sub-cells
        :param coords: coords of particle
        :param m: mass of particle
        """
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
        """
        Put particle to the cell of tree structure
        :param coords: coords of particle
        :param m: mass of particle
        """
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


def gravity(point1: list[float], point2: list[float], m1: float, m2: float) -> Vector:
    """
    Calculate the force of gravity
    :param point1: point 1
    :param point2: point 2
    :param m1: mass of point 1
    :param m2: mass of point 2
    :return: the force of gravity
    """
    r2 = (point1[0] - point2[0]) * (point1[0] - point2[0]) + \
         (point1[1] - point2[1]) * (point1[1] - point2[1]) + \
         (point1[2] - point2[2]) * (point1[2] - point2[2])
    if r2 == 0:
        return Vector(0.0, 0.0, 0.0)
    r3_inv = m1 * m2 / (r2 * np.sqrt(r2))

    f_gr = Vector(-G * (point1[0] - point2[0]) * r3_inv,
                  -G * (point1[1] - point2[1]) * r3_inv,
                  -G * (point1[2] - point2[2]) * r3_inv)
    return f_gr


def calculate_by_direct_sum(points: list[PointDS], point: list[float]):
    """
    Calculate the force of gravity by Direct sum method
    :param points: list of all points in the system
    :param point: desired point
    """
    for i in range(len(points)):
        if points[i].x != point[0] or points[i].y != point[1] or points[i].z != point[2]:
            continue
        for j in range(i + 1, len(points)):
            points[i].f_gr += gravity([points[i].x, points[i].y, points[i].z],
                                      [points[j].x, points[j].y, points[j].z],
                                      points[i].m, points[j].m)
    pass


def convert_to_tree(points: list[PointDS]) -> Tree:
    """
    Convert list of whole points to Tree by tree method
    :param points: list of all points in the system
    :return: tree
    """
    tree = Tree(1.0)
    for i in range(len(points)):
        tree.put_particle([points[i].x, points[i].y, points[i].z], points[i].m)
    return tree


def check_lr(cell: Cell, point: list[float], lr: float) -> bool:
    """
    Calculate the criteria for opening a node
    :param cell: cell from tree structure
    :param point: desired point
    :param lr: criteria for opening a node
    :return: True if calculated value < lr, False - otherwise
    """
    L = cell.size
    D = np.sqrt((cell.mass_c.x - point[0]) * (cell.mass_c.x - point[0])
                + (cell.mass_c.y - point[1]) * (cell.mass_c.y - point[1])
                + (cell.mass_c.z - point[2]) * (cell.mass_c.z - point[2]))
    if D == 0:
        return True
    LD = L / D
    return LD < lr


def calculate_by_tree(tree: Tree, point: list[float], lr: float) -> Vector:
    """
    Calculate the force of gravity by Tree method
    :param tree: tree with all points in the system
    :param point: desired point
    :param lr: criteria for opening a tree node
    :return: the force of gravity
    """
    if check_lr(tree.cur_cell, point, lr):
        # include whole cell mass
        return gravity(point, [tree.cur_cell.mass_c.x,
                               tree.cur_cell.mass_c.y,
                               tree.cur_cell.mass_c.z],
                       point[3], tree.cur_cell.M)
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
                                   tree.cur_cell.mass_c.z],
                           point[3], tree.cur_cell.M)
        return f_gr
    pass


def print_ds_result(point: PointDS):
    print(f"Fgr by Direct sum method for point [{point.x}, {point.y}, {point.z}]"
          f" is [{point.f_gr.x}, {point.f_gr.y}, {point.f_gr.z}]")
    pass


def print_tree_result(point: list[float], f_gr: Vector):
    print(f"Fgr by Tree method for point [{point[0]}, {point[1]}, {point[2]}]"
          f" is [{f_gr.x}, {f_gr.y}, {f_gr.z}]")
    pass


def print_compare_gravity(point_by_ds: PointDS, f_gr_by_tree: Vector):
    accuracy_x = np.abs(1 - point_by_ds.f_gr.x / f_gr_by_tree.x) * 100
    accuracy_y = np.abs(1 - point_by_ds.f_gr.y / f_gr_by_tree.y) * 100
    accuracy_z = np.abs(1 - point_by_ds.f_gr.z / f_gr_by_tree.z) * 100
    print(f"Accuracy between Direct sum and Tree methods is"
          f" [{accuracy_x:.2f}%, {accuracy_y:.2f}%, {accuracy_z:.2f}%]")
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

    desired_point = [0.0, 0.0, 0.0, 1.0]  # x, y, z coords and m of desired point

    start_time = time.time()
    calculate_by_direct_sum(main_points, desired_point)
    print(f"Execution time of Direct sum method is {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    main_tree = convert_to_tree(main_points)
    print(f"Execution time of converting to tree is {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    f_gravity = calculate_by_tree(main_tree, desired_point, 0.5)
    print(f"Execution time of Tree method is {time.time() - start_time:.4f} seconds")

    print_ds_result(main_points[0])
    print_tree_result(desired_point, f_gravity)
    print_compare_gravity(main_points[0], f_gravity)

    pass
