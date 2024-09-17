import os
import random

max_length = 1  # max x coord
count_el = 100  # count points


class Cell:
    size = 0  # size
    x_c = 0.0  # coordinate of centre
    x_m = 0.0  # coordinate of mass center
    x = 0.0  # point coordinate
    M = 0.0  # summary mass
    p_ngp = 0.0  # density NGP
    p_cic = 0.0  # density CIC
    p_cic_overlap = 1.0  # overlap coefficient for cell for CIC algorithm
    p_cic_adj = 0.0  # adjustment for density CIC from nearby cells

    def __init__(self, size: float, s_c_shift=0.0):
        self.size = size
        self.x_c = abs(s_c_shift + size * 0.5)
        pass


class Tree:
    cur_cell = None  # link to current cell
    rel_left = None  # link to left tree
    rel_right = None  # link to right tree

    def __init__(self, size: float, s_c_shift=0.0):
        self.cur_cell = Cell(size=size, s_c_shift=s_c_shift)
        pass

    def put_particle(self, x: float, m: float):
        if self.cur_cell.M == 0 and self.rel_left is None and self.rel_right is None:
            self.cur_cell.x = x
            self.cur_cell.p_cic_overlap = 1 - abs(self.cur_cell.x - self.cur_cell.x_c) / self.cur_cell.size
            self.cur_cell.p_cic = self.cur_cell.p_cic_overlap * m / self.cur_cell.size
        else:
            # split current cell to 2 cells and move curr point in one of the new cells
            if self.rel_left is None:
                self.rel_left = Tree(self.cur_cell.size * 0.5, -self.cur_cell.x_c)
                self.rel_right = Tree(self.cur_cell.size * 0.5, self.cur_cell.x_c)
                if self.cur_cell.x < self.cur_cell.x_c:
                    self.rel_left.put_particle(self.cur_cell.x, self.cur_cell.M)
                else:
                    self.rel_right.put_particle(self.cur_cell.x, self.cur_cell.M)
                self.cur_cell.x = 0.0
                self.cur_cell.p_cic = 0.0
                self.cur_cell.p_cic_overlap = 1.0
            # place new point in one of the cells
            if x < self.cur_cell.x_c:
                self.rel_left.put_particle(x, m)
            else:
                self.rel_right.put_particle(x, m)
        self.cur_cell.M += m
        self.cur_cell.p_ngp = self.cur_cell.M / self.cur_cell.size
        pass

    def get_particles(self) -> list[Cell]:
        if self.rel_left is None:
            return [self.cur_cell]
        else:
            return self.rel_left.get_particles() + self.rel_right.get_particles()
        pass


def calc_dens_ngp(main_tree: Tree) -> float:
    densities = [particle.p_ngp for particle in main_tree.get_particles()]
    mean = sum(densities) / len(densities)
    dispersion = sum([(density - mean) * (density - mean) for density in densities]) / (len(densities) - 1)
    return dispersion


def update_density_cic(main_tree: Tree, right_border: float):
    particles = main_tree.get_particles()
    length = len(particles)
    for i in range(len(particles)):
        cell = particles[i]
        if cell.x < cell.x_c:
            border = cell.x - cell.size
            if border < 0:
                border = right_border + border
            for j in range(1, length):
                left_cell = particles[i - j]
                if border < (left_cell.x_c + left_cell.size * 0.5):
                    left_cell.p_cic_adj += cell.M * (1 - cell.p_cic_overlap) / left_cell.size
                else:
                    break
        elif cell.x > cell.x_c:
            border = cell.x + cell.size
            if border > right_border:
                border = border - right_border
            for j in range(1, length):
                index = i + j
                if index >= length:
                    index = index - length
                right_cell = particles[index]
                if border > (right_cell.x_c - right_cell.size * 0.5):
                    right_cell.p_cic_adj += cell.M * (1 - cell.p_cic_overlap) / right_cell.size
                else:
                    break
    pass


def calc_dens_cic(main_tree: Tree, right_border: float) -> float:
    update_density_cic(main_tree, right_border)
    densities = [particle.p_cic + particle.p_cic_adj for particle in main_tree.get_particles()]
    mean = sum(densities) / len(densities)
    dispersion = sum([(density - mean) * (density - mean) for density in densities]) / (len(densities) - 1)
    return dispersion


def dump_cell(cell: Cell, tabular='', cell_position='centre') -> str:
    formated_str = f"| size={cell.size:.4f} x_c={cell.x_c:.4f} x_m={cell.x_m:.4f}" \
                   f" x={cell.x:.4f} M={cell.M:.0f} p_ngp={cell.p_ngp:.4f} p_cic={cell.p_cic:.4f}" \
                   f" p_cic_overlap={cell.p_cic_overlap:.4f} p_cic_adj={cell.p_cic_adj:.4f} |"
    top_row = f"-----{cell_position}"
    top_row = top_row + "-" * (len(formated_str) - len(top_row))
    bottom_row = "-" * len(formated_str)
    return f"{tabular}{top_row}\n" \
           f"{tabular}{formated_str}\n" \
           f"{tabular}{bottom_row}\n"


def dump_to_file(tree_for_dump: Tree, tabular='', position='centre'):
    output.write(dump_cell(tree_for_dump.cur_cell, tabular, position))
    tabular += '    '
    if tree_for_dump.rel_left is not None:
        dump_to_file(tree_for_dump.rel_left, tabular, ' left ')
    if tree_for_dump.rel_right is not None:
        dump_to_file(tree_for_dump.rel_right, tabular, ' right')
    pass


if __name__ == '__main__':
    tree = Tree(max_length)

    '''
    for _ in range(count_el):
        x_coord = random.random() * max_length
        tree.put_particle(x_coord, 1)
    '''
    for x_coord in [0.7, 0.27, 0.3]:
        tree.put_particle(x_coord, 1)

    density_ngp = calc_dens_ngp(tree)
    density_cic = calc_dens_cic(tree, max_length)

    os.makedirs("temp", exist_ok=True)
    with open('temp/Tree_dump.txt', 'w') as output:
        dump_to_file(tree)
        output.write(f"\nDensity: NGP={density_ngp:.4f} CIC={density_cic:.4f}")

    pass
