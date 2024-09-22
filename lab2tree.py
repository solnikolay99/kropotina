import os


class Cell:
    size = 0  # size
    x_c = 0.0  # coordinate of centre
    x_m = 0.0  # coordinate of mass center
    x = 0.0  # x-coord of point
    M = 0.0  # summary mass

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


def dump_cell(cell: Cell, tabular='', cell_position='centre') -> str:
    return f"{tabular}--------------------{cell_position}-----------------------------\n" \
           f"{tabular}| size={cell.size:.4f} x_c={cell.x_c:.4f} x_m={cell.x_m:.4f} x={cell.x:.4f} M={cell.M:.0f} |\n" \
           f"{tabular}-------------------------------------------------------\n"


def dump_to_file(tree_for_dump: Tree, tabular='', position='centre'):
    output.write(dump_cell(tree_for_dump.cur_cell, tabular, position))
    tabular += '    '
    if tree_for_dump.rel_left is not None:
        dump_to_file(tree_for_dump.rel_left, tabular, ' left ')
    if tree_for_dump.rel_right is not None:
        dump_to_file(tree_for_dump.rel_right, tabular, ' right')

    pass


if __name__ == '__main__':
    tree = Tree(1.0)

    for x_coord in [0.7, 0.27, 0.3]:
        tree.put_particle(x_coord, 1)

    os.makedirs("temp", exist_ok=True)
    with open('temp/Tree_dump.txt', 'w') as output:
        dump_to_file(tree)

    pass
