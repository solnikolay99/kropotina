import math
import os
import random

max_length = 20  # max x coord
cell_size = 1  # cell size
count_el = 100  # count points
mass = 1  # default mass for points


def calc_dens_ngp(points: list[float]) -> (list[float], float):
    points = sorted(points)
    densities = [0 for _ in range(math.ceil(max_length / cell_size))]
    for point in points:
        i = int(point / cell_size)
        densities[i] += mass
    mean = sum(densities) / len(densities)
    dispersion = sum([(density - mean) * (density - mean) for density in densities]) / (len(densities) - 1)
    return densities, dispersion


def calc_dens_cic(points: list[float]) -> (list[float], float):
    points = sorted(points)
    densities = [0 for _ in range(math.ceil(max_length / cell_size))]
    for point in points:
        i = int(point / cell_size)
        overlap = point - i
        densities[i] += mass * overlap
        if overlap < 0.5:
            densities[i - 1] += mass * (1 - overlap)
        else:
            k = len(densities)
            if i + 1 == len(densities):
                densities[0] += mass * (1 - overlap)
            else:
                densities[i + 1] += mass * (1 - overlap)
    mean = sum(densities) / len(densities)
    dispersion = sum([(density - mean) * (density - mean) for density in densities]) / (len(densities) - 1)
    return densities, dispersion


if __name__ == '__main__':
    particles = []

    for _ in range(count_el):
        x_coord = random.random() * max_length
        particles.append(x_coord)

    density_ngp, dispersion_ngp = calc_dens_ngp(particles)
    density_cic, dispersion_cic = calc_dens_cic(particles)

    density_ngp = [f"{density:.4f}" for density in density_ngp]
    density_cic = [f"{density:.4f}" for density in density_cic]

    os.makedirs("temp", exist_ok=True)
    with open('temp/Density_dump.txt', 'w') as output:
        output.write(f"\nDensity NGP={density_ngp}")
        output.write(f"\nDensity CIC={density_cic}")
        output.write(f"\nDispersion: NGP={dispersion_ngp:.4f} CIC={dispersion_cic:.4f}")

    pass
