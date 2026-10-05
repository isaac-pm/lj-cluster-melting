"""Build a small Lennard-Jones cluster."""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pyvista as pv


@dataclass
class Coordinates:
    x: float
    y: float
    z: float

    def squared_distance_from_origin(self) -> float:
        return self.x**2 + self.y**2 + self.z**2


@dataclass
class Particle:
    id: int
    coordinates: Coordinates


def bcc_lattice_sites(half_width: int, lattice_constant: float) -> list[Coordinates]:
    """List the sites of a body-centred cubic lattice around the origin.

    A body-centred cubic lattice is a simple cubic grid with one extra site
    in the centre of every cube.
    """
    sites = []
    for i in range(-half_width, half_width + 1):
        for j in range(-half_width, half_width + 1):
            for k in range(-half_width, half_width + 1):
                corner = Coordinates(i, j, k)
                centre = Coordinates(i + 0.5, j + 0.5, k + 0.5)
                for site in (corner, centre):
                    sites.append(
                        Coordinates(
                            site.x * lattice_constant,
                            site.y * lattice_constant,
                            site.z * lattice_constant,
                        )
                    )
    return sites


def build_cluster(num_particles: int, lattice_constant: float = 1.0) -> list[Particle]:
    """Build a compact cluster of particles on a body-centred cubic lattice.

    Keeps the lattice sites closest to the origin, so the cluster grows in
    shells around a central particle.
    """
    half_width = math.ceil(num_particles ** (1 / 3))  # big enough to hold the cluster
    sites = bcc_lattice_sites(half_width, lattice_constant)
    sites.sort(key=Coordinates.squared_distance_from_origin)

    particles = []
    for i in range(num_particles):
        particle = Particle(i, sites[i])
        particles.append(particle)
    return particles


def show_cluster(particles: list[Particle], radius: float = 0.25) -> None:
    """Open a 3D window with one sphere per particle."""
    points = np.array(
        [[p.coordinates.x, p.coordinates.y, p.coordinates.z] for p in particles]
    )
    spheres = pv.PolyData(points).glyph(
        geom=pv.Sphere(radius=radius), scale=False, orient=False
    )

    plotter = pv.Plotter()
    plotter.add_mesh(spheres, color="steelblue", smooth_shading=True)
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="view the particles in 3D")
    args = parser.parse_args()

    particles = build_cluster(13)
    for particle in particles:
        print(particle)
    if args.show:
        show_cluster(particles)
