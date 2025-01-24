import matplotlib.pyplot as plt
import numpy as np
import meshio

import create_mesh
import gtrans

# specimen dimensions:
L_TOT = 120.
L_0 = 100.
HEIGHT = 10.
WIDTH = 10.

# slicing parameters:
EXTRUSION_WIDTH = 0.45
N_PERIMETERS = 3

# dimensions in deformed configuration
et = N_PERIMETERS * EXTRUSION_WIDTH
l = 20
a = 7
b = 3
h = 7

# dimensions in original configuration:
L = 40 - et
H = 10
H1 = 8
A = 12 - 2 * et
A1 = 5
C = 3
C1 = 2
D = 7
B = 4
B1 = 6

POINTS_DEFO = np.array([
    # ### INNER POINTS ###
    # right
    [0, 0, 0],
    [0, -(2 * h**2)**.5, 0],
    [(2 * h**2)**.5, 0, 0],
    [l / 2**.5, -(2 * h**2)**.5 - l / 2**.5, 0],
    [(2 * h**2)**.5 + l / 2**.5, -l / 2**.5, 0], # 4
    [(l + a) / 2**.5, -(2 * h**2)**.5 - (l + a) / 2**.5, 0],
    [(2 * h**2)**.5 + (l + a) / 2**.5, -(l + a) / 2**.5, 0],
    [(2 * h**2)**.5 + (b + l) / 2**.5, -(l - b) / 2**.5, 0],
    [(2 * h**2)**.5 + (a + b + l) / 2**.5, -(l - b + a) / 2**.5, 0],
    [h / 2**.5, -h / 2**.5, 0], # 9
    [(l + h) / 2**.5, -(h + l) / 2**.5, 0],
    [(l + h + a) / 2**.5, -(h + l + a) / 2**.5, 0],
    # left:
    [-(2 * h**2)**.5, 0, 0],
    [-l / 2**.5, -(2 * h**2)**.5 - l / 2**.5, 0],
    [-(2 * h**2)**.5 - l / 2**.5, -l / 2**.5, 0], # 14
    [-(2 * h**2)**.5 - (b + l) / 2**.5, -(l - b) / 2**.5, 0],
    [-(2 * h**2)**.5 - (a + b + l) / 2**.5, -(l - b + a) / 2**.5, 0],
    [-(2 * h**2)**.5 - (l + a) / 2**.5, -(l + a) / 2**.5, 0],
    [-(l + h + a) / 2**.5, -(h + l + a) / 2**.5, 0],
    [-(l + a) / 2**.5, -(2 * h**2)**.5 - (l + a) / 2**.5, 0],
    [-h / 2**.5, -h / 2**.5, 0], # 20
    [-(l + h) / 2**.5, -(h + l) / 2**.5, 0],
    # ### OUTER POINTS ###
    [0, et, 0],
    [(2 * h**2)**.5 + et * np.sin(np.pi / 8), et, 0],
    [(2 * h**2)**.5 + l / 2**.5, -l / 2**.5 + et * 2**.5, 0],
    [(2 * h**2)**.5 + (b + l) / 2**.5, -(l - b) / 2**.5 + et * 2**.5, 0], # 25
    [(2 * h**2)**.5 + (a + b + l) / 2**.5 + et * 2**.5, -(l - b + a) / 2**.5, 0],
    [(2 * h**2)**.5 + (l + a + et) / 2**.5, -(l + a + et) / 2**.5, 0],
    [(l + h + a + et) / 2**.5, -(h + l + a + et) / 2**.5, 0],
    [(l + a) / 2**.5, -(2 * h**2)**.5 - (l + a) / 2**.5 - et * 2**.5, 0],
    [(l - et) / 2**.5, -(2 * h**2)**.5 - (et + l) / 2**.5, 0], # 30
    [0, -(2 * h**2)**.5 - et * 2**.5, 0],
    [-(l - et) / 2**.5, -(2 * h**2)**.5 - (et + l) / 2**.5, 0],
    [-(l + a) / 2**.5, -(2 * h**2)**.5 - (l + a) / 2**.5 - et * 2**.5, 0],
    [-(l + h + a + et) / 2**.5, -(h + l + a + et) / 2**.5, 0],
    [-(2 * h**2)**.5 - (l + a + et) / 2**.5, -(l + a + et) / 2**.5, 0],
    [-(2 * h**2)**.5 - (a + b + l) / 2**.5 - et * 2**.5, -(l - b + a) / 2**.5, 0],
    [-(2 * h**2)**.5 - (b + l) / 2**.5, -(l - b) / 2**.5 + et * 2**.5, 0],
    [-(2 * h**2)**.5 - l / 2**.5, -l / 2**.5 + et * 2**.5, 0],
    [-(2 * h**2)**.5 - et * np.sin(np.pi / 8), et, 0],
])

POINTS_ORIG = np.array([
    # ### INNER POINTS ###
    # right
    [0, 0, 0],
    [0, -A, 0],
    [H, 0, 0],
    [L - B - C - D, -A, 0],
    [L - B1, 0, 0],
    [L - B - .75 * D - C1, -A, 0], # 5
    [L - B, -A, 0],
    [L, 0, 0],
    [L, -A, 0],
    [H1, -A + A1, 0],
    [L - B - D, -A1, 0], # 10
    [L - B - .5 * D, -A, 0],
    # left
    [-H, 0, 0],
    [-L + B + C + D, -A, 0],
    [-L + B1, 0, 0],
    [-L, 0, 0], # 15
    [-L, -A, 0],
    [-L + B, -A, 0],
    [-L + B + .5 * D, -A, 0],
    [-L + B + .75 * D + C1, -A, 0],
    [-H1, A1 - A, 0], # 20
    [-L + B + D, -A1, 0],
    # ### OUTER POINTS ###
    [0, et, 0],
    [H, et, 0],
    [L - B1, et, 0],
    [L + et, et, 0], # 25
    [L + et, -A - et, 0],
    [L - B, -A - et, 0],
    [L - B - .5 * D, -A - et, 0],
    [L - B - .75 * D - C1, -A - et, 0],
    [L - B - D - C, -A - et, 0], # 30
    [0, -A - et, 0],
    [-L + B + D + C, -A - et, 0],
    [-L + B + .75 * D + C1, -A - et, 0],
    [-L + B + .5 * D, -A - et, 0],
    [-L + B, -A - et, 0], # 35
    [-L - et, -A - et, 0],
    [-L - et, et, 0],
    [-L + B1, et, 0],
    [-H, et, 0],
])

CELLS = [
    ('triangle', [
        [0, 1, 9],
        [0, 9, 2],
        [12, 20, 0],
        [20, 1, 0],
    ],),
    ('quad', [
        # ### INNER ###
        # [0, 1, 9, 2],
        [1, 3, 10, 9],
        [9, 10, 4, 2],
        [3, 5, 11, 10],
        [10, 11, 6, 4],
        [4, 6, 8, 7],
        # [12, 20, 1, 0],
        [21, 13, 1, 20],
        [14, 21, 20, 12],
        [18, 19, 13, 21],
        [17, 18, 21, 14],
        [16, 17, 14, 15],
        # ### OUTER ###
        [0, 2, 23, 22],
        [2, 4, 24, 23],
        [4, 7, 25, 24],
        [7, 8, 26, 25],
        [8, 6, 27, 26],
        [11, 28, 27, 6],
        [29, 28, 11, 5],
        [30, 29, 5, 3],
        [31, 30, 3, 1],
        [32, 31, 1, 13],
        [33, 32, 13, 19],
        [34, 33, 19, 18],
        [35, 34, 18, 17],
        [36, 35, 17, 16],
        [37, 36, 16, 15],
        [38, 37, 15, 14],
        [39, 38, 14, 12],
        [22, 39, 12, 0],
    ],)
]

DISPLACEMENT = POINTS_ORIG - POINTS_DEFO
EXTRUSION_MULTIPLIER = np.ones(DISPLACEMENT.shape[0]).reshape((-1, 1))

POINTS = np.vstack([POINTS_DEFO, POINTS_DEFO + np.array([0, 0, 10])])
PTS_PER_LAYER  = POINTS_DEFO.shape[0]
CELLS_3D = [
    ('wedge', [
        cell + [pt + PTS_PER_LAYER for pt in cell]
        for cell in CELLS[0][1]]
     ),
    ('hexahedron', [
        cell + [pt + PTS_PER_LAYER for pt in cell]
        for cell in CELLS[1][1]]
     ),
]

def main(output_file=create_mesh.get_filename('v02_beam_3D.vtk')):
    mesh = meshio.Mesh(
        POINTS, CELLS_3D,
        point_data={
            'displacement' : np.vstack([DISPLACEMENT, DISPLACEMENT]),
            'extrusion' : np.vstack([
                EXTRUSION_MULTIPLIER, EXTRUSION_MULTIPLIER]),
        },
    )
    mesh.write(output_file)

def main_2d(output_file=create_mesh.get_filename('v02_beam_2D.vtk')):
    mesh = meshio.Mesh(
        POINTS_DEFO, CELLS,
        point_data={
            'displacement' : DISPLACEMENT,
            'extrusion' : EXTRUSION_MULTIPLIER,
        },
    )
    mesh.write(output_file)

def plot_mesh(points, cells, ax=None):
    if ax is None:
        ax = plt

    ax.plot(points[:, 0], points[:, 1], '.')
    ax.axis('equal')
    ax.grid(True)

    for ii, coors in enumerate(points):
        ax.annotate(f'{ii}', xy=coors[:2])

    for cell in sum([cc[1] for cc in cells], []):
        coors = points[cell + cell[:1], :2].T
        ax.plot(coors[0], coors[1], 'b--')

if __name__ == '__main__':
    main_2d()
    main()
    create_mesh.convert_to_stl(
        create_mesh.get_filename('v02_beam_3D.vtk'),
        create_mesh.get_filename('v02_beam_3D.stl'),
    )

    fig, axes = plt.subplots(nrows=2, sharex=True)

    plot_mesh(POINTS_DEFO, CELLS, ax=axes[0])
    plot_mesh(POINTS_ORIG, CELLS, ax=axes[1])
    plt.show()
