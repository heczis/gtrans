"""
Create a three-part cube with gradient infill
"""
import matplotlib.pyplot as plt
import meshio
import numpy as np
import os

import vtk_from_stl as vfs

N_PERIMETERS = 1
EXTRUSION_WIDTH = 0.45
SHELL_WIDTH = N_PERIMETERS * EXTRUSION_WIDTH

WIDTH, HEIGHT = 20., 20.
MID_COEF = 0.3

COORS = np.array([
    # outer points
    [-WIDTH / 2, HEIGHT / 2],
    [WIDTH / 2, HEIGHT / 2],
    [WIDTH / 2, MID_COEF * HEIGHT / 2],
    [WIDTH / 2, -MID_COEF * HEIGHT / 2],
    [WIDTH / 2, -HEIGHT / 2],
    [-WIDTH / 2, - HEIGHT / 2],
    [-WIDTH / 2, -MID_COEF * HEIGHT / 2],
    [-WIDTH / 2, MID_COEF * HEIGHT / 2],
    # inner points
    [-WIDTH / 2 + SHELL_WIDTH, HEIGHT / 2 - SHELL_WIDTH],
    [WIDTH / 2 - SHELL_WIDTH, HEIGHT / 2 - SHELL_WIDTH],
    [WIDTH / 2 - SHELL_WIDTH, MID_COEF * HEIGHT / 2],
    [WIDTH / 2 - SHELL_WIDTH, -MID_COEF * HEIGHT / 2],
    [WIDTH / 2 - SHELL_WIDTH, -HEIGHT / 2 + SHELL_WIDTH],
    [-WIDTH / 2 + SHELL_WIDTH, - HEIGHT / 2 + SHELL_WIDTH],
    [-WIDTH / 2 + SHELL_WIDTH, -MID_COEF * HEIGHT / 2],
    [-WIDTH / 2 + SHELL_WIDTH, MID_COEF * HEIGHT / 2],
])

def fix_cell(cell, coors=None):
    """
    Check the orientation of a cell's vertices (clockwise/counter-) and reverse
    the order if clockwise.
    """
    if coors is None:
        coors = COORS

    is_ccw = vfs.is_clockwise(coors[list(cell) + [cell[0],]])
    if is_ccw > 0:
        return cell
    elif is_ccw < 0:
        print(f'Fixing orientation of cell {cell} to {cell[::-1]}.')
        return cell[::-1]
    else:
        raise ValueError(
            f'Cannot determine element orientation ({cell}).')

CELLS = [
    fix_cell(cc) for cc in [
        [0, 1, 9, 8], [1, 2, 10, 9], [2, 3, 11, 10], [3, 4, 12, 11],
        [4, 5, 13, 12], [5, 6, 14, 13][::-1], [6, 7, 15, 14], [7, 0, 8, 15],
        [8, 9, 10, 15], [15, 10, 11, 14], [14, 11, 12, 13],
    ]
]

DISP_COEF = 1.
DISPLACEMENT_OUTER = DISP_COEF * np.array([
    [-WIDTH / 2 + SHELL_WIDTH, HEIGHT * (1 - MID_COEF) / 2 - SHELL_WIDTH],
    [WIDTH / 2 - SHELL_WIDTH, HEIGHT * (1 - MID_COEF) / 2 - SHELL_WIDTH],
    [WIDTH / 2 - SHELL_WIDTH, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [-WIDTH / 2 + SHELL_WIDTH, 0],
])

def main(output_base_name='cube'):
    mesh = meshio.Mesh(
        COORS, (('quad', CELLS),),
    )

    outline_edges = vfs.get_outline_edges(mesh)
    outline_path = vfs.get_path(outline_edges)
    outline_coors = (
        DISPLACEMENT_OUTER + mesh.points[:len(DISPLACEMENT_OUTER)]
    )[outline_path, :2]
    displacement_inner = DISPLACEMENT_OUTER + vfs.offset_outline(
        outline_coors, offset=-SHELL_WIDTH)[:-1]

    inner_coors = mesh.points[:len(DISPLACEMENT_OUTER)] + displacement_inner

    print(f'displacement_inner:\n{displacement_inner}')

    displacement = np.vstack([
        DISPLACEMENT_OUTER, inner_coors - mesh.points[len(DISPLACEMENT_OUTER):]])

    defo_mesh = meshio.Mesh(
        COORS + displacement,
        (('quad', [cc[::-1] for cc in CELLS]),),
        point_data={
            'displacement' : -displacement,
            'extrusion' : np.ones(len(COORS)),
        },
    )
    defo_mesh.write(f'{output_base_name}.vtk', binary=False)

    vfs.convert_to_stl(f'{output_base_name}.vtk', f'{output_base_name}.stl')

if __name__ == '__main__':
    main(os.path.join(os.path.dirname(__file__), 'cube'))
