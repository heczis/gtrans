"""
This script can do the following:
* Create the mesh for beam design (slicing and further manipulation).
* Transform [uniform] gcode.
* Change extrusion rate according to given field.

Currently, all parameters are fixed as constants contained in this script.
"""
import argparse
import copy
import os

import matplotlib.pyplot as plt
import numpy as np

import meshio
import sfepy.discrete as sd
import sfepy.discrete.fem as sdf
import vtk

import gtrans

# specimen dimensions:
L_TOT = 120.
L_0 = 100.
HEIGHT = 10.
WIDTH = 10.

# slicing parameters:
EXTRUSION_WIDTH = 0.45
MAX_EXTRUSION_WIDTH = 0.45
N_PERIMETERS = 3

# transformation parameters:
T_0 = EXTRUSION_WIDTH * N_PERIMETERS
T_1 = (MAX_EXTRUSION_WIDTH - EXTRUSION_WIDTH) * N_PERIMETERS
L_SUPP_2 = 5.

# mesh constants:
L_SUPP = L_TOT - L_0 - 2 * T_0 - 2.

POINTS = [
    # width 0
    # row 0
    [-L_TOT / 2, -HEIGHT / 2, 0],
    [-(L_0 + L_SUPP) / 2, -HEIGHT / 2, 0],
    [-L_0 / 2, -HEIGHT / 2, 0],
    [-(L_0 - L_SUPP) / 2, -HEIGHT / 2, 0],
    [-L_SUPP / 2, -HEIGHT / 2, 0],
    [0, -HEIGHT / 2, 0],
    [L_SUPP / 2, -HEIGHT / 2, 0],
    [(L_0 - L_SUPP) / 2, -HEIGHT / 2, 0],
    [L_0 / 2, -HEIGHT / 2, 0],
    [(L_0 + L_SUPP) / 2, -HEIGHT / 2, 0],
    [L_TOT / 2, -HEIGHT / 2, 0],
    # row 1
    [-L_TOT / 2 + T_0, -HEIGHT / 2 + T_0, 0],
    [-(L_0 + L_SUPP) / 2, -HEIGHT / 2 + T_0, 0],
    [-L_0 / 2, -HEIGHT / 2 + T_0, 0],
    [-(L_0 - L_SUPP) / 2, -HEIGHT / 2 + T_0, 0],
    [-L_SUPP / 2, -HEIGHT / 2 + T_0, 0],
    [0, -HEIGHT / 2 + T_0, 0],
    [L_SUPP / 2, -HEIGHT / 2 + T_0, 0],
    [(L_0 - L_SUPP) / 2, -HEIGHT / 2 + T_0, 0],
    [L_0 / 2, -HEIGHT / 2 + T_0, 0],
    [(L_0 + L_SUPP) / 2, -HEIGHT / 2 + T_0, 0],
    [L_TOT / 2 - T_0, -HEIGHT / 2 + T_0, 0],
    # row 2
    [-L_TOT / 2 + T_0, HEIGHT / 2 - T_0, 0],
    [-(L_0 + L_SUPP) / 2, HEIGHT / 2 - T_0, 0],
    [-L_0 / 2, HEIGHT / 2 - T_0, 0],
    [-(L_0 - L_SUPP) / 2, HEIGHT / 2 - T_0, 0],
    [-L_SUPP / 2, HEIGHT / 2 - T_0, 0],
    [0, HEIGHT / 2 - T_0, 0],
    [L_SUPP / 2, HEIGHT / 2 - T_0, 0],
    [(L_0 - L_SUPP) / 2, HEIGHT / 2 - T_0, 0],
    [L_0 / 2, HEIGHT / 2 - T_0, 0],
    [(L_0 + L_SUPP) / 2, HEIGHT / 2 - T_0, 0],
    [L_TOT / 2 - T_0, HEIGHT / 2 - T_0, 0],
    # row 3
    [-L_TOT / 2, HEIGHT / 2, 0],
    [-(L_0 + L_SUPP) / 2, HEIGHT / 2, 0],
    [-L_0 / 2, HEIGHT / 2, 0],
    [-(L_0 - L_SUPP) / 2, HEIGHT / 2, 0],
    [-L_SUPP / 2, HEIGHT / 2, 0],
    [0, HEIGHT / 2, 0],
    [L_SUPP / 2, HEIGHT / 2, 0],
    [(L_0 - L_SUPP) / 2, HEIGHT / 2, 0],
    [L_0 / 2, HEIGHT / 2, 0],
    [(L_0 + L_SUPP) / 2, HEIGHT / 2, 0],
    [L_TOT / 2, HEIGHT / 2, 0],
    # width 1
    # row 0
    [-L_TOT / 2, -HEIGHT / 2, WIDTH],
    [-(L_0 + L_SUPP) / 2, -HEIGHT / 2, WIDTH],
    [-L_0 / 2, -HEIGHT / 2, WIDTH],
    [-(L_0 - L_SUPP) / 2, -HEIGHT / 2, WIDTH],
    [-L_SUPP / 2, -HEIGHT / 2, WIDTH],
    [0, -HEIGHT / 2, WIDTH],
    [L_SUPP / 2, -HEIGHT / 2, WIDTH],
    [(L_0 - L_SUPP) / 2, -HEIGHT / 2, WIDTH],
    [L_0 / 2, -HEIGHT / 2, WIDTH],
    [(L_0 + L_SUPP) / 2, -HEIGHT / 2, WIDTH],
    [L_TOT / 2, -HEIGHT / 2, WIDTH],
    # row 1
    [-L_TOT / 2 + T_0, -HEIGHT / 2 + T_0, WIDTH],
    [-(L_0 + L_SUPP) / 2, -HEIGHT / 2 + T_0, WIDTH],
    [-L_0 / 2, -HEIGHT / 2 + T_0, WIDTH],
    [-(L_0 - L_SUPP) / 2, -HEIGHT / 2 + T_0, WIDTH],
    [-L_SUPP / 2, -HEIGHT / 2 + T_0, WIDTH],
    [0, -HEIGHT / 2 + T_0, WIDTH],
    [L_SUPP / 2, -HEIGHT / 2 + T_0, WIDTH],
    [(L_0 - L_SUPP) / 2, -HEIGHT / 2 + T_0, WIDTH],
    [L_0 / 2, -HEIGHT / 2 + T_0, WIDTH],
    [(L_0 + L_SUPP) / 2, -HEIGHT / 2 + T_0, WIDTH],
    [L_TOT / 2 - T_0, -HEIGHT / 2 + T_0, WIDTH],
    # row 2
    [-L_TOT / 2 + T_0, HEIGHT / 2 - T_0, WIDTH],
    [-(L_0 + L_SUPP) / 2, HEIGHT / 2 - T_0, WIDTH],
    [-L_0 / 2, HEIGHT / 2 - T_0, WIDTH],
    [-(L_0 - L_SUPP) / 2, HEIGHT / 2 - T_0, WIDTH],
    [-L_SUPP / 2, HEIGHT / 2 - T_0, WIDTH],
    [0, HEIGHT / 2 - T_0, WIDTH],
    [L_SUPP / 2, HEIGHT / 2 - T_0, WIDTH],
    [(L_0 - L_SUPP) / 2, HEIGHT / 2 - T_0, WIDTH],
    [L_0 / 2, HEIGHT / 2 - T_0, WIDTH],
    [(L_0 + L_SUPP) / 2, HEIGHT / 2 - T_0, WIDTH],
    [L_TOT / 2 - T_0, HEIGHT / 2 - T_0, WIDTH],
    # row 3
    [-L_TOT / 2, HEIGHT / 2, WIDTH],
    [-(L_0 + L_SUPP) / 2, HEIGHT / 2, WIDTH],
    [-L_0 / 2, HEIGHT / 2, WIDTH],
    [-(L_0 - L_SUPP) / 2, HEIGHT / 2, WIDTH],
    [-L_SUPP / 2, HEIGHT / 2, WIDTH],
    [0, HEIGHT / 2, WIDTH],
    [L_SUPP / 2, HEIGHT / 2, WIDTH],
    [(L_0 - L_SUPP) / 2, HEIGHT / 2, WIDTH],
    [L_0 / 2, HEIGHT / 2, WIDTH],
    [(L_0 + L_SUPP) / 2, HEIGHT / 2, WIDTH],
    [L_TOT / 2, HEIGHT / 2, WIDTH],
]

N_LAYER = len(POINTS) // 2

CELLS = [
    ('hexahedron', [
        [0, 1, 12, 11, N_LAYER, N_LAYER + 1, N_LAYER + 12, N_LAYER + 11],
        [1, 2, 13, 12, N_LAYER + 1, N_LAYER + 2, N_LAYER + 13, N_LAYER + 12],
        [2, 3, 14, 13, N_LAYER + 2, N_LAYER + 3, N_LAYER + 14, N_LAYER + 13],
        [3, 4, 15, 14, N_LAYER + 3, N_LAYER + 4, N_LAYER + 15, N_LAYER + 14],
        [4, 5, 16, 15, N_LAYER + 4, N_LAYER + 5, N_LAYER + 16, N_LAYER + 15],
        [5, 6, 17, 16, N_LAYER + 5, N_LAYER + 6, N_LAYER + 17, N_LAYER + 16],
        [6, 7, 18, 17, N_LAYER + 6, N_LAYER + 7, N_LAYER + 18, N_LAYER + 17],
        [7, 8, 19, 18, N_LAYER + 7, N_LAYER + 8, N_LAYER + 19, N_LAYER + 18],
        [8, 9, 20, 19, N_LAYER + 8, N_LAYER + 9, N_LAYER + 20, N_LAYER + 19],
        [9, 10, 21, 20, N_LAYER + 9, N_LAYER + 10, N_LAYER + 21, N_LAYER + 20],
        [0, 11, 22, 33, N_LAYER, N_LAYER + 11, N_LAYER + 22, N_LAYER + 33],
        [11, 12, 23, 22, N_LAYER + 11, N_LAYER + 12, N_LAYER + 23, N_LAYER + 22],
        [12, 13, 24, 23, N_LAYER + 12, N_LAYER + 13, N_LAYER + 24, N_LAYER + 23],
        [13, 14, 25, 24, N_LAYER + 13, N_LAYER + 14, N_LAYER + 25, N_LAYER + 24],
        [14, 15, 26, 25, N_LAYER + 14, N_LAYER + 15, N_LAYER + 26, N_LAYER + 25],
        [15, 16, 27, 26, N_LAYER + 15, N_LAYER + 16, N_LAYER + 27, N_LAYER + 26],
        [16, 17, 28, 27, N_LAYER + 16, N_LAYER + 17, N_LAYER + 28, N_LAYER + 27],
        [17, 18, 29, 28, N_LAYER + 17, N_LAYER + 18, N_LAYER + 29, N_LAYER + 28],
        [18, 19, 30, 29, N_LAYER + 18, N_LAYER + 19, N_LAYER + 30, N_LAYER + 29],
        [19, 20, 31, 30, N_LAYER + 19, N_LAYER + 20, N_LAYER + 31, N_LAYER + 30],
        [20, 21, 32, 31, N_LAYER + 20, N_LAYER + 21, N_LAYER + 32, N_LAYER + 31],
        [21, 10, 43, 32, N_LAYER + 21, N_LAYER + 10, N_LAYER + 43, N_LAYER + 32],
        [22, 23, 34, 33, N_LAYER + 22, N_LAYER + 23, N_LAYER + 34, N_LAYER + 33],
        [23, 24, 35, 34, N_LAYER + 23, N_LAYER + 24, N_LAYER + 35, N_LAYER + 34],
        [24, 25, 36, 35, N_LAYER + 24, N_LAYER + 25, N_LAYER + 36, N_LAYER + 35],
        [25, 26, 37, 36, N_LAYER + 25, N_LAYER + 26, N_LAYER + 37, N_LAYER + 36],
        [26, 27, 38, 37, N_LAYER + 26, N_LAYER + 27, N_LAYER + 38, N_LAYER + 37],
        [27, 28, 39, 38, N_LAYER + 27, N_LAYER + 28, N_LAYER + 39, N_LAYER + 38],
        [28, 29, 40, 39, N_LAYER + 28, N_LAYER + 29, N_LAYER + 40, N_LAYER + 39],
        [29, 30, 41, 40, N_LAYER + 29, N_LAYER + 30, N_LAYER + 41, N_LAYER + 40],
        [30, 31, 42, 41, N_LAYER + 30, N_LAYER + 31, N_LAYER + 42, N_LAYER + 41],
        [31, 32, 43, 42, N_LAYER + 31, N_LAYER + 32, N_LAYER + 43, N_LAYER + 42],
    ]),
]

_LAYER_DISPLACEMENT = np.array([
    # row 0
    [0, 0], [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0],
    [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0],
    [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0], [0, 0],
    # row 1
    [0, 0], [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0],
    [(L_SUPP - L_SUPP_2) / 2, T_1], [0, T_1], [-(L_SUPP - L_SUPP_2) / 2, T_1],
    [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0], [0, 0],
    # row 2
    [0, 0], [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0],
    [(L_SUPP - L_SUPP_2) / 2, -T_1], [0, -T_1],
    [-(L_SUPP - L_SUPP_2) / 2, -T_1],
    [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0], [0, 0],
    # row 3
    [0, 0], [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0],
    [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0],
    [(L_SUPP - L_SUPP_2) / 2, 0], [0, 0], [-(L_SUPP - L_SUPP_2) / 2, 0], [0, 0],
])

DISPLACEMENT = np.vstack([_LAYER_DISPLACEMENT, _LAYER_DISPLACEMENT])

MAX_E = (T_0 + T_1) / T_0 #(T_0 + T_1) / T_0

EXTRUSION_MULTIPLIER = np.array([[
    # row 0
    1., 1, 1, 1, MAX_E, MAX_E, MAX_E, 1, 1, 1, 1,
    # row 1
    1., 1, 1, 1, MAX_E, MAX_E, MAX_E, 1, 1, 1, 1,
    # row 2
    1., 1, 1, 1, MAX_E, MAX_E, MAX_E, 1, 1, 1, 1,
    # row 3
    1., 1, 1, 1, MAX_E, MAX_E, MAX_E, 1, 1, 1, 1,
]]).T
EXTRUSION_MULTIPLIER = np.vstack([EXTRUSION_MULTIPLIER, EXTRUSION_MULTIPLIER])

def get_filename(file_name):
    return os.path.sep.join([os.path.dirname(__file__), file_name])

def main(output_file=get_filename('beam.vtk')):
    mesh = meshio.Mesh(
        POINTS, CELLS,
        point_data={
            'displacement' : DISPLACEMENT,
            'extrusion' : EXTRUSION_MULTIPLIER,
        },
    )
    mesh.write(output_file)

def get_mesh_domain(mesh_file, offset=np.zeros(3)):
    mesh = sdf.Mesh.from_file(mesh_file, omit_facets=True)
    mesh.transform_coors(np.hstack([np.eye(3), offset.reshape((3, 1))]))
    domain = sdf.FEDomain('domain', mesh)

    return mesh, domain

def convert_to_stl(
        input_file=get_filename('beam.vtk'),
        output_file=get_filename('beam.stl'),
):
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(input_file)
    reader.Update()

    geom_filter = vtk.vtkGeometryFilter()
    geom_filter.SetInputConnection(reader.GetOutputPort())
    geom_filter.Update()

    polydata = geom_filter.GetOutput()

    writer = vtk.vtkSTLWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(output_file)

    writer.Write()

def process_line(
        line, last_coors, displacement_fun=None, extrusion_fun=None,
        comment_fun=None, max_extrusion_step=1e-2, zero_tol=1e-4
):
    if (displacement_fun is None) and (extrusion_fun is None):
        return line,

    if (line.upper().startswith('G0')
        or line.upper().startswith('G1')):
        move = gtrans.Move.from_str(line, last_coors)

        # displacement
        if displacement_fun is not None:
            # transformed_moves = move.transform_opt(
            #     displacement_fun, max_splits=5)
            transformed_moves = move.transform(displacement_fun)
        else:
            transformed_moves = move,

        # extrusion
        # if extrusion_fun is not None:
        #     for mv in transformed_moves:
        #         if mv.get_extrusion() > 0.:
        #             multiplier = extrusion_fun(mv.get_coors(.5))
        #             mv.arguments['E'] *= multiplier
        def subdivide_by_extrusion(move):
            if move.get_extrusion() < zero_tol:
                return move,

            e1 = extrusion_fun(move.get_coors(0))
            e2 = extrusion_fun(move.get_coors(1))

            if abs(e2 - e1) < zero_tol:
                return move,

            n_subdivisions = int(np.abs(e2 - e1) / max_extrusion_step) + 2
            cut_pts = np.linspace(0, 1, n_subdivisions + 1)
            cut_mid_pts = .5 * (cut_pts[:-1] + cut_pts[1:])
            return tuple(move.split(
                cut_pts[1:-1],
                np.array([
                    extrusion_fun(move.get_coors(mid_pt))
                    for mid_pt in cut_mid_pts
                ]),
            ))

        if extrusion_fun is not None:
            transformed_moves = sum([
                subdivide_by_extrusion(mv)
                for mv in transformed_moves], tuple())

        # add debug data into comment
        if comment_fun is not None:
            for mv in transformed_moves:
                mv.comment = comment_fun(mv)

        lines_out = (str(tm) for tm in transformed_moves)
        last_coors[:] = move.get_coors()
    else:
        lines_out = line,

    return lines_out

def convert_gcode(
        input_file, output_file=None, mesh_file=get_filename('beam.vtk'),
        mesh_offset=np.array((125., 105., 0)),
):
    if output_file is None:
        output_file = '.'.rsplit(input_file, maxsplit=1)[0] + '_out.gcode'

    mesh, domain = get_mesh_domain(mesh_file, mesh_offset)
    mesh2, domain2 = get_mesh_domain(mesh_file, mesh_offset)
    mesh2.cmesh.coors[:, :] += meshio.read(mesh_file).point_data['displacement']

    DISPLACEMENT = mesh2.cmesh.coors - mesh.cmesh.coors

    omega = domain.create_region('Omega', 'all')
    # perimeters = domain2.create_region(
    #     'perimeters', 'cell 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, '
    #     '25, 26, 27, 28, 29, 30, 31',
    # )

    vector_field = sd.Field.from_args(
        'fu', np.float64, 'vector', omega, approx_order=1)
    # scalar_field = sd.Field.from_args(
    #     'cu', np.float64, 'scalar', perimeters, approx_order=1)

    # plt.plot(mesh.coors[:, 0], mesh.coors[:, 1], '.-')
    # plt.plot(
    #     mesh.coors[:, 0] + DISPLACEMENT[:, 0],
    #     mesh.coors[:, 1] + DISPLACEMENT[:, 1],
    #     'x:'
    # )
    # plt.axis('equal')
    # plt.grid()
    # plt.show()
    # exit()

    def displacement_fun(coors):
        displacement = vector_field.evaluate_at(
            coors.reshape((-1, 3)),
            np.ascontiguousarray(DISPLACEMENT))[0, :, 0]
        displacement[np.isnan(displacement)] = 0.
        return np.array([
            coors.T[0] + displacement[0],
            coors.T[1] + displacement[1],
            coors.T[2],
        ])

    # def extrusion_fun(coors):
    #     extrusion = scalar_field.evaluate_at(
    #         coors.reshape((-1, 3)),
    #         np.ascontiguousarray(EXTRUSION_MULTIPLIER))[0, 0, 0]
    #     if np.isnan(extrusion):
    #         extrusion = 1.
    #     return extrusion

    # ys = np.linspace(100, 102, 4)
    # xs = np.linspace(105, 145, 101)
    # for yi in ys:
    #     coors = np.array([[xi, yi, .2] for xi in xs])
    #     es = np.array([extrusion_fun(coor) for coor in coors])
    #     plt.plot(xs, es, label=f'y={yi:.2g}')
    # plt.legend(loc='best')
    # plt.show()

    # exit()

    def comment_fun(move):
        extrusion_rate = move.get_extrusion() / move.get_length()
        out = f'{extrusion_rate}'
        if move.comment:
            out += f' ;{move.comment}'

        return out

    with open(input_file, 'r') as gf:
        lines = gf.readlines()

    # lines = lines[:690]

    # orig_moves = gtrans.get_moves(lines)
    # trans_moves = sum([
    #     list(move.transform(displacement_fun))
    #     for move in orig_moves], [])

    last_coors = np.zeros(3)
    output_gcode = (
        ''.join(process_line(
            line, last_coors,
            displacement_fun,
            # extrusion_fun,
            # comment_fun
        ))
        for line in lines[:600]
    )

    with open(output_file, 'w') as gf:
        gf.writelines(output_gcode)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-c', '--create', action='store_true',
        help='Create the files (vtk, stl) according to hard-coded parameters.'
        ' This is the default option.'
    )
    group.add_argument(
        '-g', '--convert-gcode', action='store_true',
        help='Convert the given gcode file according to hard-coded displacement'
        ' field.'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.convert_gcode:
        convert_gcode(
            get_filename('beam_uniform.gcode'),
            get_filename('beam_out_2.gcode'),
            # get_filename('extrusion_width_test_2.gcode'),
            # get_filename('extrusion_width_test_output.gcode'),
        )
    else:
        main()
        convert_to_stl()
