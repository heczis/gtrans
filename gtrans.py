"""
Process existing gcode according to data in a mesh.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

import gcode
import femfun

def transform_gcode(gcode_in, mesh_in):
    gcode_out = gcode.GCode([])

    def displacement_fun(coors):
        displacement = mesh_in(
            coors, mesh_in.meshdata.point_data['displacement']
        )
        if displacement is None:
            return coors
        else:
            return coors + displacement

    for instruction in gcode_in.instructions:
        if not instruction.move:
            gcode_out.instructions.append(instruction)
            continue

        move = instruction.move
        extrusion, length = move.get_extrusion(), move.get_length()

        if (extrusion <= 0) or (length <= 0):
            new_move = move.transform(displacement_fun)
            gcode_out.instructions.append(
                gcode.Instruction.from_str(
                    str(new_move),
                    new_move.get_coors(0))
            )
            continue

        lbds = mesh_in.get_intersections_with_edges(
            np.array([move.get_coors(0), move.get_coors(1)])
        )
        lbds.sort(reverse=True)
        new_moves = [
            mv.transform(displacement_fun)
            for mv in move.split([1 - lbd for lbd in lbds])
        ]

        for mv in new_moves:
            start_coors = gcode_out.get_last_coors()
            gcode_out.instructions.append(
                gcode.Instruction.from_str(str(mv), start_coors))

    return gcode_out

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        metavar='GCODE_IN', dest='gcode_in', help='Input gcode')
    parser.add_argument(
        metavar='MESH_IN', dest='mesh_in', help='Input mesh with data')
    parser.add_argument(
        metavar='GCODE_OUT', dest='gcode_out', help='Output gcode')

    parser.add_argument(
        '--offset', metavar='OFFSET', dest='offset', default=[125, 105, 0],
        type=float, nargs=3,
        help='Translation vector of mesh in order to overlap with gcode. '
        '[default: %(default)s]',
    )

    return parser.parse_args()

def main(
        gcode_in_filename, mesh_in_filename, gcode_out_filename,
        mesh_offset,
):
    gcode_in = gcode.GCode.from_file(gcode_in_filename)
    mesh_in = femfun.Mesh.from_file(
        mesh_in_filename, offset=mesh_offset,
    )

    _, (ax_orig, ax_after) = plt.subplots(ncols=2, sharey=True)
    ax_orig.axis('equal')
    mesh_in.plot('k', ax=ax_orig, linewidth=.5)
    gcode_in.plot('.-', ax=ax_orig)

    gcode_out = transform_gcode(gcode_in, mesh_in)

    gcode_out.save_as(gcode_out_filename)

    ax_after.axis('equal')
    mesh_in.plot(
        'k', ax=ax_after,
        displacement=mesh_in.meshdata.point_data['displacement'],
        linewidth=.5)
    gcode_out.plot('.-', ax=ax_after)
    plt.show()

if __name__ == '__main__':
    cli_args = parse_args()
    main(
        cli_args.gcode_in, cli_args.mesh_in, cli_args.gcode_out,
        cli_args.offset,
    )
