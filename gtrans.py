"""
Process existing gcode according to data in a mesh.
"""
import argparse
import ipdb

import matplotlib.pyplot as plt
import numpy as np

import gcode
import femfun

def transform_gcode(
        gcode_in, mesh_in,
        output_count=100,
        n_lines=None,
        do_plots=False,
):
    gcode_out = gcode.GCode([])

    def displacement_fun(coors):
        displacement = mesh_in(
            coors, mesh_in.meshdata.point_data['displacement']
        )
        if displacement is None:
            return coors
        else:
            return coors + displacement

    if n_lines is None:
        n_lines = len(gcode_in.instructions)

    for ii, instruction in enumerate(gcode_in.instructions[:n_lines]):
        if output_count:
            if (ii % output_count) == 0:
                print(f'[{ii}/{n_lines}]')
        if not instruction.move:
            gcode_out.instructions.append(instruction)
            continue

        move = instruction.move
        extrusion, length = move.get_extrusion(), move.get_length()

        if do_plots:
            _, ax = plt.subplots()
            ax.axis('equal')
            mesh_in.plot('k', ax=ax, linewidth=.5)
            instruction.plot('.-', ax=ax)

        if (extrusion <= 0) or (length <= 0):
            new_moves = move.transform(displacement_fun)
            for new_move in new_moves:
                gcode_out.instructions.append(
                    gcode.Instruction.from_str(
                        str(new_move),
                        new_move.get_coors(0))
                )
            if do_plots:
                mesh_in.plot(
                    ':', linewidth=.5, ax=ax,
                    displacement=mesh_in.meshdata.point_data['displacement'])
                gcode_out.instructions[-1].plot('x--', ax=ax)
                plt.show()
            continue

        lbds = mesh_in.get_intersections_with_edges(
            np.array([move.get_coors(0), move.get_coors(1)])
        )
        lbds.sort(reverse=True)
        new_moves = sum([
            mv.transform(displacement_fun)
            for mv in move.split([1 - lbd for lbd in lbds])
        ], [])

        for mv in new_moves:
            start_coors = gcode_out.get_last_coors()
            gcode_out.instructions.append(
                gcode.Instruction.from_str(str(mv), start_coors))

        if do_plots:
            do_stop = False
            mesh_in.plot(
                ':', linewidth=.5, ax=ax,
                displacement=mesh_in.meshdata.point_data['displacement'])
            for inst in gcode_out.instructions[-len(new_moves):]:
                inst.plot('x--', ax=ax)

            for lbd in lbds:
                _coor = lbd * instruction.move.get_coors(0) \
                    +(1 - lbd) * instruction.move.get_coors(1)
                ax.annotate(
                    str(mesh_in.get_elm_id(mesh_in.get_elm_by_coors(_coor))),
                    xy=_coor[:2],
                )
                if mesh_in.get_elm_id(mesh_in.get_elm_by_coors(_coor)) == 0:
                    do_stop = True
            
            if do_stop:
                plt.show()
                ipdb.set_trace()
            else:
                plt.close()

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

    parser.add_argument(
        '--n-lines', metavar='N_LINES', dest='n_lines', default=None,
        type=int, nargs='?',
        help='Number of gcode lines to process (might be useful for debugging).'
        ' If not given, all gcode is processed.',
    )

    return parser.parse_args()

def main(
        gcode_in_filename, mesh_in_filename, gcode_out_filename,
        mesh_offset, n_lines=None, do_plot=False,
):
    gcode_in = gcode.GCode.from_file(gcode_in_filename)
    mesh_in = femfun.Mesh.from_file(
        mesh_in_filename, offset=mesh_offset,
    )

    if do_plot:
        _, (ax_orig, ax_after) = plt.subplots(ncols=2, sharex=True)
        ax_orig.axis('equal')
        mesh_in.plot('k', ax=ax_orig, linewidth=.5)
        gcode_in.plot('-', color='#1f77b4', ax=ax_orig)

    gcode_out = transform_gcode(gcode_in, mesh_in, n_lines=n_lines)

    gcode_out.save_as(gcode_out_filename)

    if do_plot:
        ax_after.axis('equal')
        mesh_in.plot(
            'k', ax=ax_after,
            displacement=mesh_in.meshdata.point_data['displacement'],
            linewidth=.5)
        gcode_out.plot('-', color='#1f77b4', ax=ax_after)
        plt.show()

if __name__ == '__main__':
    cli_args = parse_args()

    main(
        cli_args.gcode_in, cli_args.mesh_in, cli_args.gcode_out,
        cli_args.offset, n_lines=cli_args.n_lines, do_plot=True
    )
