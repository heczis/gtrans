"""
Plot the tensile specimens with different infill patterns
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import gcode

DEFAULT_PARS = (
    ('examples/03_tension/T_00_out_0.2mm_PLA_MK3.gcode',
     'examples/03_tension/T_00.pdf'),
    ('examples/03_tension/T_01_out_0.2mm_PLA_MK3.gcode',
     'examples/03_tension/T_01.pdf'),
    ('examples/03_tension/T_02_out_0.2mm_PLA_MK3.gcode',
     'examples/03_tension/T_02.pdf'),
    ('examples/03_tension/T_03_out_0.2mm_PLA_MK3.gcode',
     'examples/03_tension/T_03.pdf'),
    ('examples/03_tension/T_04_out_0.2mm_PLA_MK3.gcode',
     'examples/03_tension/T_04.pdf'),
    ('examples/03_tension/T_04_out_Hilbert_0.2mm_PLA_MK3.gcode',
     'examples/03_tension/T_04_Hilbert.pdf'),
)

def plot_file(gcode_file, output_file=None, **kwargs):
    fig = plt.figure(
        figsize=(7, 1.5),
        layout='constrained',
    )
    gs = GridSpec(1, 6, figure=fig)
    ax = fig.add_subplot(gs[0, 0:5])
    ax_zoom = fig.add_subplot(gs[0, 5])

    ax.axis('equal')
    ax_zoom.axis('equal')

    gc = gcode.GCode.from_file(gcode_file)
    end_layer = 2
    actual_layer = 0
    end_ind = -1
    for ind, ii in enumerate(gc.instructions):
        if 'AFTER_LAYER_CHANGE' in ii.comment:
            actual_layer += 1
            if actual_layer > end_layer:
                end_ind = ind
                break

    gc.instructions = gc.instructions[:end_ind]
    gc.plot(ax=ax, **kwargs)

    ax.set(xlim=(85, 165), ylim=(95, 115))
    ax.axis('off')

    gc.plot(ax=ax_zoom, **kwargs)
    ax_zoom.set(xlim=(95, 100), ylim=(102, 107))
    ax_zoom.axis('off')

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()

if __name__ == '__main__':
    pcyc = plt.rcParams['axes.prop_cycle']
    for pars, style_kwargs in zip(DEFAULT_PARS, pcyc):
        plot_file(*pars, **style_kwargs)
