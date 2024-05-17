"""
Plot the tensile specimens with different infill patterns
"""
import matplotlib.pyplot as plt

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
)[4:]

def plot_file(gcode_file, output_file=None):
    fig, ax = plt.subplots(
        figsize=(5, 1.5),
        layout='constrained'
    )
    ax.axis('equal')

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
    gc.plot(color='#1f77b4', ax=ax)

    ax.set(xlim=(85, 165), ylim=(95, 115))
    ax.axis('off')

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()

if __name__ == '__main__':
    for pars in DEFAULT_PARS:
        plot_file(*pars)
