"""
Apply transformations to the deformed gcode files
"""
import gtrans

DEFAULT_PARS = [
    (f'examples/03_tension/T_{ii:02}_defo_0.2mm_PLA_MK3.gcode',
     f'examples/03_tension/T_{ii:02}.vtk',
     f'examples/03_tension/T_{ii:02}_out_0.2mm_PLA_MK3.gcode',
     (125., 105., 0.), 1200, True
    )
    for ii in (1, 2, 3, 4)[-1:]
]

# DEFAULT_PARS = [
#     ('examples/03_tension/T_04_defo_Hilbert_0.2mm_PLA_MK3.gcode',
#      'examples/03_tension/T_04.vtk',
#      'examples/03_tension/T_04_out_Hilbert_0.2mm_PLA_MK3.gcode',
#      (125., 105., 0.), 1200, True,
#      ),
# ]

if __name__ == '__main__':
    for pars in DEFAULT_PARS:
        gtrans.main(*pars)
