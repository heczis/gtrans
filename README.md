# gtrans

A python library and some example scripts (to be gradually added) for manipulating 3D printer gcode files using geometrical transformations.
The main goal was to enable graded infill structures using any slicing software.
The principle is based on the work of [Wüthrich et al.](https://dx.doi.org/10.1007/978-3-030-54334-1_10) who used gcode transformations for support-less printing.

`gtrans` uses the transformations to create infill structures with non-uniform density and anisotropy.
The definitions of the geometrical transformations may be either analytical or based on a finite-element mesh with prescribed displacements at nodes.
