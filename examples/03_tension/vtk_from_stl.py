"""
Tools for creating deformed .stl files and the corresponding .vtk meshes, both
to be used for slicing and post-processing the resulting gcode.
"""
import argparse

import matplotlib.pyplot as plt
import meshio
import numpy as np
import vtk

N_PERIMETERS = 1
EXTRUSION_WIDTH = 0.45

class ToleranceList(list):
    def __init__(self, *args, zero_tol=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_tol = zero_tol

    def __contains__(self, other_item):
        for item in self:
            if np.linalg.norm(item - other_item) <= self.zero_tol:
                return True

        return False

    def index(self, value, start=0, stop=9223372036854775807):
        if value not in self:
            raise ValueError(f'{value} is not in list')

        for ii, list_val in enumerate(self):
            if np.linalg.norm(value - list_val) <= self.zero_tol:
                return ii

def plot_mesh(mesh, *args, ax=None, **kwargs):
    """
    Plot given mesh

    Parameters:
    -----------

    mesh : meshio.Mesh

    ax : pyplot.Axes or None

      If `None`, replaced by matplotlib.pyplot

    args, kwargs : passed to `ax.plot` calls
    """
    if ax is None:
        ax = plt

    for cell_block in mesh.cells:
        for elm_nodes in cell_block.data:
            node_ids = np.concatenate([elm_nodes, elm_nodes[:1]])
            node_coors = mesh.points[node_ids].T
            ax.plot(node_coors[0], node_coors[1], *args, **kwargs)

def plot_mesh_node_set(mesh, *args, ax=None, node_set=None, **kwargs):
    """
    Plot given node set

    Parameters:
    -----------

    mesh : meshio.Mesh

    ax : pyplot.Axes or None

      If `None`, replaced by matplotlib.pyplot

    node_set : list of ints

      List of node IDs to plot. All nodes are plotted if `None`.

    args, kwargs : passed to `ax.plot` calls
    """
    if ax is None:
        ax = plt

    if node_set is None:
        node_set = [nid for nid in range(mesh.points.size)]

    ax.plot(
        mesh.points[node_set, 0], mesh.points[node_set, 1],
        *args, **kwargs)

def plot_mesh_edge_set(mesh, edge_set, *args, ax=None, **kwargs):
    """
    Plot given set of edges

    Parameters:
    -----------

    mesh : meshio.Mesh

    ax : pyplot.Axes or None

      If `None`, replaced by matplotlib.pyplot

    edge_set : list of tuples of ints

      List of edges to plot. Each edge is defined by two node IDs that it
      connects.

    args, kwargs : passed to `ax.plot` calls
    """
    if ax is None:
        ax = plt

    x_vals, y_vals = [], []
    for n1, n2 in edge_set:
        x_vals += [None, mesh.points[n1, 0], mesh.points[n2, 0]]
        y_vals += [None, mesh.points[n1, 1], mesh.points[n2, 1]]

    ax.plot(x_vals, y_vals, *args, **kwargs)

def get_outline_points(file_name='ISO_dogbone_2.stl', zero_tol=1e-3):
    """
    Return coordinates of points that lie on the contour of a tensile specimen.
    It is assumed that the shape is planar in the XY-plane and symmetric about
    the X and Y axis.
    """
    mesh = meshio.read(file_name)

    qpts = mesh.points[
        (mesh.points[:, 2] < zero_tol) * (mesh.points[:, 1] > zero_tol)
        * (mesh.points[:, 0] > zero_tol)]

    ordering = qpts[:, 0].argsort()
    print(qpts[ordering])
    return np.vstack([
        qpts[ordering].dot(np.diag([-1, 1, 1]))[::-1],
        qpts[ordering],
    ])

class EdgeList(list):
    """
    List of edges (i.e. tuples of node IDs).
    Redefines the `__contains__` method and supports search by node ID.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __contains__(self, other_edge):
        on1, on2 = other_edge
        for n1, n2 in self:
            if (n1 == on1 and n2 == on2) or (n1 == on2 and n2 == on1):
                return True

        return False

    def get_edge_with_node(self, node_id):
        for ind, edge in enumerate(self):
            if node_id in edge:
                return ind

def get_outline_edges(mesh):
    """
    Finds edges that lie on the contour of a given 2D mesh.
    
    Parameters:
    -----------

    mesh : meshio.Mesh

      2D is assumed!
    """
    elm_edges = []
    for cell_block in mesh.cells:
        for elm_nodes in cell_block.data:
            node_ids = np.concatenate([elm_nodes, elm_nodes[:1]])
            elm_edges.append(EdgeList(
                [[n1, n2] for n1, n2 in zip(node_ids[:-1], node_ids[1:])]
            ))

    out_edges = []
    for elm_id, edges in enumerate(elm_edges):
        # print(f'ELM {elm_id}')
        for edge in edges:
            for other_eid, other_edges in enumerate(elm_edges):
                if elm_id == other_eid:
                    continue

                if edge in other_edges:
                    # print(f' Edge {edge} found among {other_edges}.')
                    break
            else:
                # print(f' Edge {edge} not found among other elements, adding.')
                out_edges.append(edge)

    return out_edges

def is_clockwise(polygon):
    """
    Tests for planar polygon orientation. Used in determining the orientation
    of outward normals and fixing element definition (necessary for finding the
    iso-coordinates in quad elements).
    
    Parameters:
    -----------
    
    polygon : array, type (n_points + 1, n_dim)

      Last node must be repeated!

    Returns:
    --------

    float; if positive, the polygon is oriented counter-clockwise, if negative,
    it is oriented clockwise, if zero, it cannot be determined (area is zero).
    """
    out = 0.
    for (x1, y1), (x2, y2) in zip(polygon[:-1], polygon[1:]):
        out += (x2 - x1) * (y2 + y1)

    return out

def get_outer_normals(polygon):
    """
    Parameters:
    -----------
    
    polygon : array, type (n_points + 1, n_dim)

      Last node must be repeated!

    Returns:
    --------

    list of otward normals to each edge
    """
    n_dim = polygon.shape[1]

    if n_dim == 2:
        rot_mat = np.array([[0, 1],[-1, 0]])
    elif n_dim == 3:
        rot_mat = np.array([[0, 1, 0],[-1, 0, 0], [0, 0, 1]])
    else:
        raise NotImplementedError(
            f'Only 2- and 3-dimensional shapes are implemented ({n_dim} given)')

    normals = []
    coef = -np.sign(is_clockwise(polygon))
    for n1, n2 in zip(polygon[:-1], polygon[1:]):
        normals.append(coef * rot_mat.dot(n2 - n1))

    return normals / np.array([[np.linalg.norm(ni)] for ni in normals])

def create_mesh_from_outline_points(outline_points):
    """
    Parameters:
    -----------

    outline_points : list of coordinates

      Points in the XY-plane, all should lie on a single side from the X-axis,
      since the other half of the mesh is created by mirroring around X.
    """
    points, cells = ToleranceList([]), []
    for p1, p2 in zip(outline_points[:-1], outline_points[1:]):
        p3 = p2.dot(np.diag([1, -1, 1]))
        p4 = p1.dot(np.diag([1, -1, 1]))

        elm_points = (p1, p2, p3, p4)

        for pt in elm_points:
            if pt not in points:
                points.append(pt)

        pt_inds = [points.index(pt) for pt in elm_points]
        cells.append(pt_inds)

    return meshio.Mesh(
        np.array(points),
        (('quad', cells),),
    )

def get_path(edges):
    """
    From a list of edges - list of pairs of node indices - construct a single
    ordered list of indices that traverse all edges.
    """
    if not isinstance(edges, EdgeList):
        edges = EdgeList(edges)

    out = [edges[0][0], edges[0][1]]
    edges.pop(0)
    for _ in range(len(edges)):
        last_node_id = out[-1]
        next_ind = edges.get_edge_with_node(last_node_id)
        next_edge = edges.pop(next_ind)
        if last_node_id == next_edge[0]:
            out.append(next_edge[1])
        else:
            out.append(next_edge[0])
    return out

def get_single_offset_displacement(n1, n2, zero_tol=1e-4):
    """
    Get the displacement of a node based on outward normals of neighboring line
    segments.
    """
    if np.linalg.norm(n2 - n1) < zero_tol:
        return n1

    mat = np.array([n1, n2])
    rhs = np.array([n1.dot(n1), n2.dot(n2)])

    return np.linalg.solve(mat, rhs)

def offset_outline(outline_path, offset=-1.35):
    """
    Parameters:
    -----------

    outline_path : ndarray, type (n_points + 1, 2)

      Coordinates of outline vertices

    offset : float

      Negative = inwards, positive = outwards
    """
    normals = get_outer_normals(outline_path)
    displacements = np.zeros_like(outline_path)

    for ii in range(len(outline_path)):
        if ii in (0, len(outline_path) - 1):
            n1, n2 = normals[0], normals[-1]
        else:
            n1, n2 = normals[ii], normals[ii - 1]

        displacements[ii] = offset * get_single_offset_displacement(n1, n2)

    return displacements

def convert_to_stl(
        input_file='beam.vtk', output_file='beam.stl',
        thickness=1.,
):
    """
    Convert a 2D .vtk to .stl, extrude by given thickness.
    """
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(input_file)
    reader.Update()

    geom_filter = vtk.vtkGeometryFilter()
    geom_filter.SetInputConnection(reader.GetOutputPort())
    geom_filter.Update()

    polydata = geom_filter.GetOutput()

    extrusion_filter = vtk.vtkLinearExtrusionFilter()
    extrusion_filter.SetInputData(polydata)
    extrusion_filter.Update()

    polydata = extrusion_filter.GetOutput()

    writer = vtk.vtkSTLWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(output_file)

    writer.Write()

def uniform_defo_fun(coors, def_grad=None):
    if def_grad is None:
        def_grad = np.eye(len(coors))
    return def_grad.dot(coors)

def main(
        input_stl_name='ISO_dogbone_2.stl',
        output_vtk_name='ISO_dogbone_2_defo.vtk',
        output_stl_name='ISO_dogbone_2_defo.stl',
        deformation_gradient=None,
        do_plots=False,
):
    pts = get_outline_points(input_stl_name)

    mesh = create_mesh_from_outline_points(pts)
    outline_edges = get_outline_edges(mesh)

    outline_path = get_path(outline_edges)

    if do_plots:
        plot_mesh(mesh, 'k', linewidth=.5)
        plot_mesh_edge_set(mesh, outline_edges, 'x--')

    path_points = mesh.points[outline_path, :2]
    inner_points = path_points + offset_outline(
        path_points, offset=-N_PERIMETERS * EXTRUSION_WIDTH)

    orig_points = np.vstack([path_points[:-1], inner_points[:-1]])

    defo_fun = lambda coors: uniform_defo_fun(coors, deformation_gradient)
    inner_points_deformed = np.array([defo_fun(pt) for pt in inner_points])
    outer_points_deformed = inner_points_deformed + offset_outline(
        inner_points_deformed, offset=N_PERIMETERS * EXTRUSION_WIDTH)

    n_outline = len(inner_points_deformed) - 1

    out_points = np.vstack([
        outer_points_deformed[:-1], inner_points_deformed[:-1],
    ])

    out_quads = [
        [ii, ii + 1, n_outline + ii + 1, n_outline + ii]
        for ii in range(n_outline - 1)]
    out_quads += [
        [n_outline - 1, 0, n_outline, 2 * n_outline - 1]]
    out_quads += [
        [n_outline + ii, n_outline + ii + 1,
         2 * n_outline - ii - 2, 2 * n_outline - ii - 1]
        for ii in range(n_outline // 2 - 1)
    ]
    out_quads = [elm[::-1] for elm in out_quads]

    displacement = orig_points - out_points

    mesh_out = meshio.Mesh(
        out_points, (('quad', out_quads),),
        point_data={
            'displacement' : displacement,
            'extrusion' : np.ones_like(out_points),
        },
    )
    mesh_out.write(output_vtk_name, binary=False)

    convert_to_stl(output_vtk_name, output_stl_name)

    # plt.plot(out_points.T[0], out_points.T[1], ':')
    if do_plots:
        plot_mesh(mesh_out, '--')

        plt.axis('equal')
        # plt.plot(pts.T[0], pts.T[1], '.-')
        plt.grid()
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--input-stl',
        metavar='INPUT_STL', dest='input_stl', help='Name of the original STL. '
        'Only its outline at Z=0 is used. [default: %(default)s]',
        default='ISO_dogbone_2.stl',
    )
    parser.add_argument(
        metavar='OUTPUT_STL', dest='output_stl', help='Name of the output STL. '
        'This is the deformed one to be sliced and transformed back.',
    )
    parser.add_argument(
        metavar='OUTPUT_VTK', dest='output_vtk', help='Name of the output VTK. '
        'This contains the finite element mesh with the deformation and '
        'extrusion fields.',
    )
    parser.add_argument(
        '--deformation-gradient', '-d', default=[1., 0., 0., 1.],
        metavar='Fij', type=float, nargs=4, help='Components of the deformation'
        ' gradient to be used to deform the shape. 2D is assumed with the '
        'ordering [F_11, F_12, F_21, F_22]. [default: %(default)s]',
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    def_grad = np.array(args.deformation_gradient).reshape((2, 2))
    main(
        args.input_stl, args.output_vtk, args.output_stl,
        deformation_gradient=def_grad,
    )
