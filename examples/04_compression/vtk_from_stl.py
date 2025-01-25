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
        if value not in self[start:stop]:
            raise ValueError(f'{value} is not in list')

        for ii, list_val in enumerate(self[start:stop]):
            if np.linalg.norm(value - list_val) <= self.zero_tol:
                return ii

def plot_mesh(mesh, *args, ax=None, **kwargs):
    if ax is None:
        ax = plt

    for cell_block in mesh.cells:
        for elm_nodes in cell_block.data:
            node_ids = np.concatenate([elm_nodes, elm_nodes[:1]])
            node_coors = mesh.points[node_ids].T
            ax.plot(node_coors[0], node_coors[1], *args, **kwargs)

def plot_mesh_node_set(mesh, *args, ax=None, node_set=None, **kwargs):
    if ax is None:
        ax = plt

    if node_set is None:
        node_set = [nid for nid in range(mesh.points.size)]

    ax.plot(
        mesh.points[node_set, 0], mesh.points[node_set, 1],
        *args, **kwargs)

def plot_mesh_edge_set(mesh, edge_set, *args, ax=None, **kwargs):
    if ax is None:
        ax = plt

    x_vals, y_vals = [], []
    for n1, n2 in edge_set:
        x_vals += [None, mesh.points[n1, 0], mesh.points[n2, 0]]
        y_vals += [None, mesh.points[n1, 1], mesh.points[n2, 1]]

    ax.plot(x_vals, y_vals, *args, **kwargs)

def get_outline_points(file_name='ISO_dogbone_2.stl', zero_tol=1e-3):
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
    list of indices that traverse all edges.
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
    Convert a 2D vtk mesh to a 3D stl by extrusion in the Z-direction
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
