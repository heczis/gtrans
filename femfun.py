"""
Tools for working with FE functions, i.e. evaluating function values, reading
meshes and values etc.
"""
import enum

import meshio
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sopt

class Element:
    """
    Base class for different types of mesh elements
    """
    def __init__(self, node_ids, node_coors):
        self.node_ids = node_ids
        self.node_coors = node_coors

    def __call__(self, iso_coors, node_values):
        """
        Evaluate a quantitiy at `iso_coors`, based on `node_values`
        """
        out = np.zeros_like(node_values[0], dtype=np.float64)

        for node_val, base_fun in zip(node_values, self.base_funs):
            out += node_val * base_fun(iso_coors)

        return out

    def get_base_funs(self):
        raise NotImplementedError('This should be implemented in subclasses.')

    base_funs = property(get_base_funs)

    def get_edges(self):
        raise NotImplementedError('This should be implemented in subclasses.')

    edges = property(get_edges)

    def get_iso_coors(self, global_coors):
        raise NotImplementedError('This should be implemented in subclasses.')

    def is_inside(self, coors):
        raise NotImplementedError('This should be implemented in subclasses.')

    def get_intersections_with_edges(self, line):
        """
        Return parametric coordinates (on the line) of intersections with
        the edges of `self`.
        """
        intersections = []
        for edge in self.edges:
            rhs = (edge[1] - line[1])[:2]
            mat = np.vstack([line[0] - line[1], edge[1] - edge[0]]).T[:2]
            try:
                lbds = np.linalg.solve(mat, rhs)
                if (
                        (lbds[0] >= 0) and (lbds[0] <= 1) and
                        (lbds[1] >= 0) and (lbds[1] <= 1)
                ):
                    intersections.append(lbds[0])
            except:
                pass

        return intersections

    def plot(self, *args, ax=None, displacement=None, **kwargs):
        if ax is None:
            ax = plt

        inds = [ii for ii in range(self.node_coors.shape[0])] + [0]
        coors = self.node_coors

        if displacement is not None:
            coors = coors + displacement

        ax.plot(coors[inds, 0], coors[inds, 1], *args, **kwargs)

    def __str__(self):
        return str(self.node_ids)

class Tria(Element):
    """
    Three-node linear element
    """
    def get_base_funs(self):
        def _get_bf(ii):
            def _fun(iso_coors):
                return iso_coors[ii]
            return _fun

        return [_get_bf(ii) for ii in range(3)]

    base_funs = property(get_base_funs)

    def get_edges(self):
        return list(zip(self.node_coors,
                        np.vstack([self.node_coors[1:], self.node_coors[:1]])))

    edges = property(get_edges)

    def get_iso_coors(self, global_coors):
        rhs = np.array([global_coors[0], global_coors[1], 1.])
        mat = np.vstack([self.node_coors.T[:2], np.ones(3)])
        out = np.linalg.solve(mat, rhs)
        return out

    def is_inside(self, coors, tol=1e-4):
        iso_coors = self.get_iso_coors(coors)

        return all((iso_coors >= -tol) * (iso_coors <= 1 + tol))

class Tetra(Element):
    """
    Four-node linear element (tetrahedron)
    """
    pass

class Quad(Element):
    """
    Four-node bilinear element
    """
    def get_base_funs(self):
        return [
            lambda ics: (1. - ics[0]) * (1 - ics[1]),
            lambda ics: ics[0] * (1. - ics[1]),
            lambda ics: ics[0] * ics[1],
            lambda ics: (1. - ics[0]) * ics[1],
        ]

    base_funs = property(get_base_funs)

    def get_edges(self):
        return list(zip(self.node_coors,
                        np.vstack([self.node_coors[1:], self.node_coors[:1]])))

    edges = property(get_edges)

    def get_iso_coors(self, global_coors):
        def obj_fun(iso_coors):
            return np.linalg.norm(
                global_coors[:2] - self(iso_coors, self.node_coors)[:2])

        result = sopt.minimize(
            obj_fun,
            x0=.5 * np.ones(2),
        )
        return result.x

    def is_inside(self, coors, tol=1e-4):
        iso_coors = self.get_iso_coors(coors)

        return all((iso_coors >= -tol) * (iso_coors <= 1 + tol))

class Hexa(Element):
    """
    Eight-node trilinear element
    """
    pass

CELL_TYPE_DICT = {
    'triangle' : Tria, 'quad' : Quad, 'tetra' : Tetra, 'hexahedron' : Hexa,
}

class ToleranceList(list):
    def __init__(self, *args, zero_tol=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.zero_tol = zero_tol

    def __contains__(self, other_item):
        for item in self:
            if np.linalg.norm(item - other_item) <= self.zero_tol:
                return True

        return False

class Mesh:
    def __init__(self, meshdata, offset=None):
        self.evaluate = None

        if offset is not None:
            meshdata.points = meshdata.points + np.array(offset).reshape(
                (-1, meshdata.points.shape[1]))

        self.meshdata = meshdata

        self.elements = sum([
            [CELL_TYPE_DICT[cd[0]](cd_ii, self.meshdata.points[cd_ii])
             for cd_ii in cd[1]]
            for cd in self.meshdata.cells_dict.items()], [])

    def __call__(self, coors, nodal_values):
        """
        Evaluate a function (represented by the mesh) at `coors`.
        """
        elm = self.get_elm_by_coors(coors)

        if elm is None:
            return None

        out = elm(
            elm.get_iso_coors(coors),
            nodal_values[elm.node_ids],
        )
        return out

    def get_elm_by_coors(self, coors):
        out = None
        for elm in self.elements:
            if elm.is_inside(coors):
                out = elm
                break

        return out

    @classmethod
    def from_file(cls, file_name, *args, **kwargs):
        mesh_data = meshio.read(file_name)
        return cls(mesh_data, *args, **kwargs)

    def get_intersections_with_edges(self, line, zero_tol=1e-4):
        all_lbds = sum([
            elm.get_intersections_with_edges(line)
            for elm in self.elements], [])
        out = ToleranceList([], zero_tol=zero_tol)
        for lbd in all_lbds:
            if lbd not in out:
                out.append(lbd)

        return out

    def plot(self, *args, ax=None, displacement=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        for elm in self.elements:
            if displacement is None:
                elm.plot(*args, ax=ax, **kwargs)
            else:
                elm.plot(
                    *args, ax=ax, displacement=displacement[elm.node_ids],
                    **kwargs)

def create_testing_vtk_2d(file_name='examples/test2d.vtk'):
    points = np.array([
        [-15, -15], [0, -15], [15, -15],
        [-15, 0], [0, 0], [15, 0],
        [-15, 15], [0, 15], [15, 15],
    ])
    cells = (
        ('quad', [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6]]),
        ('triangle', [[4, 5, 7], [5, 8, 7]]),
    )
    point_data = {
        'displacement' : [
            [2, 5], [0, 0], [5, 0],
            [0, 1], [0, 0], [2, -1],
            [0, 3], [0, 3], [-2, 7],
        ],
    }
    mesh = meshio.Mesh(points, cells, point_data)
    mesh.write(file_name, binary=False)

if __name__ == '__main__':
    # create_testing_vtk_2d()
    # exit()

    mesh = Mesh.from_file('examples/test2d.vtk')
    line = np.array([[-10, 5, 0], [20, 20, 0]])
    # line = np.array([[-5, 20, 0], [20, -5, 0]])
    # line = np.array([[15, 20, 0], [15, -20, 0]])
    lbds = mesh.get_intersections_with_edges(line)
    print('lbd : coors')
    for lbd in lbds:
        print(f'{lbd} : {lbd * line[0] + (1 - lbd) * line[1]}')

    intersection_coors = np.array([
        lbd * line[0] + (1 - lbd) * line[1] for lbd in lbds])

    eval_coors = np.vstack([
        line[:1],
        .5 * (intersection_coors[:-1] + intersection_coors[1:]),
        line[-1:],
    ])
    print('is in elm:')
    for coor in eval_coors:
        print(f'{coor}: {mesh.get_elm_by_coors(coor)}')

    mesh.plot('k', linewidth=.5)
    plt.plot(line[:2, 0], line[:2, 1])
    plt.plot(intersection_coors[:, 0], intersection_coors[:, 1], 'o')
    plt.plot(eval_coors[:, 0], eval_coors[:, 1], 's')
    plt.gca().axis('equal')
    plt.show()
