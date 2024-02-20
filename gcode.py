import matplotlib.pyplot as plt
import numpy as np

MOVE_INSTRUCTIONS = 'G0', 'G1'

def parse_arg(arg_str):
    """
    Parse a single argument of a gcode instruction.
    """
    arg_name = arg_str.strip()[0].upper()
    arg_value = float(arg_str.strip()[1:])
    return arg_name, arg_value

class Move:
    """
    Represents a move/extrude gcode instruction (G0 or G1).
    Provides parsing and operation on its arguments.

    Parameters:
    -----------

    name : str
    arguments : dict, e.g. {'X' : 0.0, 'Y' : 10.5, 'E' : 12.0}
    """
    def __init__(self, name, arguments, start_coors=np.zeros(3), comment=None):
        self.name = name
        self.arguments = arguments
        self.start_coors = start_coors
        self.comment = comment

    def __str__(self):
        out = ' '.join(
            [self.name]
            + [f'{arg}{val:.6f}' for arg, val in self.arguments.items()]
        )
        if self.comment:
            out += f' ;{self.comment}'

        out += '\n'
        return out

    @staticmethod
    def from_str(move_str, start_coors=np.zeros(3)):
        """
        Parse a string [read from a file] and return a `Move` instance.
        """
        clean_str = move_str.strip().upper()

        comment_split = clean_str.split(';', maxsplit=1)
        if len(comment_split) > 1:
            clean_str, comment = comment_split
        else:
            clean_str, comment = comment_split[0], None

        if not(clean_str.startswith('G0')
               or clean_str.startswith('G1')):
            return None

        args = clean_str.split()

        name = args[0]
        pars = dict((parse_arg(arg) for arg in args[1:]))
        return Move(name, pars, np.array(start_coors), comment)

    def get_extrusion(self):
        out = 0.
        if 'E' in self.arguments:
            out = self.arguments['E']

        return out

    def get_coors(self, par_val=1.):
        end_coors = np.array([
            self.arguments['X'] if 'X' in self.arguments else self.start_coors[0],
            self.arguments['Y'] if 'Y' in self.arguments else self.start_coors[1],
            self.arguments['Z'] if 'Z' in self.arguments else self.start_coors[2],
        ])
        return par_val * end_coors + (1 - par_val) * self.start_coors

    def set_end_coors(self, new_coors):
        if ('X' in self.arguments) or (new_coors[0] != self.start_coors[0]):
            self.arguments['X'] = new_coors[0]
        if ('Y' in self.arguments) or (new_coors[1] != self.start_coors[1]):
            self.arguments['Y'] = new_coors[1]
        if ('Z' in self.arguments) or (new_coors[2] != self.start_coors[2]):
            self.arguments['Z'] = new_coors[2]

    def get_length(self):
        return np.linalg.norm(self.get_coors(1) - self.get_coors(0))

    def transform(self, displacement_fun):
        """
        Return a Move instance transformed by `displacement_fun`.

        Parameters:
        -----------

        displacement_fun : function, accepts numpy vector of length 3, returns
                           the same.
        split_threshold : float, split the resulting move into two if the
                          distance of a transformed point from a corresponding
                          point on the transformed move is larger. Applied
                          recursively!
                          No splitting is performed if split_threshold is zero.
        split_par : float, between 0 and 1, relative location of the point,
                    at which the splitting condition is tested
        """
        start_coors = self.start_coors
        end_coors = self.get_coors()

        start_coors_trasformed = displacement_fun(start_coors)
        end_coors_transformed = displacement_fun(end_coors)

        new_args = {}

        for ind, coor_name in enumerate(['X', 'Y', 'Z']):
            if (coor_name in self.arguments
                or start_coors_trasformed[ind] != end_coors_transformed[ind]):
                new_args.update({coor_name : end_coors_transformed[ind]})

        if 'E' in self.arguments:
            len_orig = np.linalg.norm(end_coors - start_coors)
            len_trans = np.linalg.norm(
                end_coors_transformed - start_coors_trasformed)

            new_extrusion_value = self.arguments['E']
            if len_orig > 0:
                new_extrusion_value *= len_trans / len_orig

            new_args.update({'E' : new_extrusion_value})

        for arg_name, val in self.arguments.items():
            if arg_name not in new_args:
                new_args.update({arg_name : val})

        out = Move(self.name, new_args, start_coors=start_coors_trasformed)
        return out

    def transform_extrusion(
            self, extrusion_fun, split_threshold=1+1e-2, split_par=.5,
    ):
        """
        Return a list of Move instances with extrusion scaled by multipliers
        according to `extrusion_fun`.

        Parameters:
        -----------

        extrusion_fun : function, accepts numpy vector of length 3
                        (coordinates), returns float (extrusion multiplier)
        split_threshold : float, split the resulting move into two if the
                          extrusion multipliers on its two segments differ by a
                          factor greater than `split_threshold`. Applied
                          recursively!
        split_par : float, between 0 and 1, relative location of the splitting
                    point, extrusion multipliers are evaluated at the centers
                    of each segment
        """
        if 'E' not in self.arguments:
            return self,

        return adaptive_extrusion_split(
            self, extrusion_fun, split_threshold, split_par)

    def split(self, cut_distances, extrusion_multipliers=None):
        """
        Return a list of Move instances with the beginning and end being the
        same as the original move (`self`).

        Parameters:
        -----------

        cut_distances : iterable of floats between 0 and 1.
        """
        if extrusion_multipliers is None:
            extrusion_multipliers = np.ones(len(cut_distances) + 1)

        total_length = np.linalg.norm(self.get_coors() - self.start_coors)

        end_points = [self.start_coors] + [
            self.get_coors(cd) for cd in cut_distances
        ] + [self.get_coors()]

        out = [Move.from_str(str(self)) for _ in end_points[:-1]]

        # fix coordinates and extrusions
        for move, x0, x1, ext in zip(
                out, end_points[:-1], end_points[1:], extrusion_multipliers):
            move.start_coors = x0
            move.set_end_coors(x1)

            if 'E' in move.arguments:
                cut_length = np.linalg.norm(x1 - x0)
                move.arguments['E'] = (
                    cut_length / total_length  if total_length > 0 else 0.
                ) * move.arguments['E'] * ext

        return out

class Instruction:
    def __init__(self, instruction, comment, move=None):
        self.instruction = instruction
        self.comment = comment
        self.move = move

    @classmethod
    def from_str(cls, string, start_coors=np.zeros(3)):
        clean_str = string.strip()

        comment_split = clean_str.split(';', maxsplit=1)
        if len(comment_split) > 1:
            clean_str, comment = comment_split
        else:
            clean_str, comment = comment_split[0], ''

        move = Move.from_str(string, start_coors)

        return cls(clean_str, comment, move)

    def __str__(self):
        out = self.instruction
        if  self.comment:
            if out:
                out += ' '

            out += '; ' + self.comment
        return out

    def plot(self, *args, ax=None, zero_tol=1e-2, **kwargs):
        if not self.move:
            return

        if self.move.get_length() <= 0.:
            return

        if self.move.get_extrusion() / self.move.get_length() < zero_tol:
            return

        if ax is None:
            ax = plt

        ax.plot(
            [self.move.start_coors[0], self.move.get_coors()[0]],
            [self.move.start_coors[1], self.move.get_coors()[1]],
            *args, **kwargs)

class GCode:
    """
    Represents a sequence of gcode instructions
    """
    def __init__(self, instructions):
        self.instructions = instructions

    @classmethod
    def from_strings(cls, strings):
        start_coors = np.zeros(3)
        def parse_line(line):
            instruction = Instruction.from_str(line, start_coors)
            if instruction.move is not None:
                start_coors[:] = instruction.move.get_coors(1)
            return instruction

        instructions = [parse_line(line) for line in strings]
        return cls(instructions)

    @classmethod
    def from_file(cls, file_name):
        with open(file_name, 'r') as gf:
            lines = gf.readlines()

        return cls.from_strings(lines)

    def save_as(self, file_name):
        with open(file_name, 'w') as gf:
            gf.writelines('\n'.join(map(str, self.instructions)))

    def plot(self, *args, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        for instruction in self.instructions:
            instruction.plot(*args, ax=ax, **kwargs)

    def get_last_coors(self):
        out = np.zeros(3)
        for inst in self.instructions[::-1]:
            if inst.move:
                out = inst.move.get_coors(1)
                break
        return out
