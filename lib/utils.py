#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************

import datetime
import re
import sys
import os
import shutil
import readline
import numpy as np
from dataclasses import dataclass
from error import Error, exception_hook
import subprocess as sp
from globals import DEBUG, PRINT
from logger import log as logging
from typing import Optional, Any, Iterable
import sympy


class InDir:
    "small context to perform part of code in other directory"
    old = ""

    def __init__(self, dir: str):
        self.old = os.getcwd()
        self.dir = dir

    def __enter__(self):
        os.chdir(self.dir)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.chdir(self.old)
        if exc_type is not None:
            exception_hook(exc_type, exc_value, exc_traceback)


# Because itertools.batched only in python >=3.12
def batched(it: Iterable, n: int = 2):
    l = len(it)
    for ndx in range(0, l, n):
        yield it[ndx : ndx + n]


def convert_list(raw_list: list, new_type: Any = int) -> list:
    output = raw_list
    if isinstance(raw_list[0], list):
        output = [convert_list(x, new_type) for x in raw_list]
    else:
        return list(map(new_type, output))
    return output


def convert_dict(raw_dict: dict, new_type: Any = int) -> dict:
    keys = list(raw_dict.keys())
    if isinstance(raw_dict[keys[0]], dict):
        return {k: convert_dict(v, new_type) for k, v in raw_dict.items()}
    else:
        return {k: new_type(v) for k, v in raw_dict.items()}


# ======================================================================= #
def get_bool_from_env(name: str, default=False):
    var = default
    if name in os.environ and os.environ[name].lower() in [name, "false"]:
        var = os.environ[name] == "true"
    return var


# ======================================================================= #
def expand_path(path: str) -> str:
    """
    Expand variables in path, error out if variable is not resolvable
    """
    path1 = path.replace("$$", str(os.getpid()))
    expand = os.path.abspath(os.path.expanduser(os.path.expandvars(path1)))
    if "$" in expand:
        logging.error(f"Undefined env variable in {expand}")
        raise OSError(f"Undefined env variable in {expand}")
    return expand


def is_exec(path: str) -> bool:
    """
    Checks if path contains an executable (also searches in $PATH)
    """

    fpath, _ = os.path.split(path)
    if fpath:
        return os.path.isfile(path) and os.access(path, os.X_OK)
    else:
        for p in os.environ.get("PATH", "").split(os.pathsep):
            exe_file = os.path.join(p, path)
            if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
                return True
    return False


# ======================================================================= #
def question(question, typefunc, KEYSTROKES=None, default=None, autocomplete=True, ranges=False):
    if typefunc == int or typefunc == float:
        if default is not None and not isinstance(default, list):
            logging.error("Default for int or float questions must be list!")
            raise RuntimeError("Default for int or float questions must be list!")
    if typefunc == str and autocomplete:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")  # activate autocomplete
    else:
        readline.parse_and_bind("tab: ")  # deactivate autocomplete

    while True:
        s = question
        if default is not None:
            if typefunc == bool or typefunc == str:
                s += " [%s]" % (str(default))
            elif typefunc == int or typefunc == float:
                s += " ["
                for i in default:
                    s += str(i) + " "
                s = s[:-1] + "]"
        if typefunc == str and autocomplete:
            s += " (autocomplete enabled)"
        if typefunc == int and ranges:
            s += " (range comprehension enabled)"
        s += " "

        line = input(s)
        line = re.sub(r"\s+#.*$", "", line).strip()
        if not typefunc == str:
            line = line.lower()

        if line == "" or line == "\n":
            if default is not None:
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return default
            else:
                continue

        if typefunc == bool:
            posresponse = ["y", "yes", "true", "t", "ja", "si", "yea", "yeah", "aye", "sure", "definitely", "ok"]
            negresponse = ["n", "no", "false", "f", "nein", "nope"]
            if line in posresponse:
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return True
            elif line in negresponse:
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return False
            else:
                logging.warning("I didn" "t understand you.")
                continue

        if typefunc == str:
            KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
            return line

        if typefunc == float:
            # float will be returned as a list
            f = line.split()
            try:
                for i in range(len(f)):
                    f[i] = typefunc(f[i])
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return f
            except ValueError:
                logging.warning("Please enter floats!")
                continue

        if typefunc == int:
            # int will be returned as a list
            f = line.split()
            out = []
            try:
                for i in f:
                    if ranges and "~" in i:
                        q = i.split("~")
                        for j in range(int(q[0]), int(q[1]) + 1):
                            out.append(j)
                    else:
                        out.append(int(i))
                KEYSTROKES.write(line + " " * (40 - len(line)) + " #" + s + "\n")
                return out
            except ValueError:
                if ranges:
                    logging.warning('Please enter integers or ranges of integers (e.g. "-3~-1  2  5~7")!')
                else:
                    logging.warning("Please enter integers!")
                continue


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def readfile(filename) -> list[str]:
    """reads file from path and returns list of lines.
    Preferrably used for small files (for larger ones use buffer)."""
    try:
        f = open(filename)
    except IOError as e:
        raise Error(f"File {filename} does not exist!", 1)
    else:
        out = f.readlines()
        f.close()
    return out


def parse_xyz(line: str) -> tuple[str, list[float]]:
    match = re.match(r"([a-zA-Z]{1,2}\d?)(((\s+[\-\+]?\d+\.\d*)([eE][\+\-]?\d*)?){3,6})", line.strip())
    if match:
        return match[1], list(map(float, match[2].split()[:3]))
    else:
        raise Error(f"line is not xyz\n\n{line}", 43)


# ======================================================================= #


def writefile(filename, content):
    # content can be either a string or a list of strings
    try:
        f = open(filename, "w")
        if isinstance(content, list):
            for line in content:
                f.write(line)
        elif isinstance(content, str):
            f.write(content)
        else:
            print("Content %s cannot be written to file!" % (content))
        f.close()
    except IOError:
        raise Error("Could not write to file %s!" % (filename), 13)


# ======================================================================= #


def mkdir(DIR, crucial=True, force=True):
    # mkdir the DIR, or clean it if it exists
    if os.path.exists(DIR):
        if os.path.isfile(DIR) and crucial:
            raise Error("%s exists and is a file!" % (DIR), 89)
        elif os.path.isdir(DIR) and force:
            shutil.rmtree(DIR)
            os.makedirs(DIR)
    else:
        try:
            os.makedirs(DIR)
        except OSError:
            if crucial:
                raise Error("Can not create %s\n" % (DIR), 90)


# ======================================================================= #


def link(PATH, NAME, crucial=True, force=True):
    # do not create broken links
    if not os.path.exists(PATH):
        raise Error("Source %s does not exist, cannot create link!" % (PATH), 91)
    if os.path.islink(NAME):
        if not os.path.exists(NAME):
            # NAME is a broken link, remove it so that a new link can be made
            os.remove(NAME)
        else:
            # NAME is a symlink pointing to a valid file
            if force:
                # remove the link if forced to
                os.remove(NAME)
            else:
                print("%s exists, cannot create a link of the same name!" % (NAME))
                if crucial:
                    sys.exit(92)
                else:
                    return
    elif os.path.exists(NAME):
        # NAME is not a link. The interface will not overwrite files/directories with links, even with force=True
        print("%s exists, cannot create a link of the same name!" % (NAME))
        if crucial:
            sys.exit(93)
        else:
            return
    os.symlink(PATH, NAME)


# ======================================================================= #


def shorten_DIR(string) -> str:
    maxlen = 40
    front = 12
    if len(string) > maxlen:
        return string[0:front] + "..." + string[-(maxlen - 3 - front) :]
    else:
        return string + " " * (maxlen - len(string))


# ======================================================================= #


# ======================================================================= #
def strip_dir(WORKDIR, keep_files=[]):
    ls = os.listdir(WORKDIR)
    for ifile in ls:
        delete = True
        for k in keep_files:
            if k in ifile:
                delete = False
        if delete:
            rmfile = os.path.join(WORKDIR, ifile)
            if not DEBUG:
                os.remove(rmfile)


def cleandir(directory):
    if not os.path.isdir(directory):
        return
    for data in os.listdir(directory):
        path = directory + "/" + data
        if os.path.isfile(path) or os.path.islink(path):
            if DEBUG:
                print("rm %s" % (path))
            try:
                os.remove(path)
            except OSError:
                print("Could not remove file from directory: %s" % (path))
        else:
            if DEBUG:
                print("")
            cleandir(path)
            os.rmdir(path)
            if DEBUG:
                print("rm %s" % (path))
    if PRINT:
        print("===> Cleaning up directory %s" % (directory))


def save_data(scratchdir, savedir):
    # copy files to savedir
    saveable = ["mos", "coord"]
    for i in saveable:
        fromfile = os.path.join(scratchdir, "JOB", i)
        tofile = os.path.join(savedir, i)
        shutil.copy(fromfile, tofile)


# TODO: This is COLUMBUS-specific. Copying initial orbitals in general is specific.
def getmo(mofile, scratchdir):
    if os.path.exists(mofile):
        tofile = os.path.join(scratchdir, "JOB", "mos")
        shutil.copy(mofile, tofile)
    else:
        raise Error("Could not find mocoef-file %s!" % (mofile), 94)


def isbinary(path) -> bool:
    return re.search(r":.* text", sp.Popen(["file", "-L", path], stdout=sp.PIPE).stdout.read()) is None


# ======================================================================= #
def eformat(f, prec, exp_digits) -> str:
    """Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

    String looks like:
    [ -][0-9]\\.[0-9]*E[+-][0-9]*

    Arguments:
    1 float: Number to format
    2 integer: Number of decimals
    3 integer: Number of exponent digits

    Returns:
    1 string: formatted number"""

    s = "% .*e" % (prec, f)
    mantissa, exp = s.split("e")
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))


# ======================================================================= #


def removekey(d, key) -> dict:
    """Removes an entry from a dictionary and returns the dictionary.

    Arguments:
    1 dictionary
    2 anything which can be a dictionary keyword

    Returns:
    1 dictionary"""

    if key in d:
        r = dict(d)
        del r[key]
        return r
    return d


# ======================================================================= #         OK


def containsstring(string, line) -> bool:
    """Takes a string (regular expression) and another string.
    Returns True if the first string is contained in the second string.

    Arguments:
    1 string: Look for this string
    2 string: within this string

    Returns:
    1 boolean"""

    return bool(re.search(string, line))


# ======================================================================= #


class clock:
    def __init__(self, starttime: datetime = None, verbose=False):
        if starttime is None:
            self._starttime = datetime.datetime.now()
        else:
            self._starttime = starttime
        self._verbose = verbose

    @property
    def starttime(self) -> datetime.datetime:
        return self._starttime

    @starttime.setter
    def starttime(self, value):
        self._starttime = value

    def measuretime(self, log=print):
        """Calculates the time difference between global variable starttime and the time of the call of measuretime.
        Prints the Runtime, if PRINT or DEBUG are enabled.
        Arguments:
        none
        Returns:
        1 float: runtime in seconds"""

        endtime = datetime.datetime.now()
        runtime = endtime - self._starttime
        if log:
            hours = runtime.seconds // 3600
            minutes = runtime.seconds // 60 - hours * 60
            seconds = runtime.seconds % 60
            seconds += 1.0e-6 * runtime.microseconds
            log("==> Runtime:\t%i Days\t%i Hours\t%i Minutes\t%f Seconds\n\n" % (runtime.days, hours, minutes, seconds))
        return runtime.days * 24 * 3600 + runtime.seconds + runtime.microseconds // 1.0e6


# =============================================================================================== #
# =============================================================================================== #
# ============================= iterator routines  ============================================== #
# =============================================================================================== #
# =============================================================================================== #


# ======================================================================= #
def itmult(states):
    for i in range(len(states)):
        if states[i] < 1:
            continue
        yield i + 1
    return


# ======================================================================= #


def itnmstates(states: list[int]):
    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.0
    return


# =============================================================================================== #
# =============================================================================================== #
# ======================================= Matrix initialization ================================= #
# =============================================================================================== #
# =============================================================================================== #
def triangular_to_full_matrix(triangular_array: np.ndarray, num_basis_func: int, triangular="lower"):
    assert triangular in ["lower", "upper"]
    tril = np.zeros((num_basis_func, num_basis_func))
    idx = np.tril_indices(num_basis_func) if triangular == "lower" else np.triu_indices(num_basis_func)
    tril[idx] = triangular_array
    matrix = tril.T + tril
    np.fill_diagonal(matrix, np.diag(tril))
    return matrix


# ======================================================================= #         OK
def makecmatrix(a, b) -> list[list[complex]]:
    """Initialises a complex axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of complex"""

    return [x[:] for x in [[complex(0.0, 0.0)] * a] * b]  # make shallow copies (otherwise same object is referenced)


# ======================================================================= #         OK


def makermatrix(a, b) -> list[list[float]]:
    """Initialises a real axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of real"""

    return [x[:] for x in [[0.0] * a] * b]  # make shallow copies (otherwise same object is referenced)


def safe_cast(val, type, fallback=None):
    try:
        return type(val)
    except ValueError:
        return fallback


def list2dict(ls: list) -> dict:
    return {i: value for i, value in enumerate(ls)}


def build_basis_dict(
    atom_symbols: list, shell_types: list, n_prim: list, s_a_map: list, prim_exp: list, contr_coeff: list, ps_contr_coeff=None
) -> dict:
    # print(atom_symbols, shell_types, n_prim, s_a_map, prim_exp, contr_coeff, ps_contr_coeff)
    n_a = {i + 1: f"{a.upper()}{i+1}" for i, a in enumerate(atom_symbols)}
    basis = {k: [] for k in n_a.values()}
    it = 0
    for st, n_p, a in zip(shell_types, n_prim, s_a_map):
        shell = list(map(lambda x: (prim_exp[x], contr_coeff[x]), range(it, it + n_p)))
        if ps_contr_coeff and ps_contr_coeff[it] != 0.0:
            shell2 = list(map(lambda x: (prim_exp[x], ps_contr_coeff[x]), range(it, it + n_p)))
            basis[n_a[a]].append([0, *shell])
            basis[n_a[a]].append([abs(st), *shell2])
        else:
            basis[n_a[a]].append([abs(st), *shell])
        it += n_p
    #  for i in basis.keys():
    #  basis[i] = sorted(basis[i], key=lambda x: x[0])
    return basis


def get_pyscf_order_from_orca(atom_symbols, basis_dict):
    """
    Generates the reorder list to reorder atomic orbitals (from ORCA) to pyscf.

    Sources:
    ORCA: https://orcaforum.kofo.mpg.de/viewtopic.php?f=8&p=23158&t=5433&sid=f41177ec0888075a3b1e7fa438b77bd2
    pyscf:  https://pyscf.org/user/gto.html#ordering-of-basis-function

    Parameters
    ----------
    atom_symbols : list[str]
        list of element symbols for all atoms (same order as AOs)
    basis_dict : dict[str, list]
        basis set for each atom in pyscf format
    """
    #  return matrix

    # in the case of P(S=P) coefficients the order is 1S, 2S, 2Px, 2Py, 2Pz, 3S in gaussian and pyscf
    # from orca order: z, x, y
    # to  pyscf order: x, y, z
    p_order = [1, 2, 0]
    np = 3

    # from orca order: z2, xz, yz, x2-y2, xy
    # to  pyscf order: xy, yz, z2, xz, x2-y2
    d_order = [4, 2, 0, 1, 3]
    nd = 5

    # F shells spherical:
    # orca  order: zzz, xzz, yzz, xxz-yyz, xyz, xxx-xyy, xxy
    # pyscf order: xxy, xyz, yzz, zzz, xzz, xxz-yyz, xxx-xyy
    f_order = [6, 4, 2, 0, 1, 3, 5]
    nf = 7

    # compile the new_order for the whole matrix
    new_order = []
    it = 0
    for i, a in enumerate(atom_symbols):
        key = f"{a}{i+1}"
        #       s  p  d  f
        n_bf = [0, 0, 0, 0]

        # count the shells for each angular momentun
        for shell in basis_dict[key]:
            n_bf[shell[0]] += 1
        print("n_bf for", key, n_bf)

        s, p = n_bf[0:2]
        new_order.extend([it + n for n in range(s)])

        it += s
        assert it == len(new_order)

        # do p shells
        for x in range(p):
            new_order.extend([it + n for n in p_order])
            it += np

        # do d shells
        for x in range(n_bf[2]):
            new_order.extend([it + n for n in d_order])
            it += nd

        # do f shells
        for x in range(n_bf[3]):
            new_order.extend([it + n for n in f_order])
            it += nf
        assert it == len(new_order)

    return new_order


def get_cart2sph_matrix(angular_m: int, n_ao: int, atom_symbols: list[str], basis_dict) -> np.ndarray:
    from pyscf import gto
    from scipy.linalg import block_diag

    assert angular_m in [2, 3]
    # c_tensor defaults to identity matrix
    cart2sph_l = gto.cart2sph(angular_m, c_tensor=None, normalized="sp")
    n_cart, n_sph = cart2sph_l.shape
    #  assert n_cart == n_sph

    # construct full transformation matrix
    blocks = []
    #  it = 0
    for i, a in enumerate(atom_symbols):
        key = f"{a.upper()}{i+1}"

        for shell in basis_dict[key]:
            # get start indices for transformation points
            if shell[0] == angular_m:
                blocks.append(cart2sph_l)
            else:
                n = 2 * shell[0] + 1
                blocks.append(np.eye(n, dtype=float))
    return block_diag(*blocks)

    # increment iterator accordingly assuming just the specified angular momentum is cartesian
    #  it += 2 * shell[0] + 1
    #  if angular_m == 2:
    #  it += 1
    #  if angular_m == 3:
    #  it += 3


def euclidean_distance_einsum(X, Y):
    """Efficiently calculates the euclidean distance
    between two vectors using Numpys einsum function.

    Parameters
    ----------
    X : array, (n_samples x d_dimensions)
    Y : array, (n_samples x d_dimensions)

    Returns
    -------
    D : array, (n_samples, n_samples)
    """
    XX = np.einsum("ij,ij-> i", X, X)[:, np.newaxis]
    YY = np.einsum("ij,ij-> i", Y, Y)
    #    XY = 2 * np.einsum('ij,kj->ik', X, Y)
    XY = 2 * np.dot(X, Y.T)
    return np.sqrt(XX + YY - XY)


@dataclass
class ATOM:
    id: int
    qm: bool
    symbol: str
    xyz: list[float, float, float]
    bonds: set[int]

    def __str__(self):
        return "{: >5}  {: <4}  {: <16.12f} {: <16.12f} {: <16.12f}  {}".format(
            self.id + 1, self.symbol, *self.xyz, " ".join(map(lambda x: str(x + 1), sorted(self.bonds)))
        )

    def __gt__(self, other):
        return self.id > other.id

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.id == other.id


def truncate_states_in_array(array: np.ndarray, old_states: list[int], new_states: list[int], dim: int = None):
    """
    truncates the state space of an array to a smaller one

    [S,S,S,S, T,T,T,T, T,T,T,T, T,T,T,T] -> [S,S, T, T, T]

    if dim> 1: truncates first two indices!
    ---
    Parameters:
    - array: np.ndarray
    - old_states: list[int]  # list with old states
    - new_states: list[int]  # list with new states
    - dim: int # number of dims to truncates [1,2]

    ---
    Returns:
    np.ndarray
    """
    if (len(new_states) > len(old_states)) or any(a > b for (a, b) in zip(new_states, old_states)):
        raise ValueError(f"states are inconsistent! {new_states} is not a subset of {old_states}")

    if dim is None:
        dim = 1 if len(array.shape) == 1 else 2
    elif dim > len(array.shape):
        raise ValueError(f"{dim =} exceeds {len(array.shape) =}")
    new_nmstates = sum((i + 1) * n for i, n in enumerate(new_states))
    new_shape = tuple(new_nmstates if i < dim else array.shape[i] for i in range(len(array.shape)))

    new_arr = np.zeros(new_shape, dtype=array.dtype)

    leading0 = 0
    for i, nn in enumerate(new_states):
        if nn != 0:
            leading0 = i
            break

    start = sum((i + 1) * old_states[i] for i in range(leading0))
    start_new = 0
    for im, (n, nr) in filter(lambda x: x[1][1] != 0, enumerate(zip(old_states, new_states))):
        stop_new = start_new + nr
        stop = start + nr
        if dim == 1:
            new_arr[start_new:stop_new, ...] = array[start:stop, ...]
        else:
            new_arr[start_new:stop_new, start_new:stop_new, ...] = array[start:stop, start:stop, ...]

        for x in range(1, im + 1):
            s1 = start + n * x
            s1_new = start_new + nr * x
            s2 = s1 + nr
            s2_new = s1_new + nr
            if dim == 1:
                new_arr[s1_new:s2_new, ...] = array[s1:s2, ...]
            else:
                new_arr[s1_new:s2_new, s1_new:s2_new, ...] = array[s1:s2, s1:s2, ...]
        start += n * (im + 1)
        start_new += nr * (im + 1)

    return new_arr


def get_rot(theta: float, axis: int) -> np.ndarray:
    """Creates a rotation matrix 3x3 around given axis
    Parameters:
    theta: degree of rotation in degree
    axis: axis of rotation
    """
    if axis not in {0, 1, 2}:
        raise ValueError("axis not in {0,1,2}!")
        # NOTE: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rad = np.radians(theta)
    c, s = np.cos(rad), np.sin(rad)
    R = np.zeros((3, 3))
    R[axis, axis] = 1.0
    if axis == 0:
        R[1:, 1:] = np.array(((c, -s), (s, c)))
        return R
    elif axis == 1:
        R[0, 0] = c
        R[0, -1] = -s
        R[-1, 0] = s
        R[-1, -1] = c
    else:
        R[:-1, :-1] = np.array(((c, -s), (s, c)))
    return R


def Arabic2Roman(number):
    num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12
    result = ""

    while number:
        div = number // num[i]
        number %= num[i]

        while div:
            result += sym[i]
            div -= 1
        i -= 1
    return result


def mult2symbol(mult):
    if mult < 5:
        return "SDTQ"[mult - 1]
    return Arabic2Roman(mult)


@dataclass()
class electronic_state:
    """
    class to store electronic state information

    Properties:
    ---
    Z: int  Charge of the state
    S: int  Two times S quantum number (so that it is always integer), 0 for singlets, 1 for doublets, 2 for triplets etc.
    M: int  Two times M_S quantum number (so that it is always integer)
    N: int  Ordinal number of the state for its S, starting from 1
    C: dict  Anyone can add any comment about the state as an item here. Not used in hashing or comparing of electronic_state instance(s).
    """

    Z: int
    S: int
    M: int
    N: int
    C: Optional[dict] = None

    def __eq__(self, other):
        # The 'equal' operator is overloaded with the function that compares
        # only Z, S and N (not M). Comparison of 'full' electronic states
        # is not implemetented and it is supposed to be done by reference comparison
        # e.g. 'if state1 is state2:'
        return self.Z == other.Z and self.S == other.S and self.M == other.M and self.N == other.N

    def __floordiv__(self, other):
        return self.Z == other.Z and self.S == other.S and self.N == other.N

    def __gt__(self, other):
        ord1 = self.S * 1_000_000 + self.M * 1000 + self.N
        ord2 = other.S * 1_000_000 + other.M * 1000 + other.N

        return ord1 > ord2

    def __lt__(self, other):
        ord1 = self.S * 1_000_000 + self.M * 1000 + self.N
        ord2 = other.S * 1_000_000 + other.M * 1000 + other.N
        return ord1 < ord2

    def __hash__(self):
        return f"{self.Z} {self.S} {self.N} {self.M}".__hash__()

    def symbol(self, Z=True, M=True):
        string = mult2symbol(self.S + 1)
        if self.S <= 1:
            string += str(self.N - 1)
        else:
            string += str(self.N)
        if M:
            string += "_"
            if self.M == 0:
                string += "(0)"
            elif self.M % 2 == 0:
                string += f"({self.M//2:+d})"
            else:
                string += f"({self.M:+d}/2)"
        if Z:
            string += "^"
            if self.Z == 0:
                string += "(0)"
            elif self.Z > 0:
                string += "(" + str(self.Z) + "+)"
            else:
                string += "(" + str(abs(self.Z)) + "-)"
        return string

    def __repr__(self):
        return self.symbol()


def density_representation(d):  # To pring the density tuple. Can also be used for Dyson-orbital printing
    s1, s2, spin = d
    return f"[ {s1.symbol():<12s} {spin:-^6}> {s2.symbol():<12s} ]"
    #  return "[ " + s1.symbol() + " " + middle + " " + s2.symbol() + " ]"


def loewdin_atomic_charge_transfer_numbers(mol, dm: np.ndarray, s_root: np.ndarray):
    """Analysis of charge transfer numbers from loewdin analysis

    Args:
        mol (): gto.Mole object
        dm: density matrix in AO basis
        s_root: S^(1/2) where S is the AO overlap matrix
    """
    from pyscf.gto import Mole as mole

    pop = np.einsum("mi,ij,jn->mn", s_root, dm, s_root, optimize=True, casting="no") ** 2
    # pop = s_root @ dm @ s_root
    aorange = mole.aoslice_by_atom(mol)

    chrg = np.zeros((mol.natm, mol.natm))

    for i, (_, _, ao_start_i, ao_stop_i) in enumerate(aorange):
        for j, (_, _, ao_start_j, ao_stop_j) in enumerate(aorange):
            chrg[i, j] += np.sum(np.sum(pop[ao_start_i:ao_stop_i, :], axis=0)[ao_start_j:ao_stop_j])

    return chrg


# ======================================================================= #         OK

def phase_correction(matrix):
    """
    Do a phase correction of a matrix.
    Follows algorithm from J. Chem. Theory Comput. 2020, 16, 2, 835-846 (https://doi.org/10.1021/acs.jctc.9b00952)
    """
    phases = np.ones(matrix.shape[-1])
    U = matrix.real.copy()
    det_U = np.linalg.det(U)
    if det_U < 0:
        U[:, 0] *= -1.0  # this row/column convention is correct
        phases[0] *= -1.0
    U_sq = U * U

    # sweeps
    length = len(U)
    sweeps = 0
    done = False
    while not done:
        done = True
        for j in range(length):
            for k in range(j + 1, length):
                delta = 3.0 * (U_sq[j, j] + U_sq[k, k])
                delta += 6.0 * U[j, k] * U[k, j]
                delta += 8.0 * (U[k, k] + U[j, j])
                delta -= 3.0 * (U[j, :] @ U[:, j] + U[k, :] @ U[:, k])

                # Test if delta < 0
                num_zero_thres = -1e-15  # needs proper threshold towards 0
                if delta < num_zero_thres:
                    U[:, j] *= -1.0  # this row/column convention is correct
                    U[:, k] *= -1.0  # this row/column convention is correct
                    phases[j] *= -1.0
                    phases[k] *= -1.0
                    done = False
        sweeps += 1

    return U, phases