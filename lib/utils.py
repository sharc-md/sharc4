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


class InDir():
    "small context to perform part of code in other directory"
    old = ''

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


# ======================================================================= #
def get_bool_from_env(name: str, default=False):
    var = default
    if name in os.environ and os.environ[name].lower() in [name, "false"]:
        var = os.environ[name] == "true"
    return var

def expand_path(path: str) -> str:
    """
    Expand variables in path, error out if variable is not resolvable
    """
    expand = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    assert "$" not in expand, f"Undefined env variable in {expand}"
    return expand

def question(question, typefunc, KEYSTROKES=None, default=None, autocomplete=True, ranges=False):
    if typefunc == int or typefunc == float:
        if default is not None and not isinstance(default, list):
            print('Default to int or float question must be list!')
            quit(1)
    if typefunc == str and autocomplete:
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")    # activate autocomplete
    else:
        readline.parse_and_bind("tab: ")            # deactivate autocomplete

    while True:
        s = question
        if default is not None:
            if typefunc == bool or typefunc == str:
                s += ' [%s]' % (str(default))
            elif typefunc == int or typefunc == float:
                s += ' ['
                for i in default:
                    s += str(i) + ' '
                s = s[:-1] + ']'
        if typefunc == str and autocomplete:
            s += ' (autocomplete enabled)'
        if typefunc == int and ranges:
            s += ' (range comprehension enabled)'
        s += ' '

        line = input(s)
        line = re.sub('#.*$', '', line).strip()
        if not typefunc == str:
            line = line.lower()

        if line == '' or line == '\n':
            if default is not None:
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return default
            else:
                continue

        if typefunc == bool:
            posresponse = ['y', 'yes', 'true', 't', 'ja', 'si', 'yea', 'yeah', 'aye', 'sure', 'definitely']
            negresponse = ['n', 'no', 'false', 'f', 'nein', 'nope']
            if line in posresponse:
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return True
            elif line in negresponse:
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return False
            else:
                print('I didn''t understand you.')
                continue

        if typefunc == str:
            KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
            return line

        if typefunc == float:
            # float will be returned as a list
            f = line.split()
            try:
                for i in range(len(f)):
                    f[i] = typefunc(f[i])
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return f
            except ValueError:
                print('Please enter floats!')
                continue

        if typefunc == int:
            # int will be returned as a list
            f = line.split()
            out = []
            try:
                for i in f:
                    if ranges and '~' in i:
                        q = i.split('~')
                        for j in range(int(q[0]), int(q[1]) + 1):
                            out.append(j)
                    else:
                        out.append(int(i))
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return out
            except ValueError:
                if ranges:
                    print('Please enter integers or ranges of integers (e.g. "-3~-1  2  5~7")!')
                else:
                    print('Please enter integers!')
                continue

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

def readfile(filename) -> list[str]:
    '''reads file from path and returns list of lines.
    Preferrably used for small files (for larger ones use buffer).'''
    try:
        f = open(filename)
    except IOError as e:
        raise Error(f'File {filename} does not exist!', 1)
    else:
        out = f.readlines()
        f.close()
    return out


def parse_xyz(line: str) -> tuple[str, list[float]]:
    match = re.match(r'([a-zA-Z]{1,2}\d?)((\s+-?\d+\.\d*){3,6})', line.strip())
    if match:
        return match[1], list(map(float, match[2].split()[:3]))
    else:
        raise Error(f"line is not xyz\n\n{line}", 43)


# ======================================================================= #


def writefile(filename, content):
    # content can be either a string or a list of strings
    try:
        f = open(filename, 'w')
        if isinstance(content, list):
            for line in content:
                f.write(line)
        elif isinstance(content, str):
            f.write(content)
        else:
            print('Content %s cannot be written to file!' % (content))
        f.close()
    except IOError:
        raise Error('Could not write to file %s!' % (filename), 13)


# ======================================================================= #


def mkdir(DIR, crucial=True, force=True):
    # mkdir the DIR, or clean it if it exists
    if os.path.exists(DIR):
        if os.path.isfile(DIR) and crucial:
            raise Error('%s exists and is a file!' % (DIR), 89)
        elif os.path.isdir(DIR) and force:
            shutil.rmtree(DIR)
            os.makedirs(DIR)
    else:
        try:
            os.makedirs(DIR)
        except OSError:
            if crucial:
                raise Error('Can not create %s\n' % (DIR), 90)


# ======================================================================= #


def link(PATH, NAME, crucial=True, force=True):
    # do not create broken links
    if not os.path.exists(PATH):
        raise Error('Source %s does not exist, cannot create link!' % (PATH), 91)
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
                print('%s exists, cannot create a link of the same name!' % (NAME))
                if crucial:
                    sys.exit(92)
                else:
                    return
    elif os.path.exists(NAME):
        # NAME is not a link. The interface will not overwrite files/directories with links, even with force=True
        print('%s exists, cannot create a link of the same name!' % (NAME))
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
        return string[0:front] + '...' + string[-(maxlen - 3 - front):]
    else:
        return string + ' ' * (maxlen - len(string))


# ======================================================================= #


def cleandir(directory):
    if not os.path.isdir(directory):
        return
    for data in os.listdir(directory):
        path = directory + '/' + data
        if os.path.isfile(path) or os.path.islink(path):
            if DEBUG:
                print('rm %s' % (path))
            try:
                os.remove(path)
            except OSError:
                print('Could not remove file from directory: %s' % (path))
        else:
            if DEBUG:
                print('')
            cleandir(path)
            os.rmdir(path)
            if DEBUG:
                print('rm %s' % (path))
    if PRINT:
        print('===> Cleaning up directory %s' % (directory))


def save_data(scratchdir, savedir):
    # copy files to savedir
    saveable = ['mos', 'coord']
    for i in saveable:
        fromfile = os.path.join(scratchdir, 'JOB', i)
        tofile = os.path.join(savedir, i)
        shutil.copy(fromfile, tofile)

# TODO: This is COLUMBUS-specific. Copying initial orbitals in general is specific.
def getmo(mofile, scratchdir):
    if os.path.exists(mofile):
        tofile = os.path.join(scratchdir, 'JOB', 'mos')
        shutil.copy(mofile, tofile)
    else:
        raise Error('Could not find mocoef-file %s!' % (mofile), 94)


def isbinary(path) -> bool:
    return (re.search(r':.* text', sp.Popen(["file", '-L', path], stdout=sp.PIPE).stdout.read()) is None)


# ======================================================================= #
def eformat(f, prec, exp_digits) -> str:
    '''Formats a float f into scientific notation with prec number of decimals and exp_digits number of exponent digits.

    String looks like:
    [ -][0-9]\\.[0-9]*E[+-][0-9]*

    Arguments:
    1 float: Number to format
    2 integer: Number of decimals
    3 integer: Number of exponent digits

    Returns:
    1 string: formatted number'''

    s = "% .*e" % (prec, f)
    mantissa, exp = s.split('e')
    return "%sE%+0*d" % (mantissa, exp_digits + 1, int(exp))


# ======================================================================= #


def removekey(d, key) -> dict:
    '''Removes an entry from a dictionary and returns the dictionary.

    Arguments:
    1 dictionary
    2 anything which can be a dictionary keyword

    Returns:
    1 dictionary'''

    if key in d:
        r = dict(d)
        del r[key]
        return r
    return d


# ======================================================================= #         OK


def containsstring(string, line) -> bool:
    '''Takes a string (regular expression) and another string.
    Returns True if the first string is contained in the second string.

    Arguments:
    1 string: Look for this string
    2 string: within this string

    Returns:
    1 boolean'''

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

    def measuretime(self):
        '''Calculates the time difference between global variable starttime and the time of the call of measuretime.
        Prints the Runtime, if PRINT or DEBUG are enabled.
        Arguments:
        none
        Returns:
        1 float: runtime in seconds'''

        endtime = datetime.datetime.now()
        runtime = endtime - self._starttime
        if self._verbose:
            hours = runtime.seconds // 3600
            minutes = runtime.seconds // 60 - hours * 60
            seconds = runtime.seconds % 60
            seconds += 1.e-6 * runtime.microseconds
            print(
                '==> Runtime:\t%i Days\t%i Hours\t%i Minutes\t%f Seconds\n\n' % (runtime.days, hours, minutes, seconds)
            )
        return runtime.days * 24 * 3600 + runtime.seconds + runtime.microseconds // 1.e6


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
                yield i + 1, j + 1, k - i / 2.
    return


# =============================================================================================== #
# =============================================================================================== #
# ======================================= Matrix initialization ================================= #
# =============================================================================================== #
# =============================================================================================== #


# ======================================================================= #         OK
def makecmatrix(a, b) -> list[list[complex]]:
    '''Initialises a complex axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of complex'''

    return [x[:] for x in [[complex(0., 0.)] * a] * b]    # make shallow copies (otherwise same object is referenced)


# ======================================================================= #         OK


def makermatrix(a, b) -> list[list[float]]:
    '''Initialises a real axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of real'''

    return [x[:] for x in [[0.] * a] * b]    # make shallow copies (otherwise same object is referenced)


def safe_cast(val, type, fallback=None):
    try:
        return type(val)
    except ValueError:
        return fallback


def list2dict(ls: list) -> dict:
    return {i: value for i, value in enumerate(ls)}


def build_basis_dict(
    atom_symbols: list, shell_types, n_prim, s_a_map, prim_exp, contr_coeff, ps_contr_coeff=None
) -> dict:
    # print(atom_symbols, shell_types, n_prim, s_a_map, prim_exp, contr_coeff, ps_contr_coeff)
    n_a = {i + 1: f'{a.upper()}{i+1}' for i, a in enumerate(atom_symbols)}
    basis = {k: [] for k in n_a.values()}
    it = 0
    for st, n_p, a in zip(shell_types, n_prim, s_a_map):

        shell = list(map(lambda x: (prim_exp[x], contr_coeff[x]), range(it, it + n_p)))
        if ps_contr_coeff and ps_contr_coeff[it] != 0.:
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
        key = f'{a}{i+1}'
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


def get_pyscf_order_from_gaussian(atom_symbols, basis_dict, cartesian_d=False, cartesian_f=False, p_eq_s=False):
    """
    Generates the reorder list to reorder atomic orbitals (from GAUSSIAN) to pyscf.

    Sources:
    GAUSSIAN: https://gaussian.com/interfacing/
    pyscf:  https://pyscf.org/user/gto.html#ordering-of-basis-function

    Parameters
    ----------
    atom_symbols : list[str]
        list of element symbols for all atoms (same order as AOs)
    basis_dict : dict[str, list]
        basis set for each atom in pyscf format
    cartesian_d : bool
        whether the d-orbitals are cartesian
    cartesian_f : bool
        whether the f-orbitals are cartesian
    """
    #  return matrix

    # in the case of P(S=P) coefficients the order is 1S, 2S, 2Px, 2Py, 2Pz, 3S in gaussian and pyscf

    # if there are any d-orbitals they need to be swapped!!!
    if cartesian_d:
        # in the case of a cartesian basis the ordering is
        # gauss order:     xx, yy, zz, xy, xz, yz
        # pyscf order:     xx, xy, xz, yy, yz, zz
        d_order = [0, 3, 4, 1, 5, 2]
        #  d_order = [0, 1, 2, 3, 4, 5]
        nd = 6
    else:
        # from gauss order: z2, xz, yz, x2-y2, xy
        # to   pyscf order: xy, yz, z2, xz, x2-y2
        d_order = [4, 2, 0, 1, 3]
        nd = 5

    if cartesian_f:
        # F shells cartesian:
        # gauss order: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
        # pyscf order: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
        f_order = [0, 4, 5, 3, 9, 6, 1, 8, 7, 2]
        nf = 10
    else:
        # F shells spherical:
        # gauss order: zzz, xzz, yzz, xxz-yyz, xyz, xxx-xyy, xxy
        # pyscf order: xxy, xyz, yzz, zzz, xzz, xxz-yyz, xxx-xyy
        f_order = [6, 4, 2, 0, 1, 3, 5]
        nf = 7

    # G shells cartesian, not needed anyway
    # pyscf order: xxxx,xxxy,xxxz,xxyy,xxyz,xxzz,xyyy,xyyz,xyzz,xzzz,yyyy,yyyz,yyzz,yzzz,zzzz
    g_order = [8, 6, 4, 2, 0, 1, 3, 5, 7]
    ng = 9

    # H shells cartesian coordinates, not needed anyway
    # pyscf order: xxxxx,xxxxy,xxxxz,xxxyy,xxxyz,xxxzz,xxyyy,xxyyz,xxyzz,xxzzz,xyyyy,xyyyz,xyyzz,xyzzz,xzzzz,yyyyy,yyyyz,yyyzz,yyzzz,yzzzz,zzzzz
    h_order = [10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9]
    nh = 11

    # I shells cartesian coordinates, not needed anyway
    # pyscf order: xxxxxx,xxxxxy,xxxxxz,xxxxyy,xxxxyz,xxxxzz,xxxyyy,xxxyyz,xxxyzz,xxxzzz,xxyyyy,xxyyyz,xxyyzz,xxyzzz,xxzzzz,xyyyyy,xyyyyz,xyyyzz,xyyzzz,xyzzzz,xzzzzz,yyyyyy,yyyyyz,yyyyzz,yyyzzz,yyzzzz,yzzzzz,zzzzzz
    i_order = [12, 10, 8, 6, 4, 2, 0, 1, 3, 5, 7, 9, 11]
    ni = 13

    # compile the new_order for the whole matrix
    new_order = []
    it = 0
    for i, a in enumerate(atom_symbols):
        key = f'{a.upper()}{i+1}'
        #       s  p  d  f  g  h  i
        n_bf = [0, 0, 0, 0, 0, 0, 0]

        # count the shells for each angular momentun
        for shell in basis_dict[key]:
            n_bf[shell[0]] += 1
        #print("n_bf for", key, n_bf)

        if p_eq_s:
            #print("p_eq_s", key)
            s, p = n_bf[0:2]
            #print("nbf s:", s, " p", p)
            if s == p:
                s_order = [4 * n for n in range(s)]
                sp_order = s_order + [n for n in range(1, p * 3 + s) if (n) % 4 != 0]
            elif p == 0:
                s_order = [x for x in range(s)]
                sp_order = s_order
            else:
                s_order = [0] + [1 + 4 * n for n in range(s - 1)]
                sp_order = s_order + [n for n in range(2, p * 3 + s) if (n - 1) % 4 != 0]
            #print("p_eq_s", sp_order, len(sp_order))
            # offset new_order with iterator
            new_order.extend([it + n for n in sp_order])
        else:
            s, p = n_bf[0:2]
            new_order.extend([it + n for n in range(s + p * 3)])

        it += s + p * 3

        # do d shells
        for x in range(n_bf[2]):
            new_order.extend([it + n for n in d_order])
            it += nd

        # do f shells
        for x in range(n_bf[3]):
            new_order.extend([it + n for n in f_order])
            it += nf
        # do g shells
        for x in range(n_bf[4]):
            new_order.extend([it + n for n in g_order])
            it += ng

        # do h shells
        for x in range(n_bf[5]):
            new_order.extend([it + n for n in h_order])
            it += nh

        # do i shells
        for x in range(n_bf[6]):
            new_order.extend([it + n for n in i_order])
            it += ni

        assert it == len(new_order)

    return new_order




def get_cart2sph_matrix(angular_m: int, n_ao: int, atom_symbols: list[str], basis_dict) -> np.ndarray:
    from pyscf import gto
    from scipy.linalg import block_diag
    assert angular_m in [2, 3]
    # c_tensor defaults to identity matrix
    cart2sph_l = gto.cart2sph(angular_m, c_tensor=None, normalized='sp')
    n_cart, n_sph = cart2sph_l.shape
    #  assert n_cart == n_sph

    # construct full transformation matrix
    blocks = []
    #  it = 0
    for i, a in enumerate(atom_symbols):
        key = f'{a.upper()}{i+1}'

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
    XX = np.einsum('ij,ij-> i', X, X)[:, np.newaxis]
    YY = np.einsum('ij,ij-> i', Y, Y)
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
        return '{: >5}  {: <4}  {: <16.12f} {: <16.12f} {: <16.12f}  {}'.format(
            self.id + 1, self.symbol, *self.xyz, ' '.join(map(lambda x: str(x + 1), sorted(self.bonds)))
        )

    def __gt__(self, other):
        return self.id > other.id

    def __lt__(self, other):
        return self.id < other.id

    def __eq__(self, other):
        return self.id == other.id


def get_rot(theta: float, axis: int) -> np.ndarray:
    """Creates a rotation matrix 3x3 around given axis
    Parameters:
    theta: degree of rotation in degree
    axis: axis of rotation
    """
    if axis not in {0, 1, 2}:
        raise ValueError('axis not in {0,1,2}!')
        # NOTE: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rad = np.radians(theta)
    c, s = np.cos(rad), np.sin(rad)
    R = np.zeros((3, 3))
    R[axis, axis] = 1.
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
