from textwrap import wrap
from utils import itnmstates


def printcomplexmatrix(matrix, states):
    print(formatcomplexmatrix(matrix,states))

# TODO: typing
def formatcomplexmatrix(matrix,states):
    '''Prints a formatted matrix. Zero elements are not printed, blocks of different mult and MS are delimited by dashes. Also prints a matrix with the imaginary parts, if any one element has non-zero imaginary part.

    Arguments:
    1 list of list of complex: the matrix
    2 list of integers: states specs'''

    nmstates = 0
    for i in range(len(states)):
        nmstates += states[i] * (i + 1)
    string = 'Real Part:\n'
    string += '-' * (11 * nmstates + nmstates // 3)
    string += '\n'
    istate = 0
    for imult, i, ms in itnmstates(states):
        jstate = 0
        string += '|'
        for jmult, j, ms2 in itnmstates(states):
            if matrix[istate][jstate].real == 0.:
                string += ' ' * 11
            else:
                string += '% .3e ' % (matrix[istate][jstate].real)
            if j == states[jmult - 1]:
                string += '|'
            jstate += 1
        string += '\n'
        if i == states[imult - 1]:
            string += '-' * (11 * nmstates + nmstates // 3)
            string += '\n'
        istate += 1
    string += '\n'
    
    imag = False
    string2 = 'Imaginary Part:\n'
    string2 += '-' * (11 * nmstates + nmstates // 3)
    string2 += '\n'
    istate = 0
    for imult, i, ms in itnmstates(states):
        jstate = 0
        string2 += '|'
        for jmult, j, ms2 in itnmstates(states):
            if matrix[istate][jstate].imag == 0.:
                string2 += ' ' * 11
            else:
                imag = True
                string2 += '% .3e ' % (matrix[istate][jstate].imag)
            if j == states[jmult - 1]:
                string2 += '|'
            jstate += 1
        string2 += '\n'
        if i == states[imult - 1]:
            string2 += '-' * (11 * nmstates + nmstates // 3)
            string2 += '\n'
        istate += 1
    string2 += '\n'
    if imag:
        string += string2
    return string

# ======================================================================= #

def printgrad(grad, natom, elements, DEBUG=False):
    print(formatgrad(grad, natom, elements, DEBUG=False))

def formatgrad(grad, natom, elements, DEBUG=False):
    '''Prints a gradient or nac vector. Also prints the atom elements. If the gradient is identical zero, just prints one line.

    Arguments:
    1 list of list of float: gradient
    2 integer: natom
    3 list: element name'''

    string = ''
    iszero = True
    leng = min( [ len(i) for i in grad ] )
    for atom in range(natom):
        if not DEBUG:
            if atom == 5:
                string += '...\t...\n' + '\t     ...'*leng + '\n'
            if 5 <= atom < natom - 1:
                continue
        string += '%i\t%s' % (atom + 1, elements[atom])
        for xyz in range(leng):
            if grad[atom][xyz] != 0:
                iszero = False
            g = grad[atom][xyz]
            if isinstance(g, float):
                string += '\t% .5f' % (g)
            elif isinstance(g, complex):
                string += '\t% .5f\t% .5f\t' % (g.real, g.imag)
        string += '\n'
    string += '\n'
    if iszero:
        return '\t\t...is identical zero...\n'
    else:
        return string

# ======================================================================= #

def printtheodore(matrix, QMin):
    print(formattheodore(matrix, QMin))

def formattheodore(matrix, QMin):
    string = '%6s ' % 'State'
    for i in QMin['resources']['theodore_prop']:
        string += '%6s ' % i
    for i in range(len(QMin['resources']['theodore_fragment'])):
        for j in range(len(QMin['resources']['theodore_fragment'])):
            string += '  Om%1i%1i ' % (i + 1, j + 1)
    string += '\n' + '-------' * (1 + QMin['resources']['theodore_n']) + '\n'
    istate = 0
    for imult, i, ms in itnmstates(QMin['states']):
        istate += 1
        string += '%6i ' % istate
        for i in matrix[istate - 1]:
            string += '%6.4f ' % i.real
        string += '\n'
    string += '\n'
    return string

# ======================================================================= #

def printheader(content):
    print(formatheader(content))

def formatheader(content):
    '''Prints the formatted header of the log file. Prints version number and version date
    Takes nothing, returns nothing.
    Wraps the specified content lines in as :
      ================================================================================
    ||                                                                                ||
    ||                                 content line 1                                 ||
    ||                                 content line 2                                 ||
    ||                                 content line 3                                 ||
    ||                                 content line 4                                 ||
    ||                                 content line 5                                 ||
    ||                                                                                ||
      ================================================================================
    '''

    rule = '=' * 76
    lines = [rule, '', *content, '', rule]

    # wraps Authors line in case its too long
    lines[4:5] = wrap(lines[4], width=70)
    lines[1:-1] = map(lambda s: '||{:^76}||'.format(s), lines[1:-1])
    string = '\n'.join(lines) + '\n'
    return string
