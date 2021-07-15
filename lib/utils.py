import datetime
import re
import sys
import subprocess as sp

# ======================================================================= #
def readfile(filename):
    try:
        f = open(filename)
        out = f.readlines()
        f.close()
    except IOError:
        print('File %s does not exist!' % (filename))
        sys.exit(12)
    return out

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
        print('Could not write to file %s!' % (filename))
        sys.exit(13)

# ======================================================================= #


def isbinary(path):
    return (re.search(r':.* text', sp.Popen(["file", '-L', path], stdout=sp.PIPE).stdout.read()) is None)


# ======================================================================= #
def eformat(f, prec, exp_digits):
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


def removekey(d, key):
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


def containsstring(string, line):
    '''Takes a string (regular expression) and another string. Returns True if the first string is contained in the second string.

    Arguments:
    1 string: Look for this string
    2 string: within this string

    Returns:
    1 boolean'''

    a = re.search(string, line)
    if a:
        return True
    else:
        return False

# ======================================================================= #

class clock:
    def __init__(self, starttime: datetime = None, verbose=False):
        if starttime is None:
            self.starttime = datetime.datetime.now()
        else:
            self.starttime = starttime
        self.verbose = verbose

    def measuretime(self):
        '''Calculates the time difference between global variable starttime and the time of the call of measuretime.
        Prints the Runtime, if PRINT or DEBUG are enabled.
        Arguments:
        none
        Returns:
        1 float: runtime in seconds'''

        endtime = datetime.datetime.now()
        runtime = endtime - self.starttime
        if self.verbose:
            hours = runtime.seconds // 3600
            minutes = runtime.seconds // 60 - hours * 60
            seconds = runtime.seconds % 60
            print('==> Runtime:\n%i Days\t%i Hours\t%i Minutes\t%i Seconds\n\n' % (runtime.days, hours, minutes, seconds))
        total_seconds = runtime.days * 24 * 3600 + runtime.seconds + runtime.microseconds // 1.e6
        return total_seconds


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


def itnmstates(states):

    for i in range(len(states)):
        if states[i] < 1:
            continue
        for k in range(i + 1):
            for j in range(states[i]):
                yield i + 1, j + 1, k - i / 2.
    return
