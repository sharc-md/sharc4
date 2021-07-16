import datetime
import re
import sys
import os
import shutil
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


def mkdir(DIR):
    # mkdir the DIR, or clean it if it exists
    if os.path.exists(DIR):
        if os.path.isfile(DIR):
            print('%s exists and is a file!' % (DIR))
            sys.exit(89)
        elif os.path.isdir(DIR):
            shutil.rmtree(DIR)
            os.makedirs(DIR)
    else:
        try:
            os.makedirs(DIR)
        except OSError:
            print('Can not create %s\n' % (DIR))
            sys.exit(90)
# ======================================================================= #


def link(PATH, NAME, crucial=True, force=True):
    # do not create broken links
    if not os.path.exists(PATH):
        print('Source %s does not exist, cannot create link!' % (PATH))
        sys.exit(91)
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


def shorten_DIR(string):
    maxlen = 40
    front = 12
    if len(string) > maxlen:
        return string[0:front] + '...' + string[-(maxlen - 3 - front):]
    else:
        return string + ' ' * (maxlen - len(string))

# ======================================================================= #


def cleandir(directory, PRINT=True, DEBUG=False):
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


def movetoold(savedir):
    # rename all files in savedir
    saveable = ['dets', 'mos', 'coord']
    ls = os.listdir(savedir)
    if ls == []:
        return
    for f in ls:
        f2 = savedir + '/' + f
        if os.path.isfile(f2):
            if any([i in f for i in saveable]):
                if 'old' not in f:
                    fdest = f2 + '.old'
                    shutil.move(f2, fdest)


def save_data(scratchdir, savedir):
    # copy files to savedir
    saveable = ['mos', 'coord']
    for i in saveable:
        fromfile = os.path.join(scratchdir, 'JOB', i)
        tofile = os.path.join(savedir, i)
        shutil.copy(fromfile, tofile)


def getmo(mofile, scratchdir):
    if os.path.exists(mofile):
        tofile = os.path.join(scratchdir, 'JOB', 'mos')
        shutil.copy(mofile, tofile)
    else:
        print('Could not find mocoef-file %s!' % (mofile))
        sys.exit(94)


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
    def starttime(self) -> None:
        return self._starttime

    @starttime.setter
    def starttime(self, value: datetime):
        self._starttime = value

    def measuretime(self):
        '''Calculates the time difference between global variable starttime and the time of the call of measuretime.
        Prints the Runtime, if PRINT or DEBUG are enabled.
        Arguments:
        none
        Returns:
        1 float: runtime in seconds'''

        endtime = datetime.datetime.now()
        runtime = endtime - self.starttime
        if self._verbose:
            hours = runtime.seconds // 3600
            minutes = runtime.seconds // 60 - hours * 60
            seconds = runtime.seconds % 60
            print('==> Runtime:\n%i Days\t%i Hours\t%i Minutes\t%i Seconds\n\n' % (runtime.days, hours, minutes, seconds))
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


def itnmstates(states):

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
def makecmatrix(a, b):
    '''Initialises a complex axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of complex'''

    return [[complex(0., 0.) for i in range(a)] for j in range(b)]

# ======================================================================= #         OK


def makermatrix(a, b):
    '''Initialises a real axb matrix.

    Arguments:
    1 integer: first dimension
    2 integer: second dimension

    Returns;
    1 list of list of real'''

    return [[0. for i in range(a)] for j in range(b)]


def safe_cast(val, type, fallback=None):
    try:
        return type(val)
    except ValueError:
        return fallback
