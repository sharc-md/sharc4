#!/usr/bin/env python3

import copy
import math
import sys
import re
import os
import stat
import shutil
import datetime
from optparse import OptionParser
import readline
import time

version = '1.0'
versionneeded = [0.2, 1.0]
versiondate = datetime.date(2014, 10, 8)

# =========================================================

def find_between(string, start, end):
    result = re.search(start + '(.*)' + end, string)
    return result.group(1)

# ======================================================================= #


def open_keystrokes():
    global KEYSTROKES
    KEYSTROKES = open('KEYSTROKES.tmp', 'w')

# ======================================================================= #


def close_keystrokes():
    KEYSTROKES.close()
    shutil.move('KEYSTROKES.tmp', 'KEYSTROKES.setup_init')

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
# ======================================================================================================================


def displaywelcome():
    print('Script for managing exit codes...\n')
    string = '\n'
    string += '  ' + '=' * 80 + '\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('SHARC Test suite run script') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Author: Sebastian Mai') + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '||' + '{:^80}'.format('Version:' + version) + '||\n'
    string += '||' + '{:^80}'.format(versiondate.strftime("%d.%m.%y")) + '||\n'
    string += '||' + '{:^80}'.format('') + '||\n'
    string += '  ' + '=' * 80 + '\n\n'
    print(string)

# ===================================


def question(question, typefunc, default=None, autocomplete=True):
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
            posresponse = ['y', 'yes', 'true', 'ja', 'si', 'yea', 'yeah', 'aye', 'sure', 'definitely']
            negresponse = ['n', 'no', 'false', 'nein', 'nope']
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

        if typefunc == int or typefunc == float:
            # int and float will be returned as a list
            f = line.split()
            try:
                for i in range(len(f)):
                    f[i] = typefunc(f[i])
                KEYSTROKES.write(line + ' ' * (40 - len(line)) + ' #' + s + '\n')
                return f
            except ValueError:
                if typefunc == int:
                    i = 1
                elif typefunc == float:
                    i = 2
                print('Please enter a %s' % (['string', 'integer', 'float'][i]))
                continue


# =========================================================
def get_general():
    INFOS = {}

    print('{:-^60}'.format('Script') + '\n')
    print('\nGive the path to the Python script.\n')
    while True:
        pyfile = question('Python script file:', str)
        if os.path.isfile(pyfile):
            break
    INFOS['pyfile'] = pyfile
    # INFOS['docfile']=os.path.basename(pyfile)+'_doc_exit_codes.txt'

    return INFOS

# =========================================================


def perform(INFOS):
    code = readfile(INFOS['pyfile'])
    excodes = []
    iline = -1
    while True:
        iline += 1
        if iline == len(code):
            break
        line = code[iline]
        cline = re.sub('#.*$', '', line)
        if 'sys.exit(' in cline:
            ex = int(find_between(cline, 'sys.exit\\(', '\\)'))
            # backsearch for previous print(message)
            for i in range(10):
                xline = re.sub('#.*$', '', code[iline - i]).strip()
                if 'print' in xline:
                    break
            else:
                xline = ''
            # backsearch for routine
            for i in range(iline):
                rline = re.sub('#.*$', '', code[iline - i]).strip()
                if 'def ' in rline:
                    break
            else:
                rline = '<main code body>'
            excodes.append((ex, xline, iline + 1, rline))
    excodes.sort(key=lambda x: x[0])
    for i in excodes:
        s = '%i\n\tmessage: %s\n\tline %i\t%s' % i
        print(s)

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def main():
    '''Main routine'''

    usage = '''
python exit_codes.py
'''

    description = ''
    parser = OptionParser(usage=usage, description=description)

    displaywelcome()
    open_keystrokes()

    INFOS = get_general()

    print('\n' + '{:#^60}'.format('Full input') + '\n')
    for item in INFOS:
        print(item, ' ' * (25 - len(item)), INFOS[item])
    print('')
    setup = question('Do you want perform this job?', bool, True)
    print('')

    if setup:
        perform(INFOS)

    close_keystrokes()


# ======================================================================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCtrl+C makes me a sad SHARC ;-(\n')
        quit(0)
