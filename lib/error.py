# XXX: decide for further features
# something like this is also possible: (personalized exception handler)
# def my_exchandler(type, value, traceback):
#     print(value)
# sys.excepthook(type, value, traceback)
# This function prints out a given traceback and exception to sys.stderr.

# When an exception is raised and uncaught, the interpreter calls sys.excepthook with three arguments, the exception class, exception instance, and a traceback object. In an interactive session this happens just before control is returned to the prompt; in a Python program this happens just before the program exits. The handling of such top-level exceptions can be customized by assigning another three-argument function to sys.excepthook.
# import sys
# sys.excepthook = my_exchandler

import sys

sys_excepthook = sys.excepthook
class Error(Exception):
    '''Custom Error class that additionally accepts a code to further track error'''
    def __init__(self, message: str, code=1):
        self.message = message
        super(Error, self).__init__(message)
        self.code = code

    def __str__(self):
        lines = self.message.splitlines()
        message = '    ' + '\n    '.join(lines)
        return f'{self.__class__.__name__}:\n{message}\n\nExit Code: {self.code}'


def exception_hook(ty, val, tb):
    if ty == Error:
        print(val, file=sys.stderr)
        sys.exit(val.code)
    else:
        sys_excepthook(ty, val, tb)


