# XXX: decide for further features
# something like this is also possible: (personalized exception handler)
# def my_exchandler(type, value, traceback):
#     print(value)
# sys.excepthook(type, value, traceback)
# This function prints out a given traceback and exception to sys.stderr.

# When an exception is raised and uncaught, the interpreter calls sys.excepthook with three arguments, the exception class, exception instance, and a traceback object. In an interactive session this happens just before control is returned to the prompt; in a Python program this happens just before the program exits. The handling of such top-level exceptions can be customized by assigning another three-argument function to sys.excepthook.
# import sys
# sys.excepthook = my_exchandler
class Error(Exception):
    '''Custom Error class that additionally accepts a code to further track error'''
    def __init__(self, message, code=1):
        self.message = message
        super(Error, self).__init__(message)
        self.code = code

    def __str__(self):
        return f'{self.message} [{self.code}]'
