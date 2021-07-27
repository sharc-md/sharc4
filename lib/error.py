# XXX: decide for further features
# something like this is also possible: (personalized exception handler)
# def my_exchandler(type, value, traceback):
#     print(value)

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
