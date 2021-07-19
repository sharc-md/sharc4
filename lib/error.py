import os

global DEBUG
DEBUG = False
if 'DEBUG' in os.environ and os.environ["DEBUG"].lower() in ["true", "false"]:
    DEBUG = os.environ["DEBUG"] == "true"

# XXX: decide for further features
class Error(Exception):
    '''Custom Error class that additionally accepts a code to further track error'''
    def __init__(self, message, code=1):
        self.message = message
        super(Error, self).__init__(message)
        self.code = code