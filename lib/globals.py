#!/usr/bin/env python3

# give the variables debug and print as singletons

class Debug(object):
    
    val = True

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Debug, cls).__new__(cls)
        return cls.instance

    def __bool__(self):
        return self.val
    
    def set(self, val):
        self.val = val


class Print(object):
    
    val = True

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Print, cls).__new__(cls)
        return cls.instance

    def __bool__(self):
        return self.val

    def set(self, val):
        self.val = val


DEBUG = Debug()

PRINT = Print()
