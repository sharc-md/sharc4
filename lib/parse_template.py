#!/usr/bin/env python3

from error import Error
from vib_molden import atom


class TemplateParser:
    '''This class contains helper functions to parse different keys in the .template files
All functions are named by their corresponding key and are called by the args.
key and args are defined as: line.split(None, 1)
key equals everything before the first whitespace in the line and args everything after (fallback: ' ')

Author: Severin Polonius
Date: 20.07.2021
'''

    def __init__(self, nstates: int, atomcharge):
        self.nstates = nstates
        self.atomcharge = atomcharge

    def paddingstates(self, args: str) -> list[int]:
        alist = args.split()
        res = []
        if len(alist) == 1:
            res = [int(alist[0])] * self.nstates
        elif len(alist) >= self.nstates:
            res = list(map(int, alist))
        elif len(alist) == 0:
            res = [0] * self.nstates
        else:
            raise Error('Length of "paddingstates" does not match length of "states"!', 58)
        # logic check
        for i in res:
            if i < 0:
                raise Error('paddingstate specification cannot be negative!', 59)
        return res

    def charge(self, args: str) -> list[int]:
        alist = args.split()
        res = []
        try:
            res = list(map(lambda x: int(float(x)), alist))
        except (ValueError, IndexError):
            raise Error('Keyword "charge" only accepts integers (i.e charge -1)')
        if len(res) == 1:
            charge = int(float(alist[1]))
            if (self.atomcharge + charge) % 2 == 1 and self.nstates > 1:
                print('HINT: Charge shifted by -1 to be compatible with multiplicities.')
                charge -= 1
            res = [i % 2 + charge for i in range(self.nstates)]
            print(f'HINT: total charge per multiplicity automatically assigned, please check (charge {res}).')
            print('You can set the charge in the template manually for each multiplicity ("charge 0 +1 0 ...")\n')
            return res
        elif len(res) >= self.nstates:
            compatible = True
            for imult, cha in enumerate(res):
                if not (self.atomcharge + cha + imult) % 2 == 0:
                    compatible = False
                    break
            if not compatible:
                print('WARNING: Charges from template not compatible with multiplicities!  (this is probably OK if you use QM/MM)')
            return res
        else:
            raise Error('Length of "charge" does not match length of "states"!', 61)

    def basis_per_element(args: str) -> dict:
        key, value = args.split(None, 1)
        return {key: value}

    ecp_per_element = basis_per_element

    def basis_per_atom(args: str) -> dict:
        key, value = args.split(None, 1)
        return {int(key) - 1: value}

    def range_sep_settings(args: str) -> dict:
        alist = args.split()
        return {'do': True, 'mu': alist[0], 'scal': alist[1], 'ACM1': alist[2], 'ACM2': alist[3], 'ACM3': alist[4]}
