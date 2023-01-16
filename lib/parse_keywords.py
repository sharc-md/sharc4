#!/usr/bin/env python3

import os
import ast

from error import Error
from utils import readfile


class KeywordParser:
    '''This class contains helper functions to parse different keyswords files
All functions are named by their corresponding key and are called by the args.
key and args are defined as: line.split(None, 1)
key equals everything before the first whitespace in the line and args everything after (fallback: ' ')

Author: Severin Polonius
Date: 20.07.2021
'''

    def __init__(self, nmult: int, atomcharge):
        self.nmult = nmult
        self.atomcharge = atomcharge

    def paddingstates(self, args: str) -> list[int]:
        alist = args.split()
        res = []
        if len(alist) == 1:
            res = [int(alist[0])] * self.nmult
        elif len(alist) >= self.nmult:
            res = list(map(int, alist))
        elif len(alist) == 0:
            res = [0] * self.nmult
        else:
            raise Error('Length of "paddingstates" does not match length of "states"!', 58)
        # logic check
        for i in res:
            if i < 0:
                raise Error('paddingstate specification cannot be negative!', 59)
        return res

    def charge(self, args: str) -> list[int]:
        alist = args.split()
        if len(alist) < self.nmult:
            raise Error('specify charge for each multiplicity!')
        res = []
        try:
            res = list(map(lambda x: int(float(x)), alist))
        except (ValueError, IndexError):
            raise Error('Keyword "charge" only accepts integers (i.e charge -1)')
        if len(res) == 1:
            charge = int(float(alist[0]))
            if (self.atomcharge + charge) % 2 == 1 and self.nmult > 1:
                print('HINT: Charge shifted by -1 to be compatible with multiplicities.')
                charge -= 1
            res = [i % 2 + charge for i in range(self.nmult)]
            print(f'HINT: total charge per multiplicity automatically assigned, please check (charge {res}).')
            print('You can set the charge in the template manually for each multiplicity ("charge 0 +1 0 ...")\n')
            return res
        elif len(res) >= self.nmult:
            compatible = True
            for imult, cha in enumerate(res):
                if not (self.atomcharge + cha + imult) % 2 == 0:
                    compatible = False
                    break
            if not compatible:
                print(
                    'WARNING: Charges from template not compatible with multiplicities!  \
                    (this is probably OK if you use QM/MM)'
                )
            return res
        else:
            raise Error('Length of "charge" does not match length of "states"!', 61)

    @staticmethod
    def basis_per_element(args: str) -> dict:
        key, value = args.split(None, 1)
        return {key: value}

    ecp_per_element = basis_per_element

    @staticmethod
    def basis_per_atom(args: str) -> dict:
        key, value = args.split(None, 1)
        return {int(key) - 1: value}

    @staticmethod
    def range_sep_settings(args: str) -> dict:
        alist = args.split()
        return {'do': True, 'mu': alist[0], 'scal': alist[1], 'ACM1': alist[2], 'ACM2': alist[3], 'ACM3': alist[4]}

    @staticmethod
    def paste_input_file(args: str) -> list[str]:
        path = os.path.expandvars(os.path.expanduser(args))
        if os.path.isfile(path):
            return readfile(path)
        else:
            raise Error(f'Additional input file {path} not found!', 62)

    @staticmethod
    def path(args: str) -> str:
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(args)))
        if '$' in path:
            raise Error(f'Path: {path} contains undefined env variables!\nPath generated from: {args}', 67)
        return path

    @staticmethod
    def neglected_gradient(args: str) -> str:
        args = args.lower()
        if args in {'zero', 'gs', 'closest'}:
            return args
        else:
            raise Error('Unknown argument to "neglected_gradient"!', 57)

    @staticmethod
    def theodore_prop(args: str) -> list[str]:
        theodore_spelling = {
            'Om', 'PRNTO', 'Z_HE', 'S_HE', 'RMSeh', 'POSi', 'POSf', 'POS', 'PRi', 'PRf', 'PR', 'PRh', 'CT', 'CT2',
            'CTnt', 'MC', 'LC', 'MLCT', 'LMCT', 'LLCT', 'DEL', 'COH', 'COHh'
        }
        t_lower = {k.lower(): k for k in theodore_spelling}
        res = []
        if args[0] == '[':
            res = ast.literal_eval(args)
        else:
            res = args.split()
        return [t_lower[x.lower()] for x in res]

    @staticmethod
    def theodore_fragment(args: str) -> list[int]:
        res = []
        if args[0] == '[':
            res = ast.literal_eval(args)
            if isinstance(res[0], str):
                res = list(map(lambda x: [int(i) for i in x.split()], res))
        else:
            res = [[int(x) for x in args.split()]]
        return res

    @staticmethod
    def qmmm_table(args: str) -> list[list]:
        path = os.path.abspath(os.path.expanduser(os.path.expanduser(args)))
        if os.path.isfile(path):
            res = []
            with open(path, 'r') as f:
                test = f.readline().split()
                if len(test) != 2:
                    print("Warning: You might use an old QMMM.table file!\nnew format <qm/mm> <symbol> <bond1> <bond2>...")
                res.append([test[0], test[1], *map(lambda x: int(x) - 1, test[2:])])
                res.extend([[*x[0:2]] + [int(y) - 1 for y in x[2:]] for x in map(lambda x: x.split(), f)])
            return res
        else:
            raise Error(f'File {path} does not exist!', 1)

    @staticmethod
    def resp_shells(args: str) -> list[int]:
        return ast.literal_eval(args)

    @staticmethod
    def resp_vdw_radii_symbol(args: str) -> dict[str, float]:
        res = {}
        if args[0] == '[':
            lst = [x.split() for x in ast.literal_eval(args)]
            res = {x[0]: float(x[1]) for x in lst}
        else:
            lst = args.split()
            res = {lst[i]: float(lst[i + 1]) for i in range(0, len(lst), 2)}
        return res

    @staticmethod
    def resp_vdw_radii(args: str) -> list[float]:
        if args[0] == '[':
            res = ast.literal_eval(args)
        else:
            res = args.split()
        return [float(x) for x in res]

