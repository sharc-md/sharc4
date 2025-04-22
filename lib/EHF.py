#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************


import os
import sys
import time
import itertools
import math
import numpy as np

from logger import logging, CustomFormatter
from SHARC_HYBRID import SHARC_HYBRID
from qmout import QMout

#----START of EHF class--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
class EHF:
    def __init__( self, **kwargs ):
        for key, value in kwargs.items():
            setattr(self,key,value)

    def run(self):
        t1 = time.time()
        indent = ' '*4
        echarges = self.echarges  
        estates = self.estates  
        egarden = self.egarden   
        maxcycles =  self.maxcycles 
        forced = self.forced
        tQ =  self.tQ

        self.log.print(indent+'GUESS DATA')
        self.log.print('')
        for label, child in egarden.items(): 
            symbols = ''.join([ f"{    child.QMin.molecule['elements'][a]:10}" for a in range(child.QMin.molecule['natom']) ])
            charges = ''.join([ f"{ echarges[label][a]:10.5f}" for a in range(child.QMin.molecule['natom']) ])
            self.log.print(indent+'   FRAGMENT '+label+':')
            self.log.print(indent+'      Atoms:              '+symbols)
            self.log.print(indent+'      RESP charges: '+charges)
            self.log.print('')
        self.log.print('')

        cycle = 0
        convergence = { label:np.zeros(egarden[label].QMin.molecule['natom'],dtype=bool) for label in egarden.keys() }
        while True:
            cycle += 1
            self.log.print(indent+'CYCLE '+str(cycle))
            self.log.print('')

            # Check if some of the forced fragments has exceeded their max_cycles
            exceeded = [ label for label in egarden.keys() if cycle > maxcycles[label] ]
            for e in exceeded:
                if forced[e] and not np.all(convergence[e]):
                    self.log.print(indent+' Fragment '+e+' is forced to converge, but has exceeded its max. number of EHF cycles.')
                    self.log.print(indent+' Aborting the whole ECI calculation...')
                    exit()

            # Determine which fragments still need to be runned
            running_garden = { label:child for label,child in egarden.items() if cycle <= maxcycles[label] }
            self.log.print(indent+'   Running '+str(len(running_garden))+' fragments in this cycle...')
            if len(running_garden) < 1: 
                self.log.print(indent+'   ...,that is, ending EHF.')
                break

            # Write echarges as pccharges to each child
            for label1, child1 in running_garden.items():
                PCs = np.concatenate( [ echarges[label2] for label2 in egarden.keys() if label2 != label1 ] )
                child1.QMin.coords['pccharge'][0:PCs.shape[0]] = PCs
                child1.QMout = QMout(states=child1.QMin.molecule["states"], natom=child1.QMin.molecule["natom"], npc=child1.QMin.molecule["npc"], charges=child1.QMin.molecule["charge"])

            # Run running children
            SHARC_HYBRID.run_queue(self.log, running_garden, self.nproc, indent=" "*7)
            self.log.print('')

            # Check convergence and print
            dPCs = {}
            for label, child in running_garden.items(): 
                newPCs = child.QMout['multipolar_fit'][(estates[label],estates[label])][:,0]
                dPCs[label] = newPCs - echarges[label]
                echarges[label] = newPCs.copy()
                convergence[label] = np.abs(dPCs[label]) < tQ[label]

                # Printing part
                symbols = ''.join([ f"{    child.QMin.molecule['elements'][a]:10}" for a in range(child.QMin.molecule['natom']) ])
                charges = ''.join([ f"{ echarges[label][a]:10.5f}" for a in range(child.QMin.molecule['natom']) ])
                dcharges = ''.join([ f"{ dPCs[label][a]:10.5f}" for a in range(child.QMin.molecule['natom']) ])
                conv = ''.join([ '   YES    ' if convergence[label][a] else '    NO    ' for a in range(child.QMin.molecule['natom']) ]) 
                l = len(label)
                self.log.print(indent+'   FRAGMENT '+label+':')
                self.log.print(indent+'      Atoms:              '+symbols)
                self.log.print(indent+'      RESP charges: '+charges)
                self.log.print(indent+'      Delta:        '+dcharges)
                self.log.print(indent+'      Converged:      '+conv)
                self.log.print('')

            if all( [ np.all(convergence[label]) for label in running_garden ] ):
                self.log.print(indent+'EHF convergence reached in '+str(cycle)+' cycles!')
                break

            #  for label, child in running_garden.items(): 
                #  for step in ['init', 'always_orb_init', 'newstep', 'restart' ]:
                    #  child1.QMin.save[step] = False
                #  child.QMin.save['samestep'] = True
                #  child.QMin.control['densonly'] = True
        t2 = time.time()
        self.log.print(indent+'Time elapsed in EHF.run = '+str(round(t2-t1,3))+' sec.')
        return



