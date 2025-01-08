#!/usr/bin/env python3
import os
import sys
import time
import itertools
import math
import numpy as np

from logger import logging, CustomFormatter
from SHARC_HYBRID import SHARC_HYBRID

#----START of EHF class--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
class EHF:
    def __init__(self, nproc, APCs, estates, frozen, relaxed, maxcycle, tQ, output):
        self.nproc = nproc
        self.APCs = APCs
        self.estates = estates
        self.frozen = frozen
        self.relaxed = relaxed
        self.maxcycle = maxcycle
        self.tQ = tQ
        # Logger
        self.log = logging.getLogger(output+'.log')
        self.log.propagate = False
        self.log.handlers = []
        self.log.setLevel('DEBUG')  # TODO: inherit the global loglevel
        hdlr = (
            logging.FileHandler(filename=output+'.log', mode="w", encoding="utf-8")
        )
        hdlr._name = output+"Handler"
        hdlr.setFormatter(CustomFormatter())
        self.log.addHandler(hdlr)
        self.log.print = self.log.info
        return

    def run(self, external_coords, external_charges):
        frozen = self.frozen
        relaxed = self.relaxed
        estates = self.estates
        # Run frozen fragments
        #  SHARC_HYBRID.run_children(self.log,frozen)
        #  for label, child in frozen.items():
            #  self.APCs[label] = child.QMout['multipolar_fit'][(estates[label],estates[label])][:,0]

        # Manage relaxed fragments
        if len(relaxed) > 0:
            #  save = {}
            for label1, child1 in relaxed.items(): 
                pccoords = [ child2.QMin.coords['coords'] for label2, child2 in frozen.items() ] 
                pccoords = pccoords + [ child2.QMin.coords['coords'] for label2, child2 in relaxed.items() if label1 != label2 ]
                if external_coords: pccoords = pccoords + [ external_coords ] # They go last
                pccoords = np.concatenate( pccoords, axis=0 )
                child1.set_coords( pccoords, pc=True )

                #  save[label1] = child1.QMin.save.copy()

            convergence = { label: [False] for label in relaxed }
            dAPCs = {}
            for cycle in range(self.maxcycle):
                self.log.print(' CYCLE '+str(cycle))
                for label1, child1 in relaxed.items(): 
                    PCs = [ self.APCs[label2] for label2 in frozen ]
                    PCs = PCs + [ self.APCs[label2] for label2 in relaxed if label2 != label1 ]
                    if external_charges: PCs = PCs + external_charges
                    PCs = np.concatenate( PCs )
                    child1.QMin.coords['pccharge'] = PCs

                SHARC_HYBRID.run_children(self.log, relaxed, self.nproc)
                for label, child in relaxed.items(): 
                    child.writeQMout( filename=os.path.join( child.QMin.resources['pwd'],'QM_cycle'+str(cycle)+'.out' ) )
                    newAPCs = child.QMout['multipolar_fit'][(estates[label],estates[label])][:,0]
                    dAPCs[label] = newAPCs - self.APCs[label]
                    self.APCs[label] = newAPCs.copy()
                    convergence[label] = np.abs(dAPCs[label]) < self.tQ
                    self.log.print('   Fragment '+label)
                    for a in range(child.QMin.molecule['natom']):
                        yesno = 'NO'
                        if convergence[label][a]: yesno = 'YES'
                        self.log.print('   '+'      '.join( [ str(a+1), f"{self.APCs[label][a]: 8.5f}", f"{dAPCs[label][a]: 8.5f}", yesno ] ))
                if all( [ all(convergence[label]) for label in relaxed ] ):
                    self.log.print(' EHF convergence reached in '+str(cycle+1)+' cycles!')
                    break
                for label, child in relaxed.items(): 
                    for step in ['init', 'always_orb_init', 'newstep', 'restart' ]:
                        child1.QMin.save[step] = False
                    child.QMin.save['samestep'] = True
                    child.QMin.control['densonly'] = True
                
            if not all( [ all(convergence[label]) for label in relaxed ] ):
                self.log.warning(' Maximum number in EHF is exceeded but some charges are still not converged! Proceeding nevertheless...')
        return



