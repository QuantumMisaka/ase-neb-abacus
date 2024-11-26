# JamesMisaka in 2023-11-27
# Dimer calculation workflow by using ABACUS calculator
# part of ATST-Tools scripts

import os
import sys

import numpy as np
from ase.io import Trajectory, read, write
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
#from my_dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.calculators.abacus import Abacus, AbacusProfile

class AbacusDimer:
    """Customize Dimer calculation workflow by using ABACUS"""
    
    def __init__(self, init_Atoms, parameters, abacus='abacus',
                 mpi=1, omp=1, directory='ABACUS', 
                 traj_file='run_dimer.traj',
                 init_eigenmode_method='displacement',
                 displacement_vector: np.ndarray = None,):
        """Initialize Dimer method by using ASE-ABACUS

        init_Atoms (Atoms object): starting image, can be from every way including NEB result
        parameters (dict): settings of abacus input parameters
        abacus (str): Abacus executable file. Default: 'abacus'
        directory (str): calculator directory name, for parallel calculation {directory}-rank{i} will be the directory name
        mpi (int): number of MPI for abacus calculator
        omp (int): number of OpenMP for abacus calculator
        traj_file (str): trajectory file name for dimer calculation, when running dimer calculation, trajetory will be written to this file, default is 'dimer.traj'
        init_eigenmode_method (str): dimer initial eigenmode method. Choose from 'displacement' and 'gauss'.
        displacement_vector (np.ndarray): displacement vector for dimer initial eigenmode. Only used when init_eigenmode_method is 'displacement'
        """
        self.init_Atoms = init_Atoms
        self.parameters = parameters
        self.abacus = abacus
        self.mpi = mpi
        self.omp = omp
        self.directory = directory
        self.traj_file = traj_file
        self.init_eigenmode_method = init_eigenmode_method
        self.displacement_vector = displacement_vector
        
    def set_calculator(self):
        """Set Abacus calculators"""
        os.environ['OMP_NUM_THREADS'] = f'{self.omp}'
        profile = AbacusProfile(command=f"mpirun -np {self.mpi} {self.abacus}")
        out_directory = self.directory
        calc = Abacus(profile=profile, directory=out_directory,
                        **self.parameters)
        return calc
    
    def set_d_mask_by_displacement(self):
        """set mask by displacement"""
        print("=== Set mask by displacement vector where displacement is [0,0,0] ===")
        d_mask = self.displacement_vector != np.zeros(3)
        d_mask = d_mask[:,0].tolist()
        return d_mask
    
    def set_d_mask_by_constraint(self):
        """set mask by constraint of Atoms"""
        print("=== Set mask by constraint read from init Atoms ===")
        dimer_init = self.init_Atoms
        d_mask = [True] * len(dimer_init)
        const = dimer_init._get_constraints()
        # const will be empty list if no constraint
        if const:
            const_object = dimer_init._get_constraints()[0].get_indices()
            for ind in const_object:
                d_mask[ind] = False
            return d_mask
        else:
            print("--- Notice: No constraint found in init Atoms, there will be no mask in dimer calculation ---")
            return d_mask
    
    def set_d_mask_by_specified(self, moving_atoms_ind: list):
        """set mask be choosing moving atoms, the others are masked"""
        print(f"=== Set mask by specifing moving atoms {moving_atoms_ind} ===")
        dimer_init = self.init_Atoms
        d_mask = [False] * len(dimer_init)
        for ind in moving_atoms_ind:
            d_mask[ind] = True
        return d_mask
        
    def run(self, fmax=0.05, properties=["energy", "forces", "stress"], moving_atoms_ind: list = None):
        """run dimer calculation workflow
        
        Args:
            fmax (float): threshold (unit: eV/Angstrom) of the force convergence
            properties (list): properties dumped in trajectory files, default ['energy', 'forces', 'stress']
        """
        dimer_init = self.init_Atoms
        dimer_init.calc = self.set_calculator()
        dimer_traj = Trajectory(self.traj_file, 'w', dimer_init, properties=properties)
        if self.init_eigenmode_method == "displacement":
            if moving_atoms_ind:
                d_mask = self.set_d_mask_by_specified(moving_atoms_ind)
            else:
                d_mask = self.set_d_mask_by_constraint()
                # d_mask = self.set_d_mask_by_displacement()
            d_control = DimerControl(initial_eigenmode_method=self.init_eigenmode_method, 
                                    displacement_method="vector", 
                                    mask=d_mask)
            d_atoms = MinModeAtoms(dimer_init, d_control)
            d_atoms.displace(displacement_vector=self.displacement_vector)
        elif self.init_eigenmode_method == "gauss":
            # leave a way for random displacement
            d_mask = self.set_d_mask_by_constraint()
            d_control = DimerControl(initial_eigenmode_method=self.init_eigenmode_method, 
                                    mask=d_mask)
            d_atoms = MinModeAtoms(dimer_init, d_control)
        else:
            raise ValueError("init_eigenmode_method must be displacement or gauss")
        dimer_relax = MinModeTranslate(d_atoms, trajectory=dimer_traj)
        dimer_relax.run(fmax=fmax)