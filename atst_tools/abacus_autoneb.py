# JamesMisaka in 2023-11-27
# AutoNEB calculation workflow by using ABACUS calculator
# part of ASE-NEB-ABACUS scripts

import os
from ase.atoms import Atoms
from ase.io import read, write
from ase.calculators.abacus import Abacus, AbacusProfile
#from ase.mep.autoneb import AutoNEB # official
from atst_tools.my_autoneb import AutoNEB
from ase.parallel import world, parprint, paropen
from ase.optimize import FIRE, BFGS
from ase.constraints import FixAtoms


class AbacusAutoNEB:
    """Customize AutoNEB calculation workflow by using ABACUS"""

    def __init__(self, init_chain, parameters, 
                 abacus='abacus',  prefix="run_autoneb", 
                 n_simul=1, n_max=10, k=0.10,
                 algorism="improvedtangent", 
                 directory='AutoNEBrun', mpi=1, omp=1, parallel=True, ):
        """Initialize initial and final states

        init_chain (List[Atoms]): starting image chain from nem_make.py or other method, can only include Initial and Final states, autoneb will generate inter-images automactically. 
        parameters (dict): settings of abacus input parameters
        abacus (str): Abacus executable file. Default: 'abacus'
        prefix (str): prefix for AutoNEB output files, default 'run_autoneb'
        n_simul (int): number of simultaneous calculation, default use world.size, read from ase.parallel
        n_max (int): max number of all image in NEB band, default 10. The max number will not be reached if the convergence is reached before that.
        k (float): spring constant for NEB calculation, eV/Ang^2, default: 0.10
        
        algorism (str): NEB algorism. which can be
        - 'aseneb': standard ase NEB
        - 'improvedtangent' : IT-NEB (recommended by Sobereva)
        - 'eb': climbing image elastic band method (default in AutoNEB)
        
        Default: 'improvedtangent'
        
        directory (str): calculator directory name, for parallel calculation {directory}-rank{i} will be the directory name
        mpi (int): number of MPI for abacus calculator
        omp (int): number of OpenMP for abacus calculator
        parallel (bool): parallel calculation setting, default True
        """

        self.init_chain = init_chain
        self.algorism = algorism
        self.abacus = abacus
        self.directory = directory
        self.parameters = parameters
        self.prefix = prefix
        self.mpi = mpi
        self.omp = omp
        self.k = k
        self.parallel = parallel
        parprint("Notice: AutoNEB method is set")
        if (n_simul > 0) and (n_max >= n_simul):
            parprint(f"You manually set n_simul = {n_simul}, n_max = {n_max}", )
            self.n_simul = n_simul
            self.n_max = n_max
        else:
            raise ValueError("You must set n_simul > 0 and n_max >= n_simul numbers for AutoNEB")
        
            
    def set_calculator(self) -> Abacus:
        """Set Abacus calculators"""
        os.environ['OMP_NUM_THREADS'] = f'{self.omp}'
        profile = AbacusProfile(command=f"mpirun -np {self.mpi} {self.abacus}")
        if self.parallel:
            out_directory = f"{self.directory}-rank{world.rank}"
        else:
            out_directory = self.directory
        calc = Abacus(profile=profile, directory=out_directory,
                    **self.parameters)
        return calc
    
    
    def attach_calculators(self, images: list):
        """Attach calculator to a list of images supplied"""
        for num, image in enumerate(images):
            image.calc = self.set_calculator()


    def set_init_and_final_conditions(self) -> list:
        """Attach fix and init magmom information for init and final image"""
        image_use = self.init_chain[len(self.init_chain) // 2]
        init_magmom = image_use.get_initial_magnetic_moments()
        init:Atoms
        final:Atoms
        # parprint("==> set mag <==")
        cond_chain = []
        init = self.init_chain[0]
        final = self.init_chain[-1]
        init.set_initial_magnetic_moments(init_magmom)
        final.set_initial_magnetic_moments(init_magmom)
        # a round-around method should be used here
        # for add init magmom and preserve energy property at the same time
        write(f"{self.prefix}_init.extxyz", self.init_chain, format="extxyz")
        init = read(f"{self.prefix}_init.extxyz", index=0)
        final = read(f"{self.prefix}_init.extxyz", index=-1)
        # add fix later
        FixAtom_list = image_use._get_constraints()
        # parprint("==> set fix <==")
        if FixAtom_list:
            fix_indices = FixAtom_list[0].get_indices()
            fix_obj = FixAtoms(indices=fix_indices)
            init.set_constraint(fix_obj)
            final.set_constraint(fix_obj)
        cond_chain.append(init)
        cond_chain += self.init_chain[1:-1]
        cond_chain.append(final)
        # for condition check
        write("STRU-init", cond_chain[0], format="abacus")
        write("STRU-final", cond_chain[-1], format="abacus")
        return cond_chain
        
    
    def run(self, optimizer=FIRE, fmax=0.05, climb=True, smooth_curve=False):
        """Run Abacus AutoNEB

        optimizer (Optimizer object): default FIRE. BFGS and FIRE is only used
        fmax (float): threshold (unit: eV/Angstrom) of the force convergence
        climb (bool): using climbing image NEB method in last AutoNEB calculation, default True
        smooth_curve (bool): smooth the NEB curve after NEB calculation, default False
        """
        parprint("----- Running AutoNEB -----")
        parprint(f"----- {self.algorism} method is being used -----")
        # add constaints and init magmom for init and final image
        running_chain = self.set_init_and_final_conditions()
        # running
        for num, image in enumerate(running_chain):
            index = f"{num:03d}"
            write(f"{self.prefix}{index}.traj", image, format="traj")
        world.barrier()
        autoneb = AutoNEB(self.attach_calculators, self.prefix, n_simul = self.n_simul, 
                          n_max = self.n_max, fmax = fmax, climb = climb, k = self.k,
                            method = self.algorism, parallel = self.parallel, 
                            optimizer = optimizer, smooth_curve = smooth_curve)
        autoneb.run()
        parprint("----- AutoNEB calculation finished -----")