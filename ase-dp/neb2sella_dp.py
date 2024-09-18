# Author: JamesMisaka 
# Using deepmd to search TS via NEB-SELLA
# Last Update: 2024-07-23

import numpy as np
import os, sys, shutil
from copy import deepcopy

from ase.io import read, write, Trajectory
from ase import Atoms
from ase.optimize import BFGS, FIRE, QuasiNewton
from ase.constraints import FixAtoms
from ase.visualize import view
from ase.mep.neb import NEBTools, NEB, DyNEB
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
from ase.calculators.singlepoint import SinglePointCalculator

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.neb.pathfinder import IDPPSolver

from sella import Sella, Constraints

from deepmd.calculator import DP

model = "FeCHO-dpa220-100-30656.pt"
n_neb_images = 12
neb_fmax = 1.00  # neb should be rough, but need steps
sella_fmax = 0.05 # sella use neb guess
climb = True
scale_fmax = 0.0 # 0 or 1, 1 for using dyneb
omp = 16
neb_algorism = "improvedtangent" # eb is another choice
neb_log = "neb_images.traj"
sella_log = "sella_images.traj"

OPTSolver = QuasiNewton
NEBSolver = FIRE

# developers only
sella_eta = 0.0005
neb_sort_tol = 1
os.environ['OMP_NUM_THREADS'] = "omp"

# reading part
msg = '''
Usage: 
- For using IS and FS: 
    python neb2sella_dpa2.py [init_stru] [final_stru] ([format])
- For using existed NEB: 
    python neb2sella_dpa2.py [neb_latest.traj]
'''

# parse arguments, argparse lib is better 
if len(sys.argv) < 2:
    print(msg)
    sys.exit(1)
elif len(sys.argv) == 2:
    if sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print(msg)
        sys.exit(0)
    else:
        neb_traj = sys.argv[1]
        neb_abacus = read(neb_traj, ":", format="traj")
        atom_init = neb_abacus[0]
        atom_final = neb_abacus[-1]
        assert type(atom_init) == Atoms and type(atom_final) == Atoms, \
        "The input file is not a trajectory file contained Atoms object"
else:
    init_stru = sys.argv[1]
    final_stru = sys.argv[2]
    if len(sys.argv) == 4:
        format = sys.argv[3]
    else:
        format = None # auto detect
    atom_init = read(init_stru, format=format)
    atom_final = read(final_stru, format=format)


# init and final stru
atom_init.calc = DP(model=model)
init_relax = OPTSolver(atom_init)
init_relax.run(fmax=0.05)
# print(atom_init.get_potential_energy())
atom_final.calc = DP(model=model)
final_relax = OPTSolver(atom_final)
final_relax.run(fmax=0.05)

init_opted = "init_opted.traj"
final_opted = "final_opted.traj"
write(init_opted, atom_init, format="traj")
write(final_opted, atom_final, format="traj")

# run neb
# use pymatgen interpolate, which is better and avoid trans-cell problem
print(f'Generating NEB path, number of images: {n_neb_images}, sort_tol: {neb_sort_tol}')
is_pmg = AseAtomsAdaptor.get_structure(atom_init)
fs_pmg = AseAtomsAdaptor.get_structure(atom_final)
path = IDPPSolver.from_endpoints([is_pmg, fs_pmg], n_neb_images, sort_tol=neb_sort_tol)
new_path = path.run(maxiter=5000, tol=1e-5, gtol=1e-3)
# conver path to ase format, and add SinglePointCalculator
ase_path = [i.to_ase_atoms() for i in new_path]
ase_path[0] = atom_init
ase_path[-1] = atom_final
for img in ase_path[1:-1]:
    img.calc = DP(model=model)

neb = DyNEB(ase_path, 
            climb=climb, dynamic_relaxation=True, fmax=neb_fmax,
            method=neb_algorism, parallel=False, scale_fmax=scale_fmax,
            allow_shared_calculator=True)

traj = Trajectory(neb_log, 'w', neb)
opt = NEBSolver(neb, trajectory=traj)
opt.run(neb_fmax)

# neb displacement to dimer
n_images = NEBTools(ase_path)._guess_nimages()
neb_raw_barrier = max([image.get_potential_energy() for image in ase_path])
fmax = NEBTools(ase_path).get_fmax()
barrier = NEBTools(ase_path).get_barrier()[0]
TS_info = [(ind, image) 
            for ind, image in enumerate(ase_path) 
            if image.get_potential_energy() == neb_raw_barrier][0]
print(f"=== Locate TS in {TS_info[0]} of 0-{n_images-1} images  ===")
print(f"=== NEB Raw Barrier: {neb_raw_barrier:.4f} (eV) ===")
print(f"=== NEB Fmax: {fmax:.4f} (eV/A) ===")
print(f"=== Now Turn to Sella with NEB Information ===")

# displacement vector for vib analysis
step_before_TS = 1
step_after_TS = 1
norm_vector = 0.01
#out_vec = 'displacement_vector.npy',

ind_before_TS = TS_info[0] - step_before_TS
ind_after_TS = TS_info[0] + step_after_TS
img_before = ase_path[ind_before_TS]
img_after = ase_path[ind_after_TS]
image_vector = (img_before.positions - img_after.positions)
modulo_norm = np.linalg.norm(image_vector) / norm_vector
displacement_vector = image_vector / modulo_norm
print(f"=== Displacement vector generated by {ind_before_TS} and {ind_after_TS} images of NEB chain ===")
print(f"=== Which is normalized to {norm_vector} length ! ===")

def main4dis(displacement_vector, thr=0.10):
    """Get Main Parts of Displacement Vector by using threshold"""
    len_vector = np.linalg.norm(displacement_vector)
    norm_vector = np.linalg.norm(displacement_vector / len_vector, axis=1)
    main_indices = [ind for ind,vec in enumerate(norm_vector) if vec > thr]
    return main_indices, norm_vector

def thermo_analysis(atoms, T, name="vib", indices=None, delta=0.01, nfree=2):
    """Do Thermo Analysis by using ASE"""
    vib_mode_dir = f"{name}_mode"
    if os.path.exists(name):
        shutil.move(name, f"{name}_bak")
    if not os.path.exists(vib_mode_dir):
        os.mkdir(vib_mode_dir)   
    vib = Vibrations(atoms, indices=indices, name=name, delta=delta, nfree=nfree)
    vib.run()
    vib.summary()
    ROOT_DIR = os.getcwd()
    os.chdir(vib_mode_dir)
    vib.write_mode()
    os.chdir(ROOT_DIR)
    vib_energies = vib.get_energies()
    thermo = HarmonicThermo(vib_energies, ignore_imag_modes=True,)
    entropy = thermo.get_entropy(T)
    free_energy = thermo.get_helmholtz_energy(T)
    print(f"==> Entropy: {entropy:.6e} eV/K <==")
    print(f"==> Free Energy: {free_energy:.6f} eV <==")
    print()

# sella part
ts_guess = TS_info[1].copy()
ts_guess.calc = DP(model=model)

# set sella cons is optional, use cons from ASE is enough
# ts_guess.set_constraint()
# d_mask = (displacement_vector != np.zeros(3))
# cons_index = np.where(d_mask == False)[0]
# cons = Constraints(ts_guess)
# # use dimer not-moved atoms as constraints
# cons.fix_translation(cons_index)

dyn = Sella(
    ts_guess,
    trajectory = sella_log,
    eta = sella_eta,
)
dyn.run(fmax=sella_fmax)


# get struc of IS,FS,TS
write("IS_get.cif", atom_init, format="cif")
write("FS_get.cif", atom_final, format="cif")
write("TS_get.cif", ts_guess, format="cif")
write("IS_get.stru", atom_init, format="abacus")
write("FS_get.stru", atom_final, format="abacus")
write("TS_get.stru", ts_guess, format="abacus")

# get energy informations
ene_init = atom_init.get_potential_energy()
ene_final = atom_final.get_potential_energy()
ene_ts = ts_guess.get_potential_energy()
ene_delta = ene_final - ene_init
ene_activa = ene_ts - ene_init
ene_act_rev = ene_ts - ene_final
msg = f'''
==> TS-Search Results <==
- Items      Energy
- IS         {ene_init:.6f}
- FS         {ene_final:.6f}
- TS         {ene_ts:.6f}
- dE         {ene_delta:.6f}
- Ea_f       {ene_activa:.6f}
- Ea_r       {ene_act_rev:.6f}
'''
print(msg)

# use neb2dimer information to do vibration analysis
print("==> Do Vibrational Analysis by DP Potential <==")
vib_indices, norm_vector = main4dis(image_vector, thr=0.10)
print(f"=== TS main moving atoms: {vib_indices} ===")
T = 523.15 # K
delta = 0.01
nfree = 2

vib_is_name = 'vib_is'
vib_fs_name = 'vib_fs'
vib_ts_name = 'vib_ts'

print("==> For TS Structure <==")
thermo_analysis(ts_guess, T, name=vib_ts_name, indices=vib_indices, delta=delta, nfree=nfree)
print("==> For Initial Structure <==")
thermo_analysis(atom_init, T, name=vib_is_name, indices=vib_indices, delta=delta, nfree=nfree)
print("==> For Final Structure <==")
thermo_analysis(atom_final, T, name=vib_fs_name, indices=vib_indices, delta=delta, nfree=nfree)



