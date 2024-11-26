# JamesMisaka in 2023-11-06
# Analyze NEB and AutoNEB calculation result 
# part of ATST-Tools scripts

from ase.mep.neb import NEBTools # newest ase
from ase.io import read, write
import sys, os

# next dev: point-out transition state
class NEBPost():
    """Post-Processing of NEB result from Atoms object"""
    def __init__(self, images, n_max: int = 0):
        self.all_image = images
        if n_max == 0:
            print("=== n_max set to 0, automatically detect the images of chain by NEBTools ===")
            self.n_images = NEBTools(self.all_image)._guess_nimages()
            self.neb_chain = images[ - self.n_images:]
        elif (n_max > 0) and (type(n_max) == int):
            self.n_images = n_max + 2
            self.neb_chain = images[ - self.n_images:]
        else:
            raise ValueError("n_max must be a non-negative integer")

    def get_barrier(self):
        """Returns energy of all image and the barrier estimate from the NEB, along with the Delta E of the elementary reaction"""
        for i, atoms in enumerate(self.neb_chain):
            ene = atoms.get_potential_energy()
            print(f"num: {i}; Energy: {ene} (eV)")
        barrier = NEBTools(self.neb_chain).get_barrier(fit=True)
        # set fit to True/False is not determined yet
        print(f"Reaction Barrier and Energy Difference: {barrier} (eV)")
        return barrier
    
    def get_TS_stru(self, name="TS_get"):
        """Get TS structure from NEB chain"""
        raw_barrier = NEBTools(self.neb_chain).get_barrier(fit=False, raw=True)
        for atoms in self.neb_chain:
            if atoms.get_potential_energy() == raw_barrier[0]:
                write(f"{name}.cif", atoms, format="cif")
                write(f"{name}.stru", atoms, format="abacus")
                print(f"TS structure is saved as {name}.cif and {name}.stru")
                return

    def plot_neb_bands(self, label='nebplots_chain'):
        """makes plots of final neb band in the series in a single PDF"""
        return NEBTools(self.neb_chain).plot_bands(label=label)
    
    def plot_all_bands(self, label='nebplots_all'):
        """Gmakes plots of all band during neb in the series in a single PDF"""
        return NEBTools(self.all_image).plot_bands(label=label)
    
    def write_latest_bands(self, outname="neb_latest"):
        """write latest neb chain to file"""
        write(f"{outname}.traj", self.neb_chain, format="traj")
        write(f"{outname}.extxyz", self.neb_chain, format="extxyz")
        return
    
    def view_neb_bands(self):
        """view neb chain"""
        os.system(f'ase gui neb.traj@-{self.n_images}:')

if __name__ == "__main__":
    msg = '''
Usage: 
    For Traditional NEB process result: 
        python neb_post.py [traj_file] ([n_images])
    For AutoNEB result:
        python neb_post.py --autoneb \{[autoneb_traj_files]\}
    '''
    if len(sys.argv) < 2:
        print(msg)
    elif len(sys.argv) == 2:
        traj_file = sys.argv[1]
        all_images = read(traj_file, index=":", format="traj")
        result = NEBPost(all_images, 0)
        result.get_barrier()
        result.plot_all_bands()
        result.plot_neb_bands()
        result.write_latest_bands()
        result.get_TS_stru()
    else:
        if sys.argv[1] == "--autoneb":
            traj_files = sys.argv[2:]
            result_atoms = [read(traj, format="traj") for traj in traj_files]
            result = NEBPost(result_atoms, 0)
            result.get_barrier()
            result.plot_all_bands() # in autoneb we only have one chain
            result.write_latest_bands()
            result.get_TS_stru()
        else:
            traj_file = sys.argv[1]
            n_max = int(sys.argv[2])
            all_images = read(traj_file, index=":", format="traj")
            result = NEBPost(all_images, n_max)
            result.get_barrier()
            result.plot_all_bands()
            result.plot_neb_bands()
            result.write_latest_bands()
            result.get_TS_stru()
    