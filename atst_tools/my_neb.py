# Stored and Modified by QuantumMisaka in 2023-12
import sys
import threading
import warnings
from abc import ABC, abstractmethod
import time

import numpy as np

from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz

import ase.parallel
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import OptimizerConvergenceError
from ase.geometry import find_mic
from ase.utils import lazyproperty, deprecated
from ase.utils.forcecurve import fit_images
from ase.optimize.precon import Precon, PreconImages
from ase.optimize.ode import ode12r


class Spring:
    def __init__(self, atoms1, atoms2, energy1, energy2, k):
        self.atoms1 = atoms1
        self.atoms2 = atoms2
        self.energy1 = energy1
        self.energy2 = energy2
        self.k = k

    def _find_mic(self):
        pos1 = self.atoms1.get_positions()
        pos2 = self.atoms2.get_positions()
        # XXX If we want variable cells we will need to edit this.
        mic, _ = find_mic(pos2 - pos1, self.atoms1.cell, self.atoms1.pbc)
        return mic

    @lazyproperty
    def t(self):
        return self._find_mic()

    @lazyproperty
    def nt(self):
        return np.linalg.norm(self.t)


class NEBState:
    def __init__(self, neb, images, energies):
        self.neb = neb
        self.images = images
        self.energies = energies

    def spring(self, i):
        return Spring(self.images[i], self.images[i + 1],
                      self.energies[i], self.energies[i + 1],
                      self.neb.k[i])

    @lazyproperty
    def imax(self):
        return 1 + np.argsort(self.energies[1:-1])[-1]

    @property
    def emax(self):
        return self.energies[self.imax]

    @lazyproperty
    def eqlength(self):
        images = self.images
        beeline = (images[self.neb.nimages - 1].get_positions() -
                   images[0].get_positions())
        beelinelength = np.linalg.norm(beeline)
        return beelinelength / (self.neb.nimages - 1)

    @lazyproperty
    def nimages(self):
        return len(self.images)

    @property
    def precon(self):
        return self.neb.precon


class NEBMethod(ABC):
    def __init__(self, neb):
        self.neb = neb

    @abstractmethod
    def get_tangent(self, state, spring1, spring2, i):
        ...

    @abstractmethod
    def add_image_force(self, state, tangential_force, tangent, imgforce,
                        spring1, spring2, i):
        ...

    def adjust_positions(self, positions):
        return positions


class ImprovedTangentMethod(NEBMethod):
    """
    Tangent estimates are improved according to Eqs. 8-11 in paper I.
    Tangents are weighted at extrema to ensure smooth transitions between
    the positive and negative tangents.
    """

    def get_tangent(self, state, spring1, spring2, i):
        energies = state.energies
        if energies[i + 1] > energies[i] > energies[i - 1]:
            tangent = spring2.t.copy()
        elif energies[i + 1] < energies[i] < energies[i - 1]:
            tangent = spring1.t.copy()
        else:
            deltavmax = max(abs(energies[i + 1] - energies[i]),
                            abs(energies[i - 1] - energies[i]))
            deltavmin = min(abs(energies[i + 1] - energies[i]),
                            abs(energies[i - 1] - energies[i]))
            if energies[i + 1] > energies[i - 1]:
                tangent = spring2.t * deltavmax + spring1.t * deltavmin
            else:
                tangent = spring2.t * deltavmin + spring1.t * deltavmax
        # Normalize the tangent vector
        tangent /= np.linalg.norm(tangent)
        return tangent

    def add_image_force(self, state, tangential_force, tangent, imgforce,
                        spring1, spring2, i):
        imgforce -= tangential_force * tangent
        # Improved parallel spring force (formula 12 of paper I)
        imgforce += (spring2.nt * spring2.k - spring1.nt * spring1.k) * tangent


class ASENEBMethod(NEBMethod):
    """
    Standard NEB implementation in ASE. The tangent of each image is
    estimated from the spring closest to the saddle point in each
    spring pair.
    """

    def get_tangent(self, state, spring1, spring2, i):
        imax = self.neb.imax
        if i < imax:
            tangent = spring2.t
        elif i > imax:
            tangent = spring1.t
        else:
            tangent = spring1.t + spring2.t
        return tangent

    def add_image_force(self, state, tangential_force, tangent, imgforce,
                        spring1, spring2, i):
        # Magnitude for normalizing. Ensure it is not 0
        tangent_mag = np.vdot(tangent, tangent) or 1
        factor = tangent / tangent_mag
        imgforce -= tangential_force * factor
        imgforce -= np.vdot(
            spring1.t * spring1.k -
            spring2.t * spring2.k, tangent) * factor


class FullSpringMethod(NEBMethod):
    """
    Elastic band method. The full spring force is included.
    """

    def get_tangent(self, state, spring1, spring2, i):
        # Tangents are bisections of spring-directions
        # (formula C8 of paper III)
        tangent = spring1.t / spring1.nt + spring2.t / spring2.nt
        tangent /= np.linalg.norm(tangent)
        return tangent

    def add_image_force(self, state, tangential_force, tangent, imgforce,
                        spring1, spring2, i):
        imgforce -= tangential_force * tangent
        energies = state.energies
        # Spring forces
        # Eqs. C1, C5, C6 and C7 in paper III)
        f1 = -(spring1.nt -
               state.eqlength) * spring1.t / spring1.nt * spring1.k
        f2 = (spring2.nt - state.eqlength) * spring2.t / spring2.nt * spring2.k
        if self.neb.climb and abs(i - self.neb.imax) == 1:
            deltavmax = max(abs(energies[i + 1] - energies[i]),
                            abs(energies[i - 1] - energies[i]))
            deltavmin = min(abs(energies[i + 1] - energies[i]),
                            abs(energies[i - 1] - energies[i]))
            imgforce += (f1 + f2) * deltavmin / deltavmax
        else:
            imgforce += f1 + f2


class BaseSplineMethod(NEBMethod):
    """
    Base class for SplineNEB and String methods

    Can optionally be preconditioned, as described in the following article:

        S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
        150, 094109 (2019)
        https://dx.doi.org/10.1063/1.5064465
    """

    def __init__(self, neb):
        NEBMethod.__init__(self, neb)

    def get_tangent(self, state, spring1, spring2, i):
        return state.precon.get_tangent(i)

    def add_image_force(self, state, tangential_force, tangent, imgforce,
                        spring1, spring2, i):
        # project out tangential component (Eqs 6 and 7 in Paper IV)
        imgforce -= tangential_force * tangent


class SplineMethod(BaseSplineMethod):
    """
    NEB using spline interpolation, plus optional preconditioning
    """

    def add_image_force(self, state, tangential_force, tangent, imgforce,
                        spring1, spring2, i):
        super().add_image_force(state, tangential_force,
                                tangent, imgforce, spring1, spring2, i)
        eta = state.precon.get_spring_force(i, spring1.k, spring2.k, tangent)
        imgforce += eta


class StringMethod(BaseSplineMethod):
    """
    String method using spline interpolation, plus optional preconditioning
    """

    def adjust_positions(self, positions):
        # fit cubic spline to positions, reinterpolate to equispace images
        # note this uses the preconditioned distance metric.
        fit = self.neb.spline_fit(positions)
        new_s = np.linspace(0.0, 1.0, self.neb.nimages)
        new_positions = fit.x(new_s[1:-1]).reshape(-1, 3)
        return new_positions


def get_neb_method(neb, method):
    if method == 'eb':
        return FullSpringMethod(neb)
    elif method == 'aseneb':
        return ASENEBMethod(neb)
    elif method == 'improvedtangent':
        return ImprovedTangentMethod(neb)
    elif method == 'spline':
        return SplineMethod(neb)
    elif method == 'string':
        return StringMethod(neb)
    else:
        raise ValueError(f'Bad method: {method}')


class BaseNEB:
    def __init__(self, images, k=0.1, climb=False, parallel=False,
                 remove_rotation_and_translation=False, world=None,
                 method='aseneb', allow_shared_calculator=False, precon=None):

        self.images = images
        self.climb = climb
        self.parallel = parallel
        self.allow_shared_calculator = allow_shared_calculator

        for img in images:
            if len(img) != self.natoms:
                raise ValueError('Images have different numbers of atoms')
            if np.any(img.pbc != images[0].pbc):
                raise ValueError('Images have different boundary conditions')
            if np.any(img.get_atomic_numbers() !=
                      images[0].get_atomic_numbers()):
                raise ValueError('Images have atoms in different orders')
            # check periodic cell directions
            cell_ok = True
            for pbc, vc, vc0 in zip(img.pbc, img.cell, images[0].cell):
                if pbc and np.any(np.abs(vc - vc0) > 1e-8):
                    cell_ok = False
            if not cell_ok:
                raise NotImplementedError(
                    "Variable cell in periodic directions "
                    "is not implemented yet for NEB")

        self.emax = np.nan

        self.remove_rotation_and_translation = remove_rotation_and_translation

        if method in ['aseneb', 'eb', 'improvedtangent', 'spline', 'string']:
            self.method = method
        else:
            raise NotImplementedError(method)

        if precon is not None and method not in ['spline', 'string']:
            raise NotImplementedError(f'no precon implemented: {method}')
        self.precon = precon

        self.neb_method = get_neb_method(self, method)
        if isinstance(k, (float, int)):
            k = [k] * (self.nimages - 1)
        self.k = list(k)

        if world is None:
            world = ase.parallel.world
        self.world = world

        if parallel:
            if self.allow_shared_calculator:
                raise RuntimeError(
                    "Cannot use shared calculators in parallel in NEB.")
        self.real_forces = None  # ndarray of shape (nimages, natom, 3)
        self.stresses = None # ndarray of shape (nimages, 6) # all stress part added by QuantumMisaka
        self.energies = None  # ndarray of shape (nimages,)
        self.residuals = None  # ndarray of shape (nimages,)

    @property
    def natoms(self):
        return len(self.images[0])

    @property
    def nimages(self):
        return len(self.images)

    @staticmethod
    def freeze_results_on_image(atoms: ase.Atoms,
                                **results_to_include):
        atoms.calc = SinglePointCalculator(atoms=atoms, **results_to_include)

    def interpolate(self, method='linear', mic=False, apply_constraint=None):
        """Interpolate the positions of the interior images between the
        initial state (image 0) and final state (image -1).

        method: str
            Method by which to interpolate: 'linear' or 'idpp'.
            linear provides a standard straight-line interpolation, while
            idpp uses an image-dependent pair potential.
        mic: bool
            Use the minimum-image convention when interpolating.
        apply_constraint: bool
            Controls if the constraints attached to the images
            are ignored or applied when setting the interpolated positions.
            Default value is None, in this case the resulting constrained
            positions (apply_constraint=True) are compared with unconstrained
            positions (apply_constraint=False),
            if the positions are not the same
            the user is required to specify the desired behaviour
            by setting up apply_constraint keyword argument to False or True.
        """
        if self.remove_rotation_and_translation:
            minimize_rotation_and_translation(self.images[0], self.images[-1])

        interpolate(self.images, mic, apply_constraint=apply_constraint)

        if method == 'idpp':
            idpp_interpolate(images=self, traj=None, log=None, mic=mic)

    @deprecated("Please use NEB's interpolate(method='idpp') method or "
                "directly call the idpp_interpolate function from ase.mep")
    def idpp_interpolate(self, traj='idpp.traj', log='idpp.log', fmax=0.1,
                         optimizer=MDMin, mic=False, steps=100):
        idpp_interpolate(self, traj=traj, log=log, fmax=fmax,
                         optimizer=optimizer, mic=mic, steps=steps)

    def get_positions(self):
        positions = np.empty(((self.nimages - 2) * self.natoms, 3))
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            positions[n1:n2] = image.get_positions()
            n1 = n2
        return positions

    def set_positions(self, positions, adjust_positions=True):
        if adjust_positions:
            # optional reparameterisation step: some NEB methods need to adjust
            # positions e.g. string method does this to equispace the images)
            positions = self.neb_method.adjust_positions(positions)
        n1 = 0
        for image in self.images[1:-1]:
            n2 = n1 + self.natoms
            image.set_positions(positions[n1:n2])
            n1 = n2

    def get_forces(self):
        """Evaluate and return the forces."""
        images = self.images

        if not self.allow_shared_calculator:
            calculators = [image.calc for image in images
                           if image.calc is not None]
            if len(set(calculators)) != len(calculators):
                msg = ('One or more NEB images share the same calculator.  '
                       'Each image must have its own calculator.  '
                       'You may wish to use the ase.mep.SingleCalculatorNEB '
                       'class instead, although using separate calculators '
                       'is recommended.')
                raise ValueError(msg)

        forces = np.empty(((self.nimages - 2), self.natoms, 3))
        energies = np.empty(self.nimages)
        # used by QuantumMisaka
        real_forces = np.empty(((self.nimages - 2), self.natoms, 3))
        stresses = np.empty(((self.nimages - 2), 6))

        if self.remove_rotation_and_translation:
            for i in range(1, self.nimages):
                minimize_rotation_and_translation(images[i - 1], images[i])

        if self.method != 'aseneb':
            energies[0] = images[0].get_potential_energy()
            energies[-1] = images[-1].get_potential_energy()

        if not self.parallel:
            # Do all images - one at a time:
            for i in range(1, self.nimages - 1):
                forces[i - 1] = images[i].get_forces()
                energies[i] = images[i].get_potential_energy()

        elif self.world.size == 1:
            def run(image, energies, forces):
                forces[:] = image.get_forces()
                energies[:] = image.get_potential_energy()

            threads = [threading.Thread(target=run,
                                        args=(images[i],
                                              energies[i:i + 1],
                                              forces[i - 1:i]))
                       for i in range(1, self.nimages - 1)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            # Parallelize over images:
            i = self.world.rank * (self.nimages - 2) // self.world.size + 1
            try:
                forces[i - 1] = images[i].get_forces()
                real_forces[i - 1] = images[i].get_forces(apply_constraint=False)
                energies[i] = images[i].get_potential_energy()
                stresses[i] = images[i].get_stress()
            except Exception:
                # Make sure other images also fail:
                error = self.world.sum(1.0)
                raise
            else:
                error = self.world.sum(0.0)
                if error:
                    raise RuntimeError('Parallel NEB failed!')

            for i in range(1, self.nimages - 1):
                root = (i - 1) * self.world.size // (self.nimages - 2)
                self.world.broadcast(energies[i:i + 1], root)
                self.world.broadcast(forces[i - 1], root)
                self.world.broadcast(real_forces[i - 1], root)
                self.world.broadcast(stresses[i:i + 1], root)

        # if this is the first force call, we need to build the preconditioners
        if (self.precon is None or isinstance(self.precon, str) or
                isinstance(self.precon, Precon) or
                isinstance(self.precon, list)):
            self.precon = PreconImages(self.precon, images)

        # apply preconditioners to transform forces
        # for the default IdentityPrecon this does not change their values
        precon_forces = self.precon.apply(forces, index=slice(1, -1))

        # Save for later use in iterimages:
        self.energies = energies
        #self.real_forces = np.zeros((self.nimages, self.natoms, 3))
        #self.real_forces[1:-1] = forces
        self.real_forces = real_forces
        self.stresses = stresses # all stress part added by QuantumMisaka

        state = NEBState(self, images, energies)

        # Can we get rid of self.energies, self.imax, self.emax etc.?
        self.imax = state.imax
        self.emax = state.emax

        spring1 = state.spring(0)

        self.residuals = []
        for i in range(1, self.nimages - 1):
            spring2 = state.spring(i)
            tangent = self.neb_method.get_tangent(state, spring1, spring2, i)

            # Get overlap between full PES-derived force and tangent
            tangential_force = np.vdot(forces[i - 1], tangent)

            # from now on we use the preconditioned forces (equal for precon=ID)
            imgforce = precon_forces[i - 1]

            if i == self.imax and self.climb:
                """The climbing image, imax, is not affected by the spring
                   forces. This image feels the full PES-derived force,
                   but the tangential component is inverted:
                   see Eq. 5 in paper II."""
                if self.method == 'aseneb':
                    tangent_mag = np.vdot(tangent, tangent)  # For normalizing
                    imgforce -= 2 * tangential_force / tangent_mag * tangent
                else:
                    imgforce -= 2 * tangential_force * tangent
            else:
                self.neb_method.add_image_force(state, tangential_force,
                                                tangent, imgforce, spring1,
                                                spring2, i)
                # compute the residual - with ID precon, this is just max force
                residual = self.precon.get_residual(i, imgforce)
                self.residuals.append(residual)

            spring1 = spring2

        return precon_forces.reshape((-1, 3))

    def get_residual(self):
        """Return residual force along the band.

        Typically this the maximum force component on any image. For
        non-trivial preconditioners, the appropriate preconditioned norm
        is used to compute the residual.
        """
        if self.residuals is None:
            raise RuntimeError("get_residual() called before get_forces()")
        return np.max(self.residuals)

    def get_potential_energy(self, force_consistent=False):
        """Return the maximum potential energy along the band.
        Note that the force_consistent keyword is ignored and is only
        present for compatibility with ase.Atoms.get_potential_energy."""
        return self.emax

    def set_calculators(self, calculators):
        """Set new calculators to the images.

        Parameters
        ----------
        calculators : Calculator / list(Calculator)
            calculator(s) to attach to images
              - single calculator, only if allow_shared_calculator=True
            list of calculators if length:
              - length nimages, set to all images
              - length nimages-2, set to non-end images only
        """

        if not isinstance(calculators, list):
            if self.allow_shared_calculator:
                calculators = [calculators] * self.nimages
            else:
                raise RuntimeError("Cannot set shared calculator to NEB "
                                   "with allow_shared_calculator=False")

        n = len(calculators)
        if n == self.nimages:
            for i in range(self.nimages):
                self.images[i].calc = calculators[i]
        elif n == self.nimages - 2:
            for i in range(1, self.nimages - 1):
                self.images[i].calc = calculators[i - 1]
        else:
            raise RuntimeError(
                'len(calculators)=%d does not fit to len(images)=%d'
                % (n, self.nimages))

    def __len__(self):
        # Corresponds to number of optimizable degrees of freedom, i.e.
        # virtual atom count for the optimization algorithm.
        return (self.nimages - 2) * self.natoms

    def iterimages(self):
        # Allows trajectory to convert NEB into several images
        for i, atoms in enumerate(self.images):
            if i == 0 or i == self.nimages - 1:
                yield atoms
            else:
                atoms = atoms.copy()
                self.freeze_results_on_image(
                    atoms, energy=self.energies[i],
                    forces=self.real_forces[i],
                    stress=self.stresses[i])
                # all stress part added by QuantumMisaka

                yield atoms

    def spline_fit(self, positions=None, norm='precon'):
        """
        Fit a cubic spline to this NEB

        Args:
            norm (str, optional): Norm to use: 'precon' (default) or 'euclidean'

        Returns:
            fit: ase.precon.precon.SplineFit instance
        """
        if norm == 'precon':
            if self.precon is None or isinstance(self.precon, str):
                self.precon = PreconImages(self.precon, self.images)
            precon = self.precon
            # if this is the first call, we need to build the preconditioners
        elif norm == 'euclidean':
            precon = PreconImages('ID', self.images)
        else:
            raise ValueError(f'unsupported norm {norm}')
        return precon.spline_fit(positions)

    def integrate_forces(self, spline_points=1000, bc_type='not-a-knot'):
        """Use spline fit to integrate forces along MEP to approximate
        energy differences using the virtual work approach.

        Args:
            spline_points (int, optional): Number of points. Defaults to 1000.
            bc_type (str, optional): Boundary conditions, default 'not-a-knot'.

        Returns:
            s: reaction coordinate in range [0, 1], with `spline_points` entries
            E: result of integrating forces, on the same grid as `s`.
            F: projected forces along MEP
        """
        # note we use standard Euclidean rather than preconditioned norm
        # to compute the virtual work
        fit = self.spline_fit(norm='euclidean')
        forces = np.array([image.get_forces().reshape(-1)
                           for image in self.images])
        f = CubicSpline(fit.s, forces, bc_type=bc_type)

        s = np.linspace(0.0, 1.0, spline_points, endpoint=True)
        dE = f(s) * fit.dx_ds(s)
        F = dE.sum(axis=1)
        E = -cumtrapz(F, s, initial=0.0)
        return s, E, F


class DyNEB(BaseNEB):
    def __init__(self, images, k=0.1, fmax=0.05, climb=False, parallel=False,
                 remove_rotation_and_translation=False, world=None,
                 dynamic_relaxation=True, scale_fmax=0., method='aseneb',
                 allow_shared_calculator=False, precon=None):
        """
        Subclass of NEB that allows for scaled and dynamic optimizations of
        images. This method, which only works in series, does not perform
        force calls on images that are below the convergence criterion.
        The convergence criteria can be scaled with a displacement metric
        to focus the optimization on the saddle point region.

        'Scaled and Dynamic Optimizations of Nudged Elastic Bands',
        P. Lindgren, G. Kastlunger and A. A. Peterson,
        J. Chem. Theory Comput. 15, 11, 5787-5793 (2019).

        dynamic_relaxation: bool
            True skips images with forces below the convergence criterion.
            This is updated after each force call; if a previously converged
            image goes out of tolerance (due to spring adjustments between
            the image and its neighbors), it will be optimized again.
            False reverts to the default NEB implementation.

        fmax: float
            Must be identical to the fmax of the optimizer.

        scale_fmax: float
            Scale convergence criteria along band based on the distance between
            an image and the image with the highest potential energy. This
            keyword determines how rapidly the convergence criteria are scaled.
        """
        super().__init__(
            images, k=k, climb=climb, parallel=parallel,
            remove_rotation_and_translation=remove_rotation_and_translation,
            world=world, method=method,
            allow_shared_calculator=allow_shared_calculator, precon=precon)
        self.fmax = fmax
        self.dynamic_relaxation = dynamic_relaxation
        self.scale_fmax = scale_fmax

        if not self.dynamic_relaxation and self.scale_fmax:
            msg = ('Scaled convergence criteria only implemented in series '
                   'with dynamic relaxation.')
            raise ValueError(msg)

    def set_positions(self, positions):
        if not self.dynamic_relaxation:
            return super().set_positions(positions)

        n1 = 0
        for i, image in enumerate(self.images[1:-1]):
            if self.parallel:
                msg = ('Dynamic relaxation does not work efficiently '
                       'when parallelizing over images. Try AutoNEB '
                       'routine for freezing images in parallel.')
                raise ValueError(msg)
            else:
                forces_dyn = self._fmax_all(self.images)
                if forces_dyn[i] < self.fmax:
                    n1 += self.natoms
                else:
                    n2 = n1 + self.natoms
                    image.set_positions(positions[n1:n2])
                    n1 = n2

    def _fmax_all(self, images):
        """Store maximum force acting on each image in list. This is used in
           the dynamic optimization routine in the set_positions() function."""
        n = self.natoms
        forces = self.get_forces()
        fmax_images = [
            np.sqrt((forces[n * i:n + n * i] ** 2).sum(axis=1)).max()
            for i in range(self.nimages - 2)]
        return fmax_images

    def get_forces(self):
        forces = super().get_forces()
        if not self.dynamic_relaxation:
            return forces

        """Get NEB forces and scale the convergence criteria to focus
           optimization on saddle point region. The keyword scale_fmax
           determines the rate of convergence scaling."""
        n = self.natoms
        for i in range(self.nimages - 2):
            n1 = n * i
            n2 = n1 + n
            force = np.sqrt((forces[n1:n2] ** 2.).sum(axis=1)).max()
            n_imax = (self.imax - 1) * n  # Image with highest energy.

            positions = self.get_positions()
            pos_imax = positions[n_imax:n_imax + n]

            """Scale convergence criteria based on distance between an
               image and the image with the highest potential energy."""
            rel_pos = np.sqrt(((positions[n1:n2] - pos_imax) ** 2).sum())
            if force < self.fmax * (1 + rel_pos * self.scale_fmax):
                if i == self.imax - 1:
                    # Keep forces at saddle point for the log file.
                    pass
                else:
                    # Set forces to zero before they are sent to optimizer.
                    forces[n1:n2, :] = 0
        return forces


def _check_deprecation(keyword, kwargs):
    if keyword in kwargs:
        warnings.warn(f'Keyword {keyword} of NEB is deprecated.  '
                      'Please use the DyNEB class instead for dynamic '
                      'relaxation', FutureWarning)


class NEB(DyNEB):
    def __init__(self, images, k=0.1, climb=False, parallel=False,
                 remove_rotation_and_translation=False, world=None,
                 method='aseneb', allow_shared_calculator=False,
                 precon=None, **kwargs):
        """Nudged elastic band.

        Paper I:

            G. Henkelman and H. Jonsson, Chem. Phys, 113, 9978 (2000).
            :doi:`10.1063/1.1323224`

        Paper II:

            G. Henkelman, B. P. Uberuaga, and H. Jonsson, Chem. Phys,
            113, 9901 (2000).
            :doi:`10.1063/1.1329672`

        Paper III:

            E. L. Kolsbjerg, M. N. Groves, and B. Hammer, J. Chem. Phys,
            145, 094107 (2016)
            :doi:`10.1063/1.4961868`

        Paper IV:

            S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
            150, 094109 (2019)
            https://dx.doi.org/10.1063/1.5064465

        images: list of Atoms objects
            Images defining path from initial to final state.
        k: float or list of floats
            Spring constant(s) in eV/Ang.  One number or one for each spring.
        climb: bool
            Use a climbing image (default is no climbing image).
        parallel: bool
            Distribute images over processors.
        remove_rotation_and_translation: bool
            TRUE actives NEB-TR for removing translation and
            rotation during NEB. By default applied non-periodic
            systems
        method: string of method
            Choice betweeen five methods:

            * aseneb: standard ase NEB implementation
            * improvedtangent: Paper I NEB implementation
            * eb: Paper III full spring force implementation
            * spline: Paper IV spline interpolation (supports precon)
            * string: Paper IV string method (supports precon)
        allow_shared_calculator: bool
            Allow images to share the same calculator between them.
            Incompatible with parallelisation over images.
        precon: string, :class:`ase.optimize.precon.Precon` instance or list of
            instances. If present, enable preconditioing as in Paper IV. This is
            possible using the 'spline' or 'string' methods only.
            Default is no preconditioning (precon=None), which is converted to
            a list of :class:`ase.precon.precon.IdentityPrecon` instances.
        """
        for keyword in 'dynamic_relaxation', 'fmax', 'scale_fmax':
            _check_deprecation(keyword, kwargs)
        defaults = dict(dynamic_relaxation=False,
                        fmax=0.05,
                        scale_fmax=0.0)
        defaults.update(kwargs)
        # Only reason for separating BaseNEB/NEB is that we are
        # deprecating dynamic_relaxation.
        #
        # We can turn BaseNEB into NEB once we get rid of the
        # deprecated variables.
        #
        # Then we can also move DyNEB into ase.dyneb without cyclic imports.
        # We can do that in ase-3.22 or 3.23.
        super().__init__(
            images, k=k, climb=climb, parallel=parallel,
            remove_rotation_and_translation=remove_rotation_and_translation,
            world=world, method=method,
            allow_shared_calculator=allow_shared_calculator,
            precon=precon,
            **defaults)


class NEBOptimizer(Optimizer):
    """
    This optimizer applies an adaptive ODE solver to a NEB

    Details of the adaptive ODE solver are described in paper IV
    """

    def __init__(self,
                 neb,
                 restart=None, logfile='-', trajectory=None,
                 master=None,
                 append_trajectory=False,
                 method='ODE',
                 alpha=0.01,
                 verbose=0,
                 rtol=0.1,
                 C1=1e-2,
                 C2=2.0):

        super().__init__(atoms=neb, restart=restart,
                         logfile=logfile, trajectory=trajectory,
                         master=master,
                         append_trajectory=append_trajectory,
                         force_consistent=False)
        self.neb = neb

        method = method.lower()
        methods = ['ode', 'static', 'krylov']
        if method not in methods:
            raise ValueError(f'method must be one of {methods}')
        self.method = method

        self.alpha = alpha
        self.verbose = verbose
        self.rtol = rtol
        self.C1 = C1
        self.C2 = C2

    def force_function(self, X):
        positions = X.reshape((self.neb.nimages - 2) *
                              self.neb.natoms, 3)
        self.neb.set_positions(positions)
        forces = self.neb.get_forces().reshape(-1)
        return forces

    def get_residual(self, F=None, X=None):
        return self.neb.get_residual()

    def log(self):
        fmax = self.get_residual()
        T = time.localtime()
        if self.logfile is not None:
            name = f'{self.__class__.__name__}[{self.method}]'
            if self.nsteps == 0:
                args = (" " * len(name), "Step", "Time", "fmax")
                msg = "%s  %4s %8s %12s\n" % args
                self.logfile.write(msg)

            args = (name, self.nsteps, T[3], T[4], T[5], fmax)
            msg = "%s:  %3d %02d:%02d:%02d %12.4f\n" % args
            self.logfile.write(msg)
            self.logfile.flush()

    def callback(self, X, F=None):
        self.log()
        self.call_observers()
        self.nsteps += 1

    def run_ode(self, fmax):
        try:
            ode12r(self.force_function,
                   self.neb.get_positions().reshape(-1),
                   fmax=fmax,
                   rtol=self.rtol,
                   C1=self.C1,
                   C2=self.C2,
                   steps=self.max_steps,
                   verbose=self.verbose,
                   callback=self.callback,
                   residual=self.get_residual)
            return True
        except OptimizerConvergenceError:
            return False

    def run_static(self, fmax):
        X = self.neb.get_positions().reshape(-1)
        for step in range(self.max_steps):
            F = self.force_function(X)
            if self.neb.get_residual() <= fmax:
                return True
            X += self.alpha * F
            self.callback(X)
        return False

    def run(self, fmax=0.05, steps=None, method=None):
        """
        Optimize images to obtain the minimum energy path

        Parameters
        ----------
        fmax - desired force tolerance
        steps - maximum number of steps
        """
        if steps:
            self.max_steps = steps
        if method is None:
            method = self.method
        if method == 'ode':
            return self.run_ode(fmax)
        elif method == 'static':
            return self.run_static(fmax)
        else:
            raise ValueError(f'unknown method: {self.method}')


class IDPP(Calculator):
    """Image dependent pair potential.

    See:
        Improved initial guess for minimum energy path calculations.
        Søren Smidstrup, Andreas Pedersen, Kurt Stokbro and Hannes Jónsson
        Chem. Phys. 140, 214106 (2014)
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, target, mic):
        Calculator.__init__(self)
        self.target = target
        self.mic = mic

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        P = atoms.get_positions()
        d = []
        D = []
        for p in P:
            Di = P - p
            if self.mic:
                Di, di = find_mic(Di, atoms.get_cell(), atoms.get_pbc())
            else:
                di = np.sqrt((Di ** 2).sum(1))
            d.append(di)
            D.append(Di)
        d = np.array(d)
        D = np.array(D)

        dd = d - self.target
        d.ravel()[::len(d) + 1] = 1  # avoid dividing by zero
        d4 = d ** 4
        e = 0.5 * (dd ** 2 / d4).sum()
        f = -2 * ((dd * (1 - 2 * dd / d) / d ** 5)[..., np.newaxis] * D).sum(
            0)
        self.results = {'energy': e, 'forces': f}


@deprecated("SingleCalculatorNEB is deprecated. "
            "Please use NEB(allow_shared_calculator=True) instead.")
class SingleCalculatorNEB(NEB):
    def __init__(self, images, *args, **kwargs):
        kwargs["allow_shared_calculator"] = True
        super().__init__(images, *args, **kwargs)


def interpolate(images, mic=False, interpolate_cell=False,
                use_scaled_coord=False, apply_constraint=None):
    """Given a list of images, linearly interpolate the positions of the
    interior images.

    mic: bool
         Map movement into the unit cell by using the minimum image convention.
    interpolate_cell: bool
         Interpolate the three cell vectors linearly just like the atomic
         positions. Not implemented for NEB calculations!
    use_scaled_coord: bool
         Use scaled/internal/fractional coordinates instead of real ones for the
         interpolation. Not implemented for NEB calculations!
    apply_constraint: bool
         Controls if the constraints attached to the images
         are ignored or applied when setting the interpolated positions.
         Default value is None, in this case the resulting constrained positions
         (apply_constraint=True) are compared with unconstrained positions
         (apply_constraint=False), if the positions are not the same
         the user is required to specify the desired behaviour
         by setting up apply_constraint keyword argument to False or True.
    """
    if use_scaled_coord:
        pos1 = images[0].get_scaled_positions(wrap=mic)
        pos2 = images[-1].get_scaled_positions(wrap=mic)
    else:
        pos1 = images[0].get_positions()
        pos2 = images[-1].get_positions()
    d = pos2 - pos1
    if not use_scaled_coord and mic:
        d = find_mic(d, images[0].get_cell(), images[0].pbc)[0]
    d /= (len(images) - 1.0)
    if interpolate_cell:
        cell1 = images[0].get_cell()
        cell2 = images[-1].get_cell()
        cell_diff = cell2 - cell1
        cell_diff /= (len(images) - 1.0)
    for i in range(1, len(images) - 1):
        # first the new cell, otherwise scaled positions are wrong
        if interpolate_cell:
            images[i].set_cell(cell1 + i * cell_diff)
        new_pos = pos1 + i * d
        if use_scaled_coord:
            images[i].set_scaled_positions(new_pos)
        else:
            if apply_constraint is None:
                unconstrained_image = images[i].copy()
                unconstrained_image.set_positions(new_pos,
                                                  apply_constraint=False)
                images[i].set_positions(new_pos, apply_constraint=True)
                try:
                    np.testing.assert_allclose(unconstrained_image.positions,
                                               images[i].positions)
                except AssertionError:
                    raise RuntimeError(f"Constraint(s) in image number {i} \n"
                                       f"affect the interpolation results.\n"
                                       "Please specify if you want to \n"
                                       "apply or ignore the constraints \n"
                                       "during the interpolation \n"
                                       "with apply_constraint argument.")
            else:
                images[i].set_positions(new_pos,
                                        apply_constraint=apply_constraint)


def idpp_interpolate(images, traj='idpp.traj', log='idpp.log', fmax=0.1,
                     optimizer=MDMin, mic=False, steps=100):
    """Interpolate using the IDPP method. 'images' can either be a plain
    list of images or an NEB object (containing a list of images)."""
    if hasattr(images, 'interpolate'):
        neb = images
    else:
        neb = NEB(images)

    d1 = neb.images[0].get_all_distances(mic=mic)
    d2 = neb.images[-1].get_all_distances(mic=mic)
    d = (d2 - d1) / (neb.nimages - 1)
    real_calcs = []
    for i, image in enumerate(neb.images):
        real_calcs.append(image.calc)
        image.calc = IDPP(d1 + i * d, mic=mic)

    with optimizer(neb, trajectory=traj, logfile=log) as opt:
        opt.run(fmax=fmax, steps=steps)

    for image, calc in zip(neb.images, real_calcs):
        image.calc = calc


class NEBTools:
    """Class to make many of the common tools for NEB analysis available to
    the user. Useful for scripting the output of many jobs. Initialize with
    list of images which make up one or more band of the NEB relaxation."""

    def __init__(self, images):
        self.images = images

    @deprecated('NEBTools.get_fit() is deprecated.  '
                'Please use ase.utils.forcecurve.fit_images(images).')
    def get_fit(self):
        return fit_images(self.images)

    def get_barrier(self, fit=True, raw=False):
        """Returns the barrier estimate from the NEB, along with the
        Delta E of the elementary reaction. If fit=True, the barrier is
        estimated based on the interpolated fit to the images; if
        fit=False, the barrier is taken as the maximum-energy image
        without interpolation. Set raw=True to get the raw energy of the
        transition state instead of the forward barrier."""
        forcefit = fit_images(self.images)
        energies = forcefit.energies
        fit_energies = forcefit.fit_energies
        dE = energies[-1] - energies[0]
        if fit:
            barrier = max(fit_energies)
        else:
            barrier = max(energies)
        if raw:
            barrier += self.images[0].get_potential_energy()
        return barrier, dE

    def get_fmax(self, **kwargs):
        """Returns fmax, as used by optimizers with NEB."""
        neb = NEB(self.images, **kwargs)
        forces = neb.get_forces()
        return np.sqrt((forces ** 2).sum(axis=1).max())

    def plot_band(self, ax=None):
        """Plots the NEB band on matplotlib axes object 'ax'. If ax=None
        returns a new figure object."""
        forcefit = fit_images(self.images)
        ax = forcefit.plot(ax=ax)
        return ax.figure

    def plot_bands(self, constant_x=False, constant_y=False,
                   nimages=None, label='nebplots'):
        """Given a trajectory containing many steps of a NEB, makes
        plots of each band in the series in a single PDF.

        constant_x: bool
            Use the same x limits on all plots.
        constant_y: bool
            Use the same y limits on all plots.
        nimages: int
            Number of images per band. Guessed if not supplied.
        label: str
            Name for the output file. .pdf will be appended.
        """
        from matplotlib import pyplot
        from matplotlib.backends.backend_pdf import PdfPages
        if nimages is None:
            nimages = self._guess_nimages()
        nebsteps = len(self.images) // nimages
        if constant_x or constant_y:
            sys.stdout.write('Scaling axes.\n')
            sys.stdout.flush()
            # Plot all to one plot, then pull its x and y range.
            fig, ax = pyplot.subplots()
            for index in range(nebsteps):
                images = self.images[index * nimages:(index + 1) * nimages]
                NEBTools(images).plot_band(ax=ax)
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
            pyplot.close(fig)  # Reference counting "bug" in pyplot.
        with PdfPages(label + '.pdf') as pdf:
            for index in range(nebsteps):
                sys.stdout.write('\rProcessing band {:10d} / {:10d}'
                                 .format(index + 1, nebsteps))
                sys.stdout.flush()
                fig, ax = pyplot.subplots()
                images = self.images[index * nimages:(index + 1) * nimages]
                NEBTools(images).plot_band(ax=ax)
                if constant_x:
                    ax.set_xlim(xlim)
                if constant_y:
                    ax.set_ylim(ylim)
                pdf.savefig(fig)
                pyplot.close(fig)  # Reference counting "bug" in pyplot.
        sys.stdout.write('\n')

    def _guess_nimages(self):
        """Attempts to guess the number of images per band from
        a trajectory, based solely on the repetition of the
        potential energy of images. This should also work for symmetric
        cases."""
        e_first = self.images[0].get_potential_energy()
        nimages = None
        for index, image in enumerate(self.images[1:], start=1):
            e = image.get_potential_energy()
            if e == e_first:
                # Need to check for symmetric case when e_first = e_last.
                try:
                    e_next = self.images[index + 1].get_potential_energy()
                except IndexError:
                    pass
                else:
                    if e_next == e_first:
                        nimages = index + 1  # Symmetric
                        break
                nimages = index  # Normal
                break
        if nimages is None:
            sys.stdout.write('Appears to be only one band in the images.\n')
            return len(self.images)
        # Sanity check that the energies of the last images line up too.
        e_last = self.images[nimages - 1].get_potential_energy()
        e_nextlast = self.images[2 * nimages - 1].get_potential_energy()
        if not (e_last == e_nextlast):
            raise RuntimeError('Could not guess number of images per band.')
        sys.stdout.write('Number of images per band guessed to be {:d}.\n'
                         .format(nimages))
        return nimages


class NEBtools(NEBTools):
    @deprecated('NEBtools has been renamed; please use NEBTools.')
    def __init__(self, images):
        NEBTools.__init__(self, images)


@deprecated('Please use NEBTools.plot_band_from_fit.')
def plot_band_from_fit(s, E, Sfit, Efit, lines, ax=None):
    NEBTools.plot_band_from_fit(s, E, Sfit, Efit, lines, ax=None)


def fit0(*args, **kwargs):
    raise DeprecationWarning('fit0 is deprecated. Use `fit_raw` from '
                             '`ase.utils.forcecurve` instead.')
