from typing import Union
import sympy

import underworld3
import underworld3.timing as timing
from ._jitextension import getext

include "./petsc_extras.pxi"

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetObjective( PetscDS, PetscInt, PetscDSResidualFn )
    PetscErrorCode DMPlexComputeIntegralFEM( PetscDM, PetscVec, PetscScalar*, void* )
    PetscErrorCode DMPlexComputeCellwiseIntegralFEM( PetscDM, PetscVec, PetscVec, void* )


class Integral:
    """
    The `Integral` class constructs the volume integral

    .. math:: F_{i}  =   \int_V \, f(\mathbf{x}) \, \mathrm{d} V

    for some scalar function :math:`f` over the mesh domain :math:`V`.

    Parameters
    ----------
    mesh : 
        The mesh over which integration is performed.
    fn : 
        Function to be integrated.

    Example
    -------
    Calculate volume of mesh:

    >>> import underworld3 as uw
    >>> import numpy as np
    >>> mesh = uw.mesh.Box()
    >>> volumeIntegral = uw.maths.Integral(mesh=mesh, fn=1.)
    >>> np.allclose( 1., volumeIntegral.evaluate(), rtol=1e-8)
    True
    """
    
    @timing.routine_timer_decorator
    def __init__( self,
                  mesh:  underworld3.mesh.Mesh,
                  fn:    Union[float, int, sympy.Basic], 
                  degree: int=1):

        self.mesh = mesh
        self.fn = sympy.sympify(fn)
        self.degree = degree
        super().__init__()

    @timing.routine_timer_decorator
    def evaluate(self) -> float:
        if len(self.mesh.vars)==0:
            raise RuntimeError("The mesh requires at least a single variable for integration to function correctly.\n"
                               "This is a PETSc limitation.")
                               
        # Create JIT extension.
        # Note that we pass in the mesh variables as primary variables, as this
        # is how they are represented on the mesh DM.

        # Note that (at this time) PETSc does not support vector integrands, so 
        # if we wish to do vector integrals we'll need to split out the components
        # and calculate them individually. Let's support only scalars for now.
        if isinstance(self.fn, sympy.vector.Vector):
            raise RuntimeError("Integral evaluation for Vector integrands not supported.")
        elif isinstance(self.fn, sympy.vector.Dyadic):
            raise RuntimeError("Integral evaluation for Dyadic integrands not supported.")

        cdef PtrContainer ext = getext(self.mesh, [self.fn,], [], [], self.mesh.vars.values())

        # Pull out vec for variables, and go ahead with the integral
        self.mesh.update_lvec()
        cdef Vec cgvec = self.mesh.dm.getGlobalVec()
        self.mesh.dm.localToGlobal(self.mesh.lvec, cgvec)

        cdef DM dmc = self.mesh.dm.clone()
        cdef PetscInt cdegree = self.degree
        cdef FE fec = FE().createDefault(self.mesh.dim, cdegree, False, -1)
        dmc.setField(0, fec)
        dmc.createDS()

        cdef DS ds = dmc.getDS()
        CHKERRQ( PetscDSSetObjective(ds.ds, 0, ext.fns_residual[0]) )
        
        cdef PetscScalar val
        CHKERRQ( DMPlexComputeIntegralFEM(dmc.dm, cgvec.vec, <PetscScalar*>&val, NULL) )
        self.mesh.dm.restoreGlobalVec(cgvec)

        # We're making an assumption here that PetscScalar is same as double.
        # Need to check where this may not be the case.
        cdef double vald = <double> val

        return vald


class CellWiseIntegral:
    """
    The `Integral` class constructs the cell wise volume integral

    .. math:: F_{i}  =   \int_V \, f(\mathbf{x}) \, \mathrm{d} V

    for some scalar function :math:`f` over the mesh domain :math:`V`.

    Parameters
    ----------
    mesh : 
        The mesh over which integration is performed.
    fn : 
        Function to be integrated.

    Example
    -------
    Calculate volume of mesh:

    >>> import underworld3 as uw
    >>> import numpy as np
    >>> mesh = uw.mesh.Box()
    >>> volumeIntegral = uw.maths.Integral(mesh=mesh, fn=1.)
    >>> np.allclose( 1., volumeIntegral.evaluate(), rtol=1e-8)
    True
    """
    
    @timing.routine_timer_decorator
    def __init__( self,
                  mesh:  underworld3.mesh.Mesh,
                  fn:    Union[float, int, sympy.Basic],
                  degree: int=1 ):

        self.mesh = mesh
        self.fn = sympy.sympify(fn)
        self.degree = degree
        super().__init__()

    @timing.routine_timer_decorator
    def evaluate(self) -> float:
        if len(self.mesh.vars)==0:
            raise RuntimeError("The mesh requires at least a single variable for integration to function correctly.\n"
                               "This is a PETSc limitation.")
                               
        # Create JIT extension.
        # Note that we pass in the mesh variables as primary variables, as this
        # is how they are represented on the mesh DM.

        # Note that (at this time) PETSc does not support vector integrands, so 
        # if we wish to do vector integrals we'll need to split out the components
        # and calculate them individually. Let's support only scalars for now.
        if isinstance(self.fn, sympy.vector.Vector):
            raise RuntimeError("Integral evaluation for Vector integrands not supported.")
        elif isinstance(self.fn, sympy.vector.Dyadic):
            raise RuntimeError("Integral evaluation for Dyadic integrands not supported.")

        cdef PtrContainer ext = getext(self.mesh, [self.fn,], [], [], self.mesh.vars.values())

        # Pull out vec for variables, and go ahead with the integral
        self.mesh.update_lvec()
        cdef Vec cgvec = self.mesh.dm.getGlobalVec()
        self.mesh.dm.localToGlobal(self.mesh.lvec, cgvec)
       
        cdef DM dmc = self.mesh.dm.clone()
        cdef PetscInt cdegree = self.degree
        cdef FE fec = FE().createDefault(self.mesh.dim, cdegree, False, -1)
        dmc.setField(0, fec)
        dmc.createDS()

        cdef DS ds = dmc.getDS()
        CHKERRQ( PetscDSSetObjective(ds.ds, 0, ext.fns_residual[0]) )
        
        cdef Vec rvec = dmc.getGlobalVec()
        CHKERRQ( DMPlexComputeCellwiseIntegralFEM(dmc.dm, cgvec.vec, rvec.vec, NULL) )
        self.mesh.dm.restoreGlobalVec(cgvec)

        results = rvec.array.copy()
        
        return results