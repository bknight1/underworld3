from petsc4py.PETSc cimport DM, PetscDM, DS, PetscDS, Vec, PetscVec, PetscQuadrature, FE, PetscFE
from .petsc_types cimport PetscInt, PetscScalar, PetscErrorCode, PetscDSResidualFn
from .petsc_types cimport PtrContainer
from ._jitextension import getext
from sympy import sympify
import sympy
from typing import Union
import underworld3

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")

cdef extern from "petsc.h" nogil:
    PetscErrorCode PetscDSSetObjective( PetscDS, PetscInt, PetscDSResidualFn )
    PetscErrorCode DMPlexComputeIntegralFEM( PetscDM, PetscVec, PetscScalar*, void* )
    PetscErrorCode PetscFEGetQuadrature(PetscFE fem, PetscQuadrature *q)
    PetscErrorCode PetscFESetQuadrature(PetscFE fem, PetscQuadrature q)


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
    >>> mesh = uw.mesh.Mesh()
    >>> volumeIntegral = uw.maths.Integral(mesh=mesh, fn=1.)
    >>> np.allclose( 1., volumeIntegral.evaluate(), rtol=1e-8)
    True

    """
    def __init__( self,
                  mesh:  underworld3.mesh.Mesh,
                  fn:    Union[float, int, sympy.Basic] ):

        self.mesh = mesh
        self.fn = sympify(fn)
        super().__init__()

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

        # Now, find var with the highest degree. We will then configure the integration 
        # to use this variable's quadrature object for all variables. 
        # This needs to be double checked.  
        deg = 0
        for key, var in self.mesh.vars.items():
            if var.degree >= deg:
                deg = var.degree
                var_base = var

        cdef FE c_fe = var_base.petsc_fe
        cdef PetscQuadrature quad_base
        ierr = PetscFEGetQuadrature(c_fe.fe, &quad_base); CHKERRQ(ierr)
        for fe in [var.petsc_fe for var in self.mesh.vars.values()]:
            c_fe = fe
            ierr = PetscFESetQuadrature(c_fe.fe,quad_base); CHKERRQ(ierr)        

        cdef DM dm = self.mesh.dm
        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()
        cdef DS ds = self.mesh.dm.getDS()
        # Now set callback... note that we set the highest degree var_id (as determined
        # above) for the second parameter. 
        ierr = PetscDSSetObjective(ds.ds, 0, ext.fns_residual[0]); CHKERRQ(ierr)
        
        cdef PetscScalar val
        cdef Vec clvec
        # Pull out vec for variables, and go ahead with the integral
        with self.mesh.getInterlacedLocalVariableVecManaged() as a_local:
            clvec = a_local
            ierr = DMPlexComputeIntegralFEM(dm.dm, clvec.vec, <PetscScalar*>&val, NULL); CHKERRQ(ierr)

        # We're making an assumption here that PetscScalar is same as double.
        # Need to check where this may not be the case.
        cdef double vald = <double> val

        return vald