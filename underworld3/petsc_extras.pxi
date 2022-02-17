from petsc4py.PETSc cimport DM,  PetscDM
from petsc4py.PETSc cimport DS,  PetscDS
from petsc4py.PETSc cimport Vec, PetscVec
from petsc4py.PETSc cimport Mat, PetscMat
from petsc4py.PETSc cimport IS,  PetscIS
from petsc4py.PETSc cimport FE,  PetscFE
from petsc4py.PETSc cimport PetscDMLabel
from petsc4py.PETSc cimport PetscQuadrature
from petsc4py.PETSc cimport MPI_Comm, PetscMat, GetCommDefault, PetscViewer

from underworld3.petsc_types cimport PetscBool, PetscInt, PetscReal, PetscScalar
from underworld3.petsc_types cimport PetscErrorCode 
from underworld3.petsc_types cimport DMBoundaryConditionType
from underworld3.petsc_types cimport PetscDSResidualFn, PetscDSJacobianFn
from underworld3.petsc_types cimport PtrContainer
from underworld3.petsc_gen_xdmf import generateXdmf

ctypedef enum PetscBool:
    PETSC_FALSE
    PETSC_TRUE

cdef CHKERRQ(PetscErrorCode ierr):
    cdef int interr = <int>ierr
    if ierr != 0: raise RuntimeError(f"PETSc error code '{interr}' was encountered.\nhttps://www.mcs.anl.gov/petsc/petsc-current/include/petscerror.h.html")

cdef extern from "petsc_compat.h":
    PetscErrorCode PetscDSAddBoundary_UW( PetscDM, DMBoundaryConditionType, const char[], const char[] , PetscInt, PetscInt, const PetscInt *,                                                      void (*)(), void (*)(), PetscInt, const PetscInt *, void *)

cdef extern from "petsc.h" nogil:
    PetscErrorCode DMPlexSNESComputeBoundaryFEM( PetscDM, void *, void *)
    PetscErrorCode DMPlexSetSNESLocalFEM( PetscDM, void *, void *, void *)
    PetscErrorCode DMCreateSubDM(PetscDM, PetscInt, const PetscInt *, PetscIS *, PetscDM *)
    PetscErrorCode DMDestroy(PetscDM *dm)
    PetscErrorCode DMPlexComputeGeometryFVM( PetscDM dm, PetscVec *cellgeom, PetscVec *facegeom)
    # PetscErrorCode DMPlexCreateBallMesh(MPI_Comm, PetscInt, PetscReal, PetscDM*)
    PetscErrorCode DMPlexExtrude(PetscDM idm, PetscInt layers, PetscReal height, PetscBool orderHeight, const PetscReal extNormal[], PetscBool interpolate, PetscDM* dm)
    PetscErrorCode DMPlexGetMinRadius(PetscDM dm, PetscReal *minradius)
    PetscErrorCode DMProjectCoordinates(PetscDM dm, PetscFE disc)
    PetscErrorCode DMSetAuxiliaryVec(PetscDM dm, PetscDMLabel label, PetscInt value, PetscVec aux)
    PetscErrorCode MatInterpolate(PetscMat A, PetscVec x, PetscVec y)
    PetscErrorCode PetscDSSetJacobian( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode PetscDSSetJacobianPreconditioner( PetscDS, PetscInt, PetscInt, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn, PetscDSJacobianFn)
    PetscErrorCode PetscDSSetResidual( PetscDS, PetscInt, PetscDSResidualFn, PetscDSResidualFn )
    # PetscErrorCode PetscViewerHDF5PushTimestepping(PetscViewer viewer)
    PetscErrorCode VecDestroy(PetscVec *v)
