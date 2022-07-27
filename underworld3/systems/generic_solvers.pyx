from xmlrpc.client import Boolean
import sympy
from sympy import sympify
from sympy.vector import gradient, divergence

from typing import Optional


from petsc4py import PETSc

import underworld3 
import underworld3 as uw
from .._jitextension import getext  #, diff_fn1_wrt_fn2
import underworld3.timing as timing

include "../petsc_extras.pxi"


class SNES_Scalar:

    instances = 0

    @timing.routine_timer_decorator
    def __init__(self, 
                 mesh     : uw.discretisation.Mesh, 
                 u_Field  : uw.discretisation.MeshVariable = None, 
                 degree     = 2,
                 solver_name: str = "",
                 verbose    = False):

        ## Keep track

        SNES_Scalar.instances += 1

        ## Todo: this is obviously not particularly robust

        if solver_name != "" and not solver_name.endswith("_"):
            self.petsc_options_prefix = solver_name+"_"
        else:
            self.petsc_options_prefix = solver_name

        

        self.petsc_options = PETSc.Options(self.petsc_options_prefix)

        # Here we can set some defaults for this set of KSP / SNES solvers
        self.petsc_options["snes_type"] = "newtonls"
        self.petsc_options["ksp_rtol"] = 1.0e-3
        self.petsc_options["ksp_monitor"] = None
        self.petsc_options["ksp_type"] = "fgmres"
        self.petsc_options["pre_type"] = "gamg"
        self.petsc_options["snes_converged_reason"] = None
        self.petsc_options["snes_monitor_short"] = None
        # self.petsc_options["snes_view"] = None
        self.petsc_options["snes_rtol"] = 1.0e-3

        ## Todo: some validity checking on the size / type of u_Field supplied
        if not u_Field:
            self._u = uw.discretisation.MeshVariable( mesh=mesh, num_components=1, name="Us{}".format(SNES_Scalar.instances),
                                            vtype=uw.VarType.SCALAR, degree=degree )
        else:
            self._u = u_Field

        self.mesh = mesh
        self._F0 = sympy.Matrix.zeros(1,1)
        self._F1 = sympy.Matrix.zeros(1,mesh.dim)

        ## sympy.Array 
        self._L = sympy.derive_by_array(self._u.sym, self.mesh.X).reshape(1,self.mesh.dim).tomatrix()

        self.bcs = []

        self.is_setup = False
        self.verbose = verbose

        # Build the DM / FE structures (should be done on remeshing)

        self._build_dm_and_mesh_discretisation()
        self._rebuild_after_mesh_update = self._build_dm_and_mesh_discretisation

        # Some other setup 

        self.mesh._equation_systems_register.append(self)

        super().__init__()


    @property
    def F0(self):
        return self._F0
    @F0.setter
    def F0(self, value):
        self.is_setup = False
        # should add test here to make sure this is conformal
        self._F0 = sympy.Matrix((value,))  # Make sure it is a scalar / 1x1 Matrix

    @property
    def F1(self):
        return self._F1
    @F1.setter
    def F1(self, value):
        self.is_setup = False
        # should add test here to make sure this is conformal
        self._F1 = self.mesh.vector.to_matrix(value)  # Make sure this is a vector
 

    def _build_dm_and_mesh_discretisation(self):

        degree = self._u.degree
        mesh = self.mesh

        self.dm = mesh.dm.clone()

        # Set quadrature to consistent value given by mesh quadrature
        # We force the rebuild which allows the solvers to ratchet up
        # the quadrature order when they are defined (but not reduce it)
        # This can occur if variables are defined by the solver itself

        self.mesh._align_quadratures(force=True)
        u_quad = self.mesh.petsc_fe.getQuadrature()

        # create private variables
        options = PETSc.Options()
        options.setValue("{}_private_petscspace_degree".format(self.petsc_options_prefix), degree) # for private variables
        self.petsc_fe_u = PETSc.FE().createDefault(mesh.dim, 1, mesh.isSimplex, degree,"{}_private_".format(self.petsc_options_prefix), PETSc.COMM_WORLD)
        self.petsc_fe_u.setQuadrature(u_quad)
        self.petsc_fe_u_id = self.dm.getNumFields()

        self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )

        self.is_setup = False

        return

    @property
    def u(self):
        return self._u


    @timing.routine_timer_decorator
    def add_dirichlet_bc(self, fn, boundaries, components=[0]):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries, 'dirichlet'))

    @timing.routine_timer_decorator
    def add_neumann_bc(self, fn, boundaries, components=[0]):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries, "neumann"))


    ## This function is the one we will typically over-ride to build specific solvers. 
    ## This example is a poisson-like problem with isotropic coefficients

    @timing.routine_timer_decorator
    def _setup_problem_description(self):

        # We might choose to modify the definition of F0  / F1 
        # by changing this function in a sub-class
        # For example, to simplify the user interface by pre-definining
        # the form of the input. See the projector class for an example 

        # f0 residual term (weighted integration) - scalar RHS function
        self._f0 = self.F0 # some_expression_F0(self._U, self._L)

        # f1 residual term (integration by parts / gradients)
        self._f1 = self.F1 # some_expresion_F1(self._U, self._L)

        return 

    # The properties that are used in the problem description
    # F0 is a scalar function (can include u, grad_u)
    # F1_i is a vector valued function (can include u, grad_u)

    # We don't add any validation here ... we should check that these
    # can be ingested by the _setup_terms() function

    @timing.routine_timer_decorator
    def _setup_terms(self):
        from sympy.vector import gradient
        import sympy

        N = self.mesh.N
        dim = self.mesh.dim

        ## The residual terms describe the problem and 
        ## can be changed by the user in inherited classes

        self._setup_problem_description()

        ## The jacobians are determined from the above (assuming we 
        ## do not concern ourselves with the zeros)

        F0 = sympy.Array(self._f0).reshape(1).as_immutable()
        F1 = sympy.Array(self._f1).reshape(dim).as_immutable()

        U = sympy.Array(self._u.sym).reshape(1).as_immutable() # scalar works better in derive_by_array
        L = sympy.Array(self._L).reshape(dim).as_immutable() # unpack one index here too

        fns_residual = [F0, F1] 

        G0 = sympy.derive_by_array(F0, U)
        G1 = sympy.derive_by_array(F0, L)
        G2 = sympy.derive_by_array(F1, U)
        G3 = sympy.derive_by_array(F1, L)

        # Re-organise if needed / make hashable
        
        self._G0 = sympy.ImmutableMatrix(G0)
        self._G1 = sympy.ImmutableMatrix(G1)
        self._G2 = sympy.ImmutableMatrix(G2)
        self._G3 = sympy.ImmutableMatrix(G3)

        ##################

        fns_jacobian = (self._G0, self._G1, self._G2, self._G3)

        ################## 

        # generate JIT code.
        # first, we must specify the primary fields.
        # these are fields for which the corresponding sympy functions 
        # should be replaced with the primary (instead of auxiliary) petsc 
        # field value arrays. in this instance, we want to switch out 
        # `self.u` and `self.p` for their primary field 
        # petsc equivalents. without specifying this list, 
        # the aux field equivalents will be used instead, which 
        # will give incorrect results for non-linear problems.
        # note also that the order here is important.

        prim_field_list = [self.u,]
        cdef PtrContainer ext = getext(self.mesh, tuple(fns_residual), tuple(fns_jacobian), [x[1] for x in self.bcs], primary_field_list=prim_field_list)

        # set functions 
        self.dm.createDS()
        cdef DS ds = self.dm.getDS()
        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[0], ext.fns_residual[1])
        # TODO: check if there's a significant performance overhead in passing in 
        # identically `zero` pointwise functions instead of setting to `NULL`
        PetscDSSetJacobian(ds.ds, 0, 0, ext.fns_jacobian[0], ext.fns_jacobian[1], ext.fns_jacobian[2], ext.fns_jacobian[3])
        
        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm

        for index,bc in enumerate(self.bcs):
            comps_view = bc.components
            for boundary in bc.boundaries:
                if self.verbose:
                    print("Setting bc {} ({})".format(index, bc.type))
                    print(" - components: {}".format(bc.components))
                    print(" - boundary:   {}".format(bc.boundaries))
                    print(" - fn:         {} ".format(bc.fn))

                # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
                # use type 6 bc for `DM_BC_NATURAL_FIELD` enum  (is this implemented for non-zero values ?)
                if bc.type == 'neumann':
                    bc_type = 6
                else:
                    bc_type = 5

                PetscDSAddBoundary_UW( cdm.dm, bc_type, str(boundary).encode('utf8'), str(boundary).encode('utf8'), 0, comps_view.shape[0], <const PetscInt *> &comps_view[0], <void (*)()>ext.fns_bcs[index], NULL, 1, <const PetscInt *> &ind, NULL)

        self.dm.setUp()

        self.dm.createClosureIndex(None)
        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setOptionsPrefix(self.petsc_options_prefix)
        self.snes.setFromOptions()
        cdef DM dm = self.dm
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)

        self.is_setup = True

    def viewDiscretisation(self):
        """
        Return a copy of the system discretisation. Nodes and the unknown u scalar field.

        Output
        ------
        (n, u): numpy vector of with dimensions (n, x) 
            n is the number the local mesh nodes.
            x is the size of dim plus the unknown values.
        """
        import numpy as np

        with self.mesh.access():
           return np.hstack( [self.u.coords.copy(), self.u.data.copy()] )

    @timing.routine_timer_decorator
    def solve(self, 
              zero_init_guess: bool =True, 
              _force_setup:    bool =False ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the 
            system solution. Otherwise, the current values of `self.u` 
            and `self.p` will be used.
        """
        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            with self.mesh.access():
                self.dm.localToGlobal(self.u.vec, gvec)
        else:
            gvec.array[:] = 0.

        # Set quadrature to consistent value given by mesh quadrature.
        self.mesh._align_quadratures()

        # Call `createDS()` on aux dm. This is necessary after the 
        # quadratures are set above, as it generates the tablatures 
        # from the quadratures (among other things no doubt). 
        # TODO: What does createDS do?
        # TODO: What are the implications of calling this every solve.

        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()

        cdef DM dm = self.dm

        self.mesh.update_lvec()
        cdef Vec cmesh_lvec
        # PETSc == 3.16 introduced an explicit interface 
        # for setting the aux-vector which we'll use when available.
        cmesh_lvec = self.mesh.lvec
        ierr = DMSetAuxiliaryVec_UW(dm.dm, NULL, 0, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        # solve
        self.snes.solve(None,gvec)

        lvec = self.dm.getLocalVec()
        cdef Vec clvec = lvec
        # Copy solution back into user facing variable
        with self.mesh.access(self.u,):
            self.dm.globalToLocal(gvec, lvec)
            # add back boundaries.
            # Note that `DMPlexSNESComputeBoundaryFEM()` seems to need to use an lvec
            # derived from the system-dm (as opposed to the var.vec local vector), else 
            # failures can occur. 
            ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
            self.u.vec.array[:] = lvec.array[:]

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)


### =================================

# LM: this is probably not something we need ... The petsc interface is 
# general enough to have one class to handle Vector and Scalar

class SNES_Vector:

    instances = 0

    @timing.routine_timer_decorator
    def __init__(self, 
                 mesh     : uw.discretisation.Mesh, 
                 u_Field  : uw.discretisation.MeshVariable = None, 
                 degree     = 2,
                 solver_name: str = "",
                 verbose    = False):

        ## Keep track

        SNES_Vector.instances += 1

        ## Todo: this is obviously not particularly robust

        if solver_name != "" and not solver_name.endswith("_"):
            self.petsc_options_prefix = solver_name+"_"
        else:
            self.petsc_options_prefix = solver_name

        self.petsc_options = PETSc.Options(self.petsc_options_prefix)

        # Here we can set some defaults for this set of KSP / SNES solvers
        self.petsc_options["snes_type"] = "newtonls"
        self.petsc_options["ksp_rtol"] = 1.0e-3
        self.petsc_options["ksp_monitor"] = None
        self.petsc_options["ksp_type"] = "fgmres"
        self.petsc_options["pre_type"] = "gamg"
        self.petsc_options["snes_converged_reason"] = None
        self.petsc_options["snes_monitor_short"] = None
        # self.petsc_options["snes_view"] = None
        self.petsc_options["snes_rtol"] = 1.0e-3

        ## Todo: some validity checking on the size / type of u_Field supplied
        if not u_Field:
            self._u = uw.discretisation.MeshVariable( mesh=mesh, num_components=mesh.dim, name="Uv{}".format(SNES_Scalar.instances),
                                            vtype=uw.VarType.SCALAR, degree=degree )
        else:
            self._u = u_Field

        self.mesh = mesh
        self._F0 = sympy.Matrix.zeros(1, self.mesh.dim)
        self._F1 = sympy.Matrix.zeros(self.mesh.dim, self.mesh.dim)

        ## sympy.Matrix

        self._U = self._u.sym
        self._L = self._u.sym.jacobian(self.mesh.X) # This works for vector / vector inputs

        self.bcs = []

        self.is_setup = False
        self.verbose = verbose

        # Build the DM / FE structures (should be done on remeshing)

        self._build_dm_and_mesh_discretisation()
        self._rebuild_after_mesh_update = self._build_dm_and_mesh_discretisation

        # Some other setup 

        self.mesh._equation_systems_register.append(self)

        super().__init__()

    @property
    def F0(self):
        return self._F0
    @F0.setter
    def F0(self, value):
        self.is_setup = False
        # should add test here to make sure k is conformal
        self._F0 = sympify(value)

    @property
    def F1(self):
        return self._F1
    @F1.setter
    def F1(self, value):
        self.is_setup = False
        # should add test here to make sure k is conformal
        self._F1 = sympify(value)
 

    def _build_dm_and_mesh_discretisation(self):

        degree = self._u.degree
        mesh = self.mesh

        self.dm = mesh.dm.clone()

        # Set quadrature to consistent value given by mesh quadrature.
        self.mesh._align_quadratures(force=True)
        u_quad = self.mesh.petsc_fe.getQuadrature()

        # create private variables
        options = PETSc.Options()
        options.setValue("{}_private_petscspace_degree".format(self.petsc_options_prefix), degree) # for private variables

        self.petsc_fe_u = PETSc.FE().createDefault(mesh.dim, mesh.dim, mesh.isSimplex, degree,"{}_private_".format(self.petsc_options_prefix), PETSc.COMM_WORLD)
        self.petsc_fe_u.setQuadrature(u_quad)
        self.petsc_fe_u_id = self.dm.getNumFields()
        self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )

        self.is_setup = False

        return

    @property
    def u(self):
        return self._u


    @timing.routine_timer_decorator
    def add_dirichlet_bc(self, fn, boundaries, components=[0]):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries, 'dirichlet'))

    @timing.routine_timer_decorator
    def add_neumann_bc(self, fn, boundaries, components=[0]):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries, "neumann"))


    ## This function is the one we will typically over-ride to build specific solvers. 
    ## This example is a poisson-like problem with isotropic coefficients

    @timing.routine_timer_decorator
    def _setup_problem_description(self):

        # We might choose to modify the definition of F0  / F1 
        # by changing this function in a sub-class
        # For example, to simplify the user interface by pre-definining
        # the form of the input. See the projector class for an example 

        # f0 residual term (weighted integration) - vector RHS function
        self._f0 = self.F0 # some_expression_F0(self._U, self._L)

        # f1 residual term (integration by parts / gradients)
        self._f1 = self.F1 # some_expresion_F1(self._U, self._L)

        return 

    # The properties that are used in the problem description
    # F0 is a vector function (can include u, grad_u)
    # F1_i is a vector valued function (can include u, grad_u)

    # We don't add any validation here ... we should check that these
    # can be ingested by the _setup_terms() function

    @timing.routine_timer_decorator
    def _setup_terms(self):
        from sympy.vector import gradient
        import sympy

        N = self.mesh.N
        dim = self.mesh.dim

        ## The residual terms describe the problem and 
        ## can be changed by the user in inherited classes

        self._setup_problem_description()

        ## The jacobians are determined from the above (assuming we 
        ## do not concern ourselves with the zeros)
        ## Convert to arrays for the moment to allow 1D arrays (size dim, not 1xdim)
        ## otherwise we have many size-1 indices that we have to collapse

        F0 = sympy.Array(self.mesh.vector.to_matrix(self._f0)).reshape(dim)
        F1 = sympy.Array(self._f1).reshape(dim,dim)

        # JIT compilation needs immutable, matrix input (not arrays)
        u_F0 = sympy.ImmutableDenseMatrix(F0)
        u_F1 = sympy.ImmutableDenseMatrix(F1)
        fns_residual = [u_F0, u_F1] 

        # This is needed to eliminate extra dims in the tensor
        U = sympy.Array(self._U).reshape(dim)

        G0 = sympy.derive_by_array(F0, U)
        G1 = sympy.derive_by_array(F0, self._L)
        G2 = sympy.derive_by_array(F1, U)
        G3 = sympy.derive_by_array(F1, self._L)

        '''
        self._0F0 = F0
        self._0F1 = F1

        self._u_F0 = u_F0
        self._u_F1 = u_F1

        self._GG0 = G0
        self._GG1 = G1
        self._GG2 = G2
        self._GG3 = G3
        '''

        # reorganise indices from sympy to petsc ordering 
        # reshape to Matrix form
        # Make hashable (immutable)

        self._G0 = sympy.ImmutableMatrix(G0.reshape(dim,dim))
        self._G1 = sympy.ImmutableMatrix(sympy.permutedims(G1, (2,1,0)  ).reshape(dim,dim*dim))
        self._G2 = sympy.ImmutableMatrix(sympy.permutedims(G2, (2,1,0)  ).reshape(dim*dim,dim))
        self._G3 = sympy.ImmutableMatrix(sympy.permutedims(G3, (3,1,2,0)).reshape(dim*dim,dim*dim))

        ##################

        fns_jacobian = (self._G0, self._G1, self._G2, self._G3)

        ################## 

        # generate JIT code.
        # first, we must specify the primary fields.
        # these are fields for which the corresponding sympy functions 
        # should be replaced with the primary (instead of auxiliary) petsc 
        # field value arrays. in this instance, we want to switch out 
        # `self.u` and `self.p` for their primary field 
        # petsc equivalents. without specifying this list, 
        # the aux field equivalents will be used instead, which 
        # will give incorrect results for non-linear problems.
        # note also that the order here is important.

        prim_field_list = [self.u,]
        cdef PtrContainer ext = getext(self.mesh, tuple(fns_residual), tuple(fns_jacobian), [x[1] for x in self.bcs], primary_field_list=prim_field_list)

        # set functions 
        self.dm.createDS()
        cdef DS ds = self.dm.getDS()
        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[0], ext.fns_residual[1])
    
        # TODO: check if there's a significant performance overhead in passing in 
        # identically `zero` pointwise functions instead of setting to `NULL`
        PetscDSSetJacobian(ds.ds, 0, 0, ext.fns_jacobian[0], ext.fns_jacobian[1], ext.fns_jacobian[2], ext.fns_jacobian[3])
        
        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm

        for index,bc in enumerate(self.bcs):
            comps_view = bc.components
            for boundary in bc.boundaries:
                if self.verbose:
                    print("Setting bc {} ({})".format(index, bc.type))
                    print(" - components: {}".format(bc.components))
                    print(" - boundary:   {}".format(bc.boundaries))
                    print(" - fn:         {} ".format(bc.fn))

                # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
                # use type 6 bc for `DM_BC_NATURAL_FIELD` enum  (is this implemented for non-zero values ?)
                if bc.type == 'neumann':
                    bc_type = 6
                else:
                    bc_type = 5

                PetscDSAddBoundary_UW( cdm.dm, bc_type, str(boundary).encode('utf8'), str(boundary).encode('utf8'), 0, comps_view.shape[0], <const PetscInt *> &comps_view[0], <void (*)()>ext.fns_bcs[index], NULL, 1, <const PetscInt *> &ind, NULL)

        self.dm.setUp()

        self.dm.createClosureIndex(None)
        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setOptionsPrefix(self.petsc_options_prefix)
        self.snes.setFromOptions()
        cdef DM dm = self.dm
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)

        self.is_setup = True

    @timing.routine_timer_decorator
    def solve(self, 
              zero_init_guess: bool =True, 
              _force_setup:    bool =False ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the 
            system solution. Otherwise, the current values of `self.u` 
            and `self.p` will be used.
        """
        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            with self.mesh.access():
                self.dm.localToGlobal(self.u.vec, gvec)
        else:
            gvec.array[:] = 0.

        # Set quadrature to consistent value given by mesh quadrature.
        self.mesh._align_quadratures()

        # Call `createDS()` on aux dm. This is necessary after the 
        # quadratures are set above, as it generates the tablatures 
        # from the quadratures (among other things no doubt). 
        # TODO: What does createDS do?
        # TODO: What are the implications of calling this every solve.

        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()

        self.mesh.update_lvec()
        cdef DM dm = self.dm
        cdef Vec cmesh_lvec
        # PETSc == 3.16 introduced an explicit interface 
        # for setting the aux-vector which we'll use when available.
        cmesh_lvec = self.mesh.lvec
        ierr = DMSetAuxiliaryVec_UW(dm.dm, NULL, 0, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        # solve
        self.snes.solve(None,gvec)

        lvec = self.dm.getLocalVec()
        cdef Vec clvec = lvec
        # Copy solution back into user facing variable
        with self.mesh.access(self.u):
            self.dm.globalToLocal(gvec, lvec)
            # add back boundaries.
            # Note that `DMPlexSNESComputeBoundaryFEM()` seems to need to use an lvec
            # derived from the system-dm (as opposed to the var.vec local vector), else 
            # failures can occur. 
            ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
            self.u.vec.array[:] = lvec.array[:]

        self.dm.restoreLocalVec(lvec)
        self.dm.restoreGlobalVec(gvec)


### =================================

class SNES_SaddlePoint:

    instances = 0   # count how many of these there are in order to create unique private mesh variable ids

    @timing.routine_timer_decorator
    def __init__(self, 
                 mesh          : underworld3.discretisation.Mesh, 
                 velocityField : Optional[underworld3.discretisation.MeshVariable] =None,
                 pressureField : Optional[underworld3.discretisation.MeshVariable] =None,
                 u_degree      : Optional[int]                           =2, 
                 p_degree      : Optional[int]                           =None,
                 p_continuous  : Optional[bool]                          =True,
                 solver_name   : Optional[str]                           ="stokes_",
                 verbose       : Optional[str]                           =False,
                 _Ppre_fn      = None
                  ):
        """
        This class provides functionality for a discrete representation
        of the Stokes flow equations.

        Specifically, the class uses a mixed finite element implementation to
        construct a system of linear equations which may then be solved.

        The strong form of the given boundary value problem, for :math:`f`,
        :math:`g` and :math:`h` given, is

        .. math::
            \\begin{align}
            \\sigma_{ij,j} + f_i =& \\: 0  & \\text{ in }  \\Omega \\\\
            u_{k,k} =& \\: 0  & \\text{ in }  \\Omega \\\\
            u_i =& \\: g_i & \\text{ on }  \\Gamma_{g_i} \\\\
            \\sigma_{ij}n_j =& \\: h_i & \\text{ on }  \\Gamma_{h_i} \\\\
            \\end{align}

        where,

        * :math:`\\sigma_{i,j}` is the stress tensor
        * :math:`u_i` is the velocity,
        * :math:`p`   is the pressure,
        * :math:`f_i` is a body force,
        * :math:`g_i` are the velocity boundary conditions (DirichletCondition)
        * :math:`h_i` are the traction boundary conditions (NeumannCondition).

        The problem boundary, :math:`\\Gamma`,
        admits the decompositions :math:`\\Gamma=\\Gamma_{g_i}\\cup\\Gamma_{h_i}` where
        :math:`\\emptyset=\\Gamma_{g_i}\\cap\\Gamma_{h_i}`. The equivalent weak form is:

        .. math::
            \\int_{\Omega} w_{(i,j)} \\sigma_{ij} \\, d \\Omega = \\int_{\\Omega} w_i \\, f_i \\, d\\Omega + \sum_{j=1}^{n_{sd}} \\int_{\\Gamma_{h_j}} w_i \\, h_i \\,  d \\Gamma

        where we must find :math:`u` which satisfies the above for all :math:`w`
        in some variational space.

        Parameters
        ----------
        mesh : 
            The mesh object which forms the basis for problem discretisation,
            domain specification, and parallel decomposition.
        velocityField :
            Optional. Variable used to record system velocity. If not provided,
            it will be generated and will be available via the `u` stokes object property.
        pressureField :
            Optional. Variable used to record system pressure. If not provided,
            it will be generated and will be available via the `p` stokes object property.
            If provided, it is up to the user to ensure that it is of appropriate order
            relative to the provided velocity variable (usually one order lower degree).
        u_degree :
            Optional. The polynomial degree for the velocity field elements.
        p_degree :
            Optional. The polynomial degree for the pressure field elements. 
            If provided, it is up to the user to ensure that it is of appropriate order
            relative to the provided velocitxy variable (usually one order lower degree).
            If not provided, it will be set to one order lower degree than the velocity field.
        solver_name :
            Optional. The petsc options prefix for the SNES solve. This is important to provide
            a name space when multiples solvers are constructed that may have different SNES options.
            For example, if you name the solver "stokes", the SNES options such as `snes_rtol` become `stokes_snes_rtol`.
            The default is blank, and an underscore will be added to the end of the solver name if not already present.
 
        Notes
        -----
        Constructor must be called by collectively all processes.

        """       

        SNES_SaddlePoint.instances += 1

        self.mesh = mesh
        self.verbose = verbose
        self.p_continous = p_continuous

        if (velocityField is None) ^ (pressureField is None):
            raise ValueError("You must provided *both* `pressureField` and `velocityField`, or neither, but not one or the other.")
        
        # I expect the following to break for anyone who wants to name their solver _stokes__ etc etc (LM)

        if solver_name != "" and not solver_name.endswith("_"):
            self.petsc_options_prefix = solver_name+"_"
        else:
            self.petsc_options_prefix = solver_name

        self.petsc_options = PETSc.Options(self.petsc_options_prefix)

        # Here we can set some defaults for this set of KSP / SNES solvers
        # self.petsc_options["snes_type"] = "newtonls"
        self.petsc_options["ksp_rtol"] = 1.0e-3
        self.petsc_options["ksp_monitor"] = None
        # self.petsc_options["ksp_type"] = "fgmres"
        # self.petsc_options["pre_type"] = "gamg"
        self.petsc_options["snes_converged_reason"] = None
        self.petsc_options["snes_monitor_short"] = None
        # self.petsc_options["snes_view"] = None
        self.petsc_options["snes_rtol"] = 1.0e-3
        self.petsc_options["pc_type"] = "fieldsplit"
        self.petsc_options["pc_fieldsplit_type"] = "schur"
        self.petsc_options["pc_fieldsplit_schur_factorization_type"] = "full"
        self.petsc_options["pc_fieldsplit_schur_precondition"] = "a11"
        self.petsc_options["pc_fieldsplit_off_diag_use_amat"] = None    # These two seem to be needed in petsc 3.17
        self.petsc_options["pc_use_amat"] = None                        # These two seem to be needed in petsc 3.17
        self.petsc_options["fieldsplit_velocity_ksp_type"] = "fgmres"
        self.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-4
        self.petsc_options["fieldsplit_velocity_pc_type"]  = "gamg"
        self.petsc_options["fieldsplit_pressure_ksp_rtol"] = 3.e-4
        self.petsc_options["fieldsplit_pressure_pc_type"] = "gamg" 


        if not velocityField:
            if p_degree==None:
                p_degree = u_degree - 1

            # create public velocity/pressure variables
            self._u = uw.discretisation.MeshVariable( mesh=mesh, num_components=mesh.dim, name="usp_{}".format(self.instances), vtype=uw.VarType.VECTOR, degree=u_degree )
            self._p = uw.discretisation.MeshVariable( mesh=mesh, num_components=1,        name="psp_{}".format(self.instances), vtype=uw.VarType.SCALAR, degree=p_degree )
        else:
            self._u = velocityField
            self._p = pressureField

        # Create this dict
        self.fields = {}
        self.fields["pressure"] = self.p
        self.fields["velocity"] = self.u

        # Some other setup 

        self.mesh._equation_systems_register.append(self)

        # Build the DM / FE structures (should be done on remeshing, which is usually handled by the mesh register above)

        self._build_dm_and_mesh_discretisation()
        self._rebuild_after_mesh_update = self._build_dm_and_mesh_discretisation

        self.UF0 = sympy.Matrix.zeros(1, self.mesh.dim) 
        self.UF1 = sympy.Matrix.zeros(self.mesh.dim, self.mesh.dim)
        self.PF0 = 0.0

        self._Ppre_fn = _Ppre_fn

        self.bcs = []

        # Construct strainrate tensor for future usage.
        # Grab gradients, and let's switch out to sympy.Matrix notation
        # immediately as it is probably cleaner for this.
        N = mesh.N

        ## sympy.Array 
        self._U = sympy.Array((self._u.fn.to_matrix(self.mesh.N))[0:self.mesh.dim])
        self._P = sympy.Array([self._p.fn])
        self._X = sympy.Array(self.mesh.r)
        self._L = sympy.derive_by_array(self._U, self._X).transpose()
        self._G = sympy.derive_by_array(self._P, self._X)

        # this attrib records if we need to re-setup
        self.is_setup = False
        super().__init__()

    def _build_dm_and_mesh_discretisation(self):

        """
        Most of what is in the init phase that is not called by _setup_terms()

        """
        
        mesh = self.mesh
        u_degree = self.u.degree
        p_degree = self.p.degree

        self.dm   = mesh.dm.clone()

        # In the following, I'm not sure if we should make this self.instances or go all the way up to the SNES_SADDLE class 
        # to make sure there is no namespace clash ... I haven't seen any manifestations of these clashing but
        # I wonder if that is only because we use the same degree variables most of the time in one problem ... LM

        # Set quadrature to consistent value given by mesh quadrature.
        self.mesh._align_quadratures(force=True)
        quad = self.mesh.petsc_fe.getQuadrature()
            
        options = PETSc.Options()
        options.setValue("{}_uprivate_petscspace_degree".format(self.petsc_options_prefix), u_degree) # for private variables
        self.petsc_fe_u = PETSc.FE().createDefault(mesh.dim, mesh.dim, mesh.isSimplex, u_degree, "{}_uprivate_".format(self.petsc_options_prefix), PETSc.COMM_WORLD)
        self.petsc_fe_u.setQuadrature(quad)
        self.petsc_fe_u.setName("velocity")
        self.petsc_fe_u_id = self.dm.getNumFields()  ## can we avoid re-numbering ?
        self.dm.setField( self.petsc_fe_u_id, self.petsc_fe_u )

        options.setValue("{}_pprivate_petscspace_degree".format(self.petsc_options_prefix), p_degree)
        options.setValue("{}_pprivate_petscdualspace_lagrange_continuity".format(self.petsc_options_prefix), self.p_continous)
        self.petsc_fe_p = PETSc.FE().createDefault(mesh.dim,        1, mesh.isSimplex, u_degree, "{}_pprivate_".format(self.petsc_options_prefix), PETSc.COMM_WORLD)
        self.petsc_fe_p.setQuadrature(quad)
        self.petsc_fe_p.setName("pressure")
        self.petsc_fe_p_id = self.dm.getNumFields()
        self.dm.setField( self.petsc_fe_p_id, self.petsc_fe_p)
        self.is_setup = False

        return

    @property
    def UF0(self):
        return self._UF0
    @UF0.setter
    def UF0(self, value):
        self.is_setup = False
        # should add test here to make sure k is conformal
        self._UF0 = value

    @property
    def UF1(self):
        return self._UF1
    @UF1.setter
    def UF1(self, value):
        self.is_setup = False
        # should add test here to make sure k is conformal
        self._UF1 = value

    @property
    def PF0(self):
        return self._PF0
    @PF0.setter
    def PF0(self, value):
        self.is_setup = False
        # should add test here to make sure k is conformal
        self._PF0 = value

    @property
    def u(self):
        return self._u
    @property
    def p(self):
        return self._p

    @timing.routine_timer_decorator
    def add_dirichlet_bc(self, fn, boundaries, components):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries,'dirichlet'))

    @timing.routine_timer_decorator
    def add_neumann_bc(self, fn, boundaries, components):
        # switch to numpy arrays
        # ndmin arg forces an array to be generated even
        # where comps/indices is a single value.
        self.is_setup = False
        import numpy as np
        components = np.array(components, dtype=np.int32, ndmin=1)
        boundaries = np.array(boundaries, dtype=object,   ndmin=1)
        from collections import namedtuple
        BC = namedtuple('BC', ['components', 'fn', 'boundaries', 'type'])
        self.bcs.append(BC(components,sympify(fn),boundaries,'neumann'))


    @timing.routine_timer_decorator
    def _setup_problem_description(self):

        # residual terms can be redefined by
        # writing your own version of this method

        # terms that become part of the weighted integral
        self._u_f0 = self.UF0  # some_expression_u_f0(_V,_P. _L, _G)

        # Integration by parts into the stiffness matrix
        self._u_f1 = self.UF1  # some_expression_u_f1(_V,_P, _L, _G)

        # rhs in the constraint (pressure) equations
        self._p_f0 = self.PF0  # some_expression_p_f0(_V,_P, _L, _G)

        return 

    @timing.routine_timer_decorator
    def _setup_terms(self):
        dim = self.mesh.dim
        N = self.mesh.N
        
        # residual terms
        self._setup_problem_description()

        # Array form to work well with what is below
        # The basis functions are 3-vectors by default, even for 2D meshes, soooo ...
        F0  = sympy.Array(self.mesh.vector.to_matrix(self._u_f0)).reshape(dim)
        F1  = sympy.Array(self._u_f1).reshape(dim,dim)
        FP0 = sympy.Array(self._p_f0).reshape(1)

        # JIT compilation needs immutable, matrix input (not arrays)
        u_F0 = sympy.ImmutableDenseMatrix(F0)
        u_F1 = sympy.ImmutableDenseMatrix(F1)
        p_F0 = sympy.ImmutableDenseMatrix(FP0)

        fns_residual = [u_F0, u_F1, p_F0] 

        ## jacobian terms

        fns_jacobian = []

        ## Alternative ... using sympy ARRAY which should generalize well
        ## but has some issues with the change in ordering in petsc v. sympy.
        ## so we will leave both here to compare across a range of problems.

        # This is needed to eliminate extra dims in the tensor
        U = sympy.Array(self._U).reshape(dim)

        G0 = sympy.derive_by_array(F0, U)
        G1 = sympy.derive_by_array(F0, self._L)
        G2 = sympy.derive_by_array(F1, U)
        G3 = sympy.derive_by_array(F1, self._L)

        # reorganise indices from sympy to petsc ordering / reshape to Matrix form

        self._uu_G0 = sympy.ImmutableMatrix(G0)
        self._uu_G1 = sympy.ImmutableMatrix(sympy.permutedims(G1, (2,1,0)  ).reshape(dim,dim*dim))
        self._uu_G2 = sympy.ImmutableMatrix(sympy.permutedims(G2, (2,1,0)  ).reshape(dim*dim,dim))
        self._uu_G3 = sympy.ImmutableMatrix(sympy.permutedims(G3, (3,1,2,0)).reshape(dim*dim,dim*dim))

        fns_jacobian += [self._uu_G0, self._uu_G1, self._uu_G2, self._uu_G3]

        # U/P block (check permutations - they don't seem to be necessary here)

        self._up_G0 = sympy.ImmutableMatrix(sympy.derive_by_array(F0, self._P).reshape(dim))
        self._up_G1 = sympy.ImmutableMatrix(sympy.derive_by_array(F0, self._G).reshape(dim,dim))
        self._up_G2 = sympy.ImmutableMatrix(sympy.derive_by_array(F1, self._P).reshape(dim,dim))
        self._up_G3 = sympy.ImmutableMatrix(sympy.derive_by_array(F1, self._G).reshape(dim*dim,dim))

        fns_jacobian += [self._up_G0, self._up_G1, self._up_G2, self._up_G3]

        # P/U block (check permutations - they don't seem to be necessary here)

        self._pu_G0 = sympy.ImmutableMatrix(sympy.derive_by_array(FP0, self._U).reshape(dim))
        self._pu_G1 = sympy.ImmutableMatrix(sympy.derive_by_array(FP0, self._L).reshape(dim,dim))
        # self._pu_G2 = sympy.ImmutableMatrix(sympy.derive_by_array(FP1, self._P).reshape(dim,dim))
        # self._pu_G3 = sympy.ImmutableMatrix(sympy.derive_by_array(FP1, self._G).reshape(dim,dim*2))

        # fns_jacobian += [self._pu_G0, self._pu_G1, self._pu_G2, self._pu_G3]
        fns_jacobian += [self._pu_G0, self._pu_G1]
        # fns_jacobian.append(self._pu_G1)

        ## PP block is a preconditioner term, not auto-constructed

        if self._Ppre_fn is None:
            self._pp_G0 = 1/(self.viscosity)
        else:
            self._pp_G0 = self._Ppre_fn

        fns_jacobian.append(self._pp_G0)

        # generate JIT code.
        # first, we must specify the primary fields.
        # these are fields for which the corresponding sympy functions 
        # should be replaced with the primary (instead of auxiliary) petsc 
        # field value arrays. in this instance, we want to switch out 
        # `self.u` and `self.p` for their primary field 
        # petsc equivalents. without specifying this list, 
        # the aux field equivalents will be used instead, which 
        # will give incorrect results for non-linear problems.
        # note also that the order here is important.

        prim_field_list = [self.u, self.p]
        cdef PtrContainer ext = getext(self.mesh, tuple(fns_residual), tuple(fns_jacobian), [x[1] for x in self.bcs], primary_field_list=prim_field_list)
        # create indexes so that we don't rely on indices that can change
        i_res = {}
        for index,fn in enumerate(fns_residual):
            i_res[fn] = index
        i_jac = {}
        for index,fn in enumerate(fns_jacobian):
            i_jac[fn] = index

        # set functions 
        self.dm.createDS()
        cdef DS ds = self.dm.getDS()
        PetscDSSetResidual(ds.ds, 0, ext.fns_residual[i_res[u_F0]], ext.fns_residual[i_res[u_F1]])
        PetscDSSetResidual(ds.ds, 1, ext.fns_residual[i_res[p_F0]],                          NULL)
        # TODO: check if there's a significant performance overhead in passing in 
        # identically `zero` pointwise functions instead of setting to `NULL`
        PetscDSSetJacobian(              ds.ds, 0, 0, ext.fns_jacobian[i_jac[self._uu_G0]], ext.fns_jacobian[i_jac[self._uu_G1]], ext.fns_jacobian[i_jac[self._uu_G2]], ext.fns_jacobian[i_jac[self._uu_G3]])
        PetscDSSetJacobian(              ds.ds, 0, 1,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._up_G2]], ext.fns_jacobian[i_jac[self._up_G3]])
        PetscDSSetJacobian(              ds.ds, 1, 0,                                 NULL, ext.fns_jacobian[i_jac[self._pu_G1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 0, ext.fns_jacobian[i_jac[self._uu_G0]], ext.fns_jacobian[i_jac[self._uu_G1]], ext.fns_jacobian[i_jac[self._uu_G2]], ext.fns_jacobian[i_jac[self._uu_G3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 0, 1,                                 NULL,                                 NULL, ext.fns_jacobian[i_jac[self._up_G2]], ext.fns_jacobian[i_jac[self._up_G3]])
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 0,                                 NULL, ext.fns_jacobian[i_jac[self._pu_G1]],                                 NULL,                                 NULL)
        PetscDSSetJacobianPreconditioner(ds.ds, 1, 1, ext.fns_jacobian[i_jac[self._pp_G0]],                                 NULL,                                 NULL,                                 NULL)

        cdef int ind=1
        cdef int [::1] comps_view  # for numpy memory view
        cdef DM cdm = self.dm

        for index,bc in enumerate(self.bcs):
            comps_view = bc.components
            for boundary in bc.boundaries:
                if self.verbose:
                    print("Setting bc {} ({})".format(index, bc.type))
                    print(" - components: {}".format(bc.components))
                    print(" - boundary:   {}".format(bc.boundaries))
                    print(" - fn:         {} ".format(bc.fn))
                # use type 5 bc for `DM_BC_ESSENTIAL_FIELD` enum
                # use type 6 bc for `DM_BC_NATURAL_FIELD` enum  (is this implemented for non-zero values ?)
                if bc.type == 'neumann':
                    bc_type = 6
                else:
                    bc_type = 5

                PetscDSAddBoundary_UW(cdm.dm, bc_type, str(boundary).encode('utf8'), str(boundary).encode('utf8'), 0, comps_view.shape[0], <const PetscInt *> &comps_view[0], <void (*)()>ext.fns_bcs[index], NULL, 1, <const PetscInt *> &ind, NULL)  
        
        self.dm.setUp()

        self.dm.createClosureIndex(None)
        self.snes = PETSc.SNES().create(PETSc.COMM_WORLD)
        self.snes.setDM(self.dm)
        self.snes.setOptionsPrefix(self.petsc_options_prefix)
        self.snes.setFromOptions()
        cdef DM dm = self.dm
        DMPlexSetSNESLocalFEM(dm.dm, NULL, NULL, NULL)

        # Setup subdms here too.
        # These will be used to copy back/forth SNES solutions
        # into user facing variables.
        
        names, isets, dms = self.dm.createFieldDecomposition()
        self._subdict = {}
        for index,name in enumerate(names):
            self._subdict[name] = (isets[index],dms[index])

        self.is_setup = True

    @timing.routine_timer_decorator
    def solve(self, 
              zero_init_guess: bool =True, 
              _force_setup:    bool =False ):
        """
        Generates solution to constructed system.

        Params
        ------
        zero_init_guess:
            If `True`, a zero initial guess will be used for the 
            system solution. Otherwise, the current values of `self.u` 
            and `self.p` will be used.
        """
        if (not self.is_setup) or _force_setup:
            self._setup_terms()

        gvec = self.dm.getGlobalVec()

        if not zero_init_guess:
            with self.mesh.access():
                for name,var in self.fields.items():
                    sgvec = gvec.getSubVector(self._subdict[name][0])  # Get global subvec off solution gvec.
                    sdm   = self._subdict[name][1]                     # Get subdm corresponding to field
                    sdm.localToGlobal(var.vec,sgvec)                   # Copy variable data into gvec
        else:
            gvec.array[:] = 0.


        # Set quadrature to consistent value given by mesh quadrature.
        self.mesh._align_quadratures()

        # Call `createDS()` on aux dm. This is necessary after the 
        # quadratures are set above, as it generates the tablatures 
        # from the quadratures (among other things no doubt). 
        # TODO: What does createDS do?
        # TODO: What are the implications of calling this every solve.

        self.mesh.dm.clearDS()
        self.mesh.dm.createDS()

        self.mesh.update_lvec()
        cdef DM dm = self.dm
        cdef Vec cmesh_lvec
        # PETSc == 3.16 introduced an explicit interface 
        # for setting the aux-vector which we'll use when available.
        cmesh_lvec = self.mesh.lvec
        ierr = DMSetAuxiliaryVec_UW(dm.dm, NULL, 0, 0, cmesh_lvec.vec); CHKERRQ(ierr)

        # solve
        self.snes.solve(None, gvec)

        cdef Vec clvec
        cdef DM csdm
        # Copy solution back into user facing variables
        with self.mesh.access(self.p, self.u):
            for name,var in self.fields.items():
                ## print("Copy field {} to user variables".format(name), flush=True)
                sgvec = gvec.getSubVector(self._subdict[name][0])  # Get global subvec off solution gvec.
                sdm   = self._subdict[name][1]                     # Get subdm corresponding to field.
                lvec = sdm.getLocalVec()                           # Get a local vector to push data into.
                sdm.globalToLocal(sgvec,lvec)                      # Do global to local into lvec
                # Put in boundaries values.
                # Note that `DMPlexSNESComputeBoundaryFEM()` seems to need to use an lvec
                # derived from the sub-dm (as opposed to the var.vec local vector), else 
                # failures can occur. 
                clvec = lvec
                csdm = sdm
                ierr = DMPlexSNESComputeBoundaryFEM(csdm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
                # Now copy into the user vec.
                var.vec.array[:] = lvec.array[:]
                sdm.restoreLocalVec(lvec)

        self.dm.restoreGlobalVec(gvec)
