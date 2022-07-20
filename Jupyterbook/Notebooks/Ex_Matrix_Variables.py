# # Overview of the sympy.Matrix interface
#
# Demonstrating the use of the `sympy.matrix` interface to solvers and mesh variables.
#

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

render = True

# +
import meshio

mesh = uw.util_mesh.Annulus(radiusOuter=1.0, radiusInner=0.0, cellSize=0.05)


# +
import sympy

# Some useful coordinate stuff 

x = mesh.N.x
y = mesh.N.y
# +
# but we have this on the mesh in matrix form

mesh.X
# -

mesh.r

mesh.vector.to_vector(mesh.X)

v  = uw.mesh.MeshVariable('U',    mesh,  mesh.dim, degree=2 )
v2 = uw.mesh.MeshVariable('U2',   mesh,  mesh.dim, degree=2 )
p  = uw.mesh.MeshVariable('P',    mesh, 1, degree=1 )


mesh.vector.to_matrix(v._ijk)

mesh.vector.to_matrix(v.f)

vec = sympy.vector.matrix_to_vector(v.f, mesh.N)
# check it 
print(isinstance(vec, sympy.vector.Vector))

mesh.vector.to_vector(v.f)

# +
# This is the way to get back to a matrix after diff by array for two row vectors (yuck)
# We might just try to wrap this so it give back sane types.
# Diff by array blows up a 1x3 by 1x3 into a 1,3,1,3 tensor rather than a 3x3 matrix 
# and the 1 indices cannot be automatically removed 

VX = sympy.derive_by_array(v.f, mesh.X).reshape(v.f.shape[1], mesh.X.shape[1]).tomatrix().T

v.f.jacobian(mesh.X)

# -

VX

p.f.jacobian(mesh.X)

sympy.vector.divergence(v.fn)

sympy.vector.divergence(mesh.vector.to_vector(v.f))

sympy.vector.curl(mesh.vector.to_vector(v.f))

mesh.vector.curl(v.f)

v.f.diff(y)[1]

mesh.vector.divergence(v.f) == sympy.vector.divergence(v.fn)

mesh.vector.gradient(p.f)

v.f.dot(v.f)

v.f.dot(v2.f)

v.fn.dot(v2.fn)

sympy.Matrix((p.f, p.f)).T + v.f

mesh.X

sympy.vector.gradient(p.fn)

mesh.vector.to_matrix(sympy.vector.curl(v.fn))


