# # Cookbook: examples for uw.mesh.vector 
#
# ## Vector calculus module
#
# `Underworld` provides vector calculus functions for `MeshVariables` comparable to those found in `sympy.vector` but catering for the fact that we use sympy.matrix for `underworld` data objects and equation systems. 
#

import underworld3 as uw
import numpy as np
import sympy
import pytest
from IPython.display import display  # since pytest runs pure python

# ### Mesh based data types
#
# For illustration, we define two and three dimensional meshes and some vector, scalar fields on each one. The validation tests below are designed to run on each of these meshes.

mesh1 = uw.util_mesh.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
                                            maxCoords=(1.0,1.0), 
                                            cellSize=1.0/8.0)


## two vectors and a scalar for testing
v11 = uw.mesh.MeshVariable('U11', mesh1,  mesh1.dim, degree=2 ) 
v12 = uw.mesh.MeshVariable('U12', mesh1,  mesh1.dim, degree=2 )
v13 = uw.mesh.MeshVariable('U13', mesh1,  num_components=3, degree=2, vtype=uw.VarType.VECTOR)  # e.g. stress components
p11 = uw.mesh.MeshVariable('P11', mesh1, 1, degree=1 )


mesh2 = uw.util_mesh.Annulus(radiusOuter=1.0, radiusInner=0.0, cellSize=0.2)

v21 = uw.mesh.MeshVariable('U21', mesh2,  mesh2.dim, degree=2 )
v22 = uw.mesh.MeshVariable('U22', mesh2,  mesh2.dim, degree=2 )
p21 = uw.mesh.MeshVariable('P21', mesh2,  1, degree=1 )

mesh3 = uw.util_mesh.UnstructuredSimplexBox(minCoords=(0.0,0.0,0.0), 
                                            maxCoords=(1.0,1.0,1.0), 
                                            cellSize=1.0/8.0)

## two vectors and a scalar for testing
v31 = uw.mesh.MeshVariable('U31', mesh3,  mesh3.dim, degree=2 ) 
v32 = uw.mesh.MeshVariable('U32', mesh3,  mesh3.dim, degree=2 )
p31 = uw.mesh.MeshVariable('P31', mesh3, 1, degree=1 )

# Validate the meshes / mesh variables

# # Testing

# +
## Tests required to run this notebook 

m1_args = (mesh1,v11,v12,p11)
m2_args = (mesh2,v21,v22,p21)
m3_args = (mesh3,v31,v32,p31)

@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_check_meshes(mesh,v1, v2, p):

    assert mesh is not None
    assert v1 is not None
    assert v2 is not None
    assert p is not None
    
@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_check_mesh_X(mesh,v1, v2, p):
    
    assert mesh1.X.shape == (1,2)
    assert mesh3.X.shape == (1,3)
    
    
# This tests the vector / vector field exists as expected
# and that the vector.to_vector works

@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_ijk(mesh,v1, v2, p):
    assert mesh.vector.to_vector(v1.f) == v1.ijk

@pytest.mark.xfail(raises=AttributeError)
def test_no_ijk():
    
    # This object should not exist - Attribute error
    v13.ijk
    
# tests for dot, div, grad, curl

@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_mesh_vector_dot(mesh,v1, v2, p):

    v1_dot_v2 = v1.f.dot(v2.f)

    v1_dot_v2_explicit = 0.0
    for i in range(mesh.dim):
        v1_dot_v2_explicit += v1.f[i] * v2.f[i]

    assert v1_dot_v2 == v1_dot_v2_explicit

    ## and 
    assert v1_dot_v2 == v1.fn.dot(v2.fn)
    
@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_mesh_vector_div(mesh,v1, v2, p):

    div_v = mesh.vector.divergence(v2.f)

    div_v_explicit = 0.0

    for i,coord in enumerate(mesh.X):
        div_v_explicit += v2.f.diff(coord)[i]

    assert div_v == div_v_explicit

    ## This should also be equivalent, if the .fn interface is not broken !

    assert mesh.vector.divergence(v1.f) == sympy.vector.divergence(v1.fn)
    
@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_mesh_vector_grad(mesh,v1, v2, p):

    grad_p = mesh.vector.gradient(p.f)

    for i,coord in enumerate(mesh.X):
        assert grad_p[i] == p.f[0].diff(coord)

    ## This should also be equivalent, if the .fn interface is not broken !

    assert mesh.vector.gradient(p.f) == mesh.vector.to_matrix(sympy.vector.gradient(p.fn))
    
# Note: The curl is slightly odd - sympy returns the vector in the third dimension, 
# `underworld` returns a scalar because all vectors are assumed to be 2d

@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_mesh_vector_curl(mesh,v1, v2, p):

    curl_v = mesh.vector.curl(v1.f)
    curl_v_sym = sympy.vector.curl(v1.fn)

    if mesh.dim == 2: 
        assert curl_v == curl_v_sym.dot(mesh.N.k)
    else:
        assert curl_v == mesh.vector.to_matrix(curl_v_sym)
        
# Check the jacobians

@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_mesh_vector_jacobian(mesh,v1, v2, p):

    jac_v = mesh.vector.jacobian(v1.f)
    jac_v_sym = v1.f.jacobian(mesh.X)

    assert jac_v == jac_v_sym
    
    # shorthand form
    assert v1.jacobian() == jac_v_sym

## Note this test should be in the MeshVariable section (when that is complete !)

@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_mesh_vector_add(mesh,v1, v2, p):

    v1_plus_v2 = v1.f + v2.f
    v1_plus_v2_explicit = 0.0 * v1.f
    for i in range(mesh.dim):
        v1_plus_v2_explicit[i] = v1.f[i] + v2.f[i]

    assert v1_plus_v2 == v1_plus_v2_explicit


# -

# ## Mesh coordinate systems
#
# The mesh basis vectors are in `mesh.X` which is a 1x2 row vector (`sympy.matrix`) in two dimensions and 1x3 in three dimensions. 

display(mesh1.X)
display(mesh3.X)

# ## Mesh variables as `sympy.matrix` objects
#
# Mesh variables have a `.f` attribute that is a row vectors stored as `sympy.matrix` object. Note: mesh variable scalars `.f` attributes are 1x1 row vectors, not bare `underworld` objects. 
#
#

display(v11.f)
display(v31.f)
display(p11.f)
display(v13.f)

# The mesh variable objects are matrices, but they are structured as vectors. If the fields in the object are of the correct dimension, then they can be interpreted as vector fields and the differential operators such as **div**, **grad**, **curl** are implemented. These broadly correspond to the ones in `sympy.vector` but there are some minor differences in implementation to account for the differences in the underlying objects.
#
# Variables that are vector fields also have a `.ijk` representation that is the conversion to a `sympy.vector` object. In this example, `v11` should be a valid 2d vector field, `v31` is a valid 3d vector field whereas `v13` is "something else".

# +
display(v11.ijk)
display(v31.ijk)

try:
    display(v13.ijk)
except AttributeError:
    print("")
    print("AttributeError: 'MeshVariable' object has no attribute 'ijk'")
# -



# ## div, grad and curl
#
# `sympy` offers symbolic derivatives of functions, and uw variables can be differentiated as follows:
#
# ```
#    v11.f.diff(mesh1.X[0])
#    v13.f.diff(mesh1.X[1])
#    p11.f.diff(mesh1.X[1])
#    
# ```
#
# This is used to construct symbolic forms of the common differential operators:
#   - `mesh.vector.divergence`
#   - `mesh.vector.gradient`
#   - `mesh.vector.curl`
#
# Second derivatives of `underworld` variables are not available at present, and note that the derivatives should be constructed using the mesh that is attached to the mesh variable. The variables themselves know what mesh they live on, so we provide this:
#
#   - `v11.curl()`
#   - `v31.divergence()`
#   - `p11.gradient()`
#   
# for convenience.

display( v11.f.diff(mesh1.X[0]))
display( v13.f.diff(mesh1.X[1]))
display( p11.f.diff(mesh1.X[1]))

# +
# divergence of a vector
display(mesh1.vector.divergence(v11.f))
display(v11.divergence())
display(mesh3.vector.divergence(v31.f))

# gradient of a scalar field
display(mesh1.vector.gradient(p11.f))

# curl of a vector
display(mesh1.vector.curl(v11.f))
display(mesh3.vector.curl(v31.f))

try:
    mesh3.vector.curl(mesh3.vector.gradient(p31.f))
except RuntimeError:
    print("")
    print("RuntimeError: Second derivatives of Underworld functions are not supported at this time.")
# -

# # Jacobian derivatives
#
# By default, the `sympy.diff` operation
#

v31.f.jacobian(v31.mesh.X)

v31.jacobian()

v31.f.diff(v31.mesh.X)

# ## Symbolic forms
#
# As the operations are all in the form of `sympy` manipulations and `sympy` vector calculus routines, the fully symbolic equivalents work just fine, and so do mixtures of symbolic functions and mesh variables.  

# +
x,y,z = mesh3.X

F = sympy.sin(x**2) + sympy.cos(y**2) + z**2 

gradF = mesh3.vector.gradient(F)
curlgradF = mesh3.vector.curl(gradF)
curlgradFplus = mesh3.vector.curl(gradF+v31.f)


display(gradF)
display(curlgradF)
display(curlgradFplus)


G = sympy.Matrix([sympy.cos(2*x*y*z), sympy.exp((y**2+z**2)/sympy.sin(x)), sympy.erfc(z)]).T
display(mesh3.vector.divergence(G))
display(mesh3.vector.curl(G))
