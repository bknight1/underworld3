import underworld3 as uw
import numpy as np
import sympy
import pytest

mesh1 = uw.util_mesh.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
                                            maxCoords=(1.0,1.0), 
                                            cellSize=1.0/8.0)


## two vectors and a scalar for testing
v11 = uw.mesh.MeshVariable('U11', mesh1,  mesh1.dim, degree=2 ) 
v12 = uw.mesh.MeshVariable('U12', mesh1,  mesh1.dim, degree=2 )
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

# Validate the meshes / mesh variables just in case !

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
def test_mesh_vector_add(mesh,v1, v2, p):

    v1_plus_v2 = v1.f + v2.f
    v1_plus_v2_explicit = 0.0 * v1.f
    for i in range(mesh.dim):
        v1_plus_v2_explicit[i] = v1.f[i] + v2.f[i]


    assert v1_plus_v2 == v1_plus_v2_explicit



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


## The curl is slightly odd - sympy returns the vector in the third dimension,
## uw returns a scalar because all vectors are assumed to be 2d

@pytest.mark.parametrize("mesh, v1, v2, p", [m1_args, m2_args, m3_args])
def test_mesh_vector_curl(mesh,v1, v2, p):

    curl_v = mesh.vector.curl(v1.f)
    curl_v_sym = sympy.vector.curl(v1.fn)

    if mesh.dim == 2: 
        assert curl_v == curl_v_sym.dot(mesh.N.k)
    else:
        assert curl_v == mesh.vector.to_matrix(curl_v_sym)









