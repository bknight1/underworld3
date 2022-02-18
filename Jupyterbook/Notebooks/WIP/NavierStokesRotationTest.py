# # Navier Stokes test: boundary driven ring with step change in boundary conditions
#
# This should develop a boundary layer with sqrt(t) growth rate

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3.systems import NavierStokesSwarm
from underworld3.systems import Navier_Stokes_SLCN

from underworld3 import function

import numpy as np

options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None
# options.getAll()



# +
import meshio

meshball = uw.meshes.SphericalShell(dim=2, radius_inner=0.5,
                                    radius_outer=1.0, 
                                    cell_size=0.075,
                                    cell_size_lower=0.05,
                                    degree=1, verbose=False)


# +
# Define some functions on the mesh

import sympy

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10+radius_fn)

# Some useful coordinate stuff 

x = meshball.N.x
y = meshball.N.y

r  = sympy.sqrt(x**2+y**2)
th = sympy.atan2(y+1.0e-5,x+1.0e-5)

# Rigid body rotation v_theta = constant, v_r = 0.0

theta_dot = 2.0 * np.pi # i.e one revolution in time 1.0
v_x = -1.0 *  r * theta_dot * sympy.sin(th) * y
v_y =         r * theta_dot * sympy.cos(th) * y
# -

swarm = uw.swarm.Swarm(mesh=meshball)
v_star = uw.swarm.SwarmVariable("Vs", swarm, meshball.dim, proxy_degree=3)
swarm.populate(fill_param=3)


v_soln = uw.mesh.MeshVariable('U',    meshball, meshball.dim, degree=2 )
p_soln = uw.mesh.MeshVariable('P',    meshball, 1, degree=1 )


# +
# Mesh restore function for advection of points
# Which probably should be a feature of the mesh type ...

def points_in_disc(coords): 
    r = np.sqrt(coords[:,0]**2 + coords[:,1]**2).reshape(-1,1)
    outside = np.where(r>1.0)
    coords[outside] *= 0.999 / r[outside]
    return coords


# -

navier_stokes = NavierStokesSwarm(meshball, 
                velocityField=v_soln, 
                pressureField=p_soln, 
                velocityStar=v_star,
                u_degree=2, 
                p_degree=1, 
                rho=1.0,
                theta=0.9,
                solver_name="navier_stokes",
                projection=True,
                restore_points_func=points_in_disc
            )


# +
# Constant visc

expt_name = "Cylinder_NS_rho1"

navier_stokes.viscosity = 1.0
navier_stokes.penalty=0.0
navier_stokes.rho = 1.0

navier_stokes.bodyforce = unit_rvec * 1.0e-16

# Velocity boundary conditions
navier_stokes.add_dirichlet_bc( (v_x,v_y), "Upper" , (0,1) )
navier_stokes.add_dirichlet_bc( (0.0,0.0), "Lower" , (0,1) )

# +
with meshball.access(v_soln):
    v_soln.data[...] = 0.0

with swarm.access(v_star):
    v_star.data[...] = 0.0
# -

navier_stokes.solve(timestep=0.01)
navier_stokes.estimate_dt()



# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0] 
    pv.global_theme.camera['position'] = [0.0, 0.0, 10.0] 

    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    with swarm.access():
            points = np.zeros((swarm.data.shape[0],3))
            points[:,0] = swarm.data[:,0]
            points[:,1] = swarm.data[:,1]

    point_cloud = pv.PolyData(points)


    with meshball.access():
        usol = v_soln.data.copy()

    arrow_loc = np.zeros((v_soln.coords.shape[0],3))
    arrow_loc[:,0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 

    pl = pv.Plotter()

 
    pl.add_arrows(arrow_loc, arrow_length, mag=5.0e-2, opacity=0.75)

    pl.add_points(point_cloud, color="Black",
                      render_points_as_spheres=True,
                      point_size=2, opacity=0.66
                    )
 

    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

    # pl.remove_scalar_bar("T")
    pl.remove_scalar_bar("mag")

    pl.show()


# -
def plot_V_mesh(filename):

    import mpi4py

    if mpi4py.MPI.COMM_WORLD.size==1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = 'white'
        pv.global_theme.window_size = [750, 750]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = 'pythreejs'
        pv.global_theme.smooth_shading = True
        pv.global_theme.camera['viewup'] = [0.0, 1.0, 0.0] 
        pv.global_theme.camera['position'] = [0.0, 0.0, 5.0] 

        pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)
        
        with swarm.access():
            points = np.zeros((swarm.data.shape[0],3))
            points[:,0] = swarm.data[:,0]
            points[:,1] = swarm.data[:,1]

        point_cloud = pv.PolyData(points)

        with meshball.access():
             pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, meshball.data)

        with meshball.access():
            usol = v_soln.data.copy()

        arrow_loc = np.zeros((v_soln.coords.shape[0],3))
        arrow_loc[:,0:2] = v_soln.coords[...]

        arrow_length = np.zeros((v_soln.coords.shape[0],3))
        arrow_length[:,0:2] = usol[...] 
        
        pl = pv.Plotter()

        pl.add_arrows(arrow_loc, arrow_length, mag=0.033, opacity=0.75)

        

        pl.add_points(point_cloud, color="Black",
                      render_points_as_spheres=True,
                      point_size=2, opacity=0.66
                    )
        

        # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)
        pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="P",
                  use_transparency=False, opacity=0.5)



        pl.remove_scalar_bar("P")
        pl.remove_scalar_bar("mag")

        pl.screenshot(filename="{}.png".format(filename), window_size=(1280,1280), 
                      return_img=False)

       # pl.show()



# +
# Convection model / update in time


for step in range(0,20):
        
    delta_t = 0.01 # 5.0*navier_stokes.estimate_dt() 
    navier_stokes.solve(timestep=delta_t)
    
    with swarm.access(v_star):
        v_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data)
     
    # advect swarm
    print("Swarm advection")
    swarm.advection(v_soln.fn, delta_t)
    print("Swarm advection, complete")



    if mpi4py.MPI.COMM_WORLD.rank==0:
        print("Timestep {}, dt {}".format(step, delta_t))

    if step%1 == 0:
        plot_V_mesh(filename="output/{}_step_{}".format(expt_name,step))

# -




