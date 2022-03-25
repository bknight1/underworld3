# # Navier-Stokes flow in a Spherical Domain
#
# This notebook models slow flow in a viscous sphere (or spherical shell)
#
# ## Problem description
#
# Assume that the 
#
# ## Mathematical formulation
#
# The Navier-Stokes equation describes the time-dependent flow of a viscous fluid in response to buoyancy forces and pressure gradients:
#
# \\[
# \rho \frac{\partial \mathbf{u}}{\partial t} + \eta\nabla^2 \mathbf{u} -\nabla p = \rho \mathbf{g}
# \\]
#
# Where $\rho$ is the density, $\eta$ is dynamic viscosity and $\mathbf{g}$ is the gravitational acceleration vector. We here assume that density changes are due to temperature and are small enough to be consistent with an assumption of incompressibility (the Boussinesq approximation). We can rescale this equation of motion using units for length, time, temperature and mass that are specific to the problem and, in this way, obtain a scale-independent form:
#
# \\[
# \frac{1}{\mathrm{Pr}} \frac{\partial \mathbf{u}}{\partial t} + \nabla^2 \mathbf{u} -\nabla p = \mathrm{Ra} T' \hat{\mathbf{g}}
# \\]
#

# where we have assumed that buoyancy forces on the right hand side are due to temperature variations, and the two dimensionless numbers, $\mathrm{Ra}$ and $\mathrm{Pr}$ are measures of the importance of buoyancy forcing and intertial terms, respectively.
#
# \\[
# \mathrm{Ra} = \frac{g\rho_0 \alpha \Delta T d^3}{\kappa \eta} 
# \quad \textrm{and} \quad
# \mathrm{Pr} = \frac{\eta}{\rho \kappa}
# \\]
#
# Here $\alpha$ is the thermal expansivity, $\Delta T$ is the range of the temperature variation, $d$ is the typical length scale over which temperature varies, $\kappa$ is the thermal diffusivity ( $\kappa = k / \rho_0 C_p$; $k$ is thermal conductivity, and $C_p$ is heat capacity). 

# If we assume that the Prandtl number is large, then the inertial terms will not contribute significantly to the balance of forces in the equation of motion because we have rescaled the equations so that the velocity and pressure gradient terms are of order 1. This assumption eliminates the time dependent terms in the equations and tells us that the flow velocity and pressure field are always in equilibrium with the pattern of density variations and this also tells us that we can evaluate the flow without needing to know the history or origin of the buoyancy forces. When the viscosity is independent of velocity and dynamic pressure, the velocity and pressure scale proportionally with $\mathrm{Ra}$ but the flow pattern itself is unchanged. 
#
# The scaling that we use for the non-dimensionalisation is as follows:
#  
# \\[
#     x = d x', \quad t = \frac{d^2}{\kappa} t', \quad T=\Delta T T', \quad 
#     p = p_0 + \frac{\eta \kappa}{d^2} p'
# \\]
#
# where the stress (pressure) scaling using viscosity ($\eta$) determines how the mass scales. In the above, $d$ is the radius of the inner core, a typical length scale for the problem, $\Delta T$ is the order-of-magnitude range of the temperature variation from our observations, and $\kappa$ is thermal diffusivity. The scaled velocity is obtained as $v = \kappa / d v'$.

# ## Formulation & model
#
#
# The model consists of a spherical ball divided into an unstructured tetrahedral mesh of quadratic velocity, linear pressure elements with a free slip upper boundary and with a buoyancy force pre-defined :
#
# \\[
# T(r,\theta,\phi) =  T_\textrm{TM}(\theta, \phi) \cdot r  \sin(\pi r) 
# \\] 

# ## Coriolis and Lorentz Forces
#
# Have to implement the Coriolis as a semi-dynamic term, Lorentz force can be prescribed
#

# ## Computational script in python

# +
visuals = 0
output_dir = "output"
expt_name="NS_Sphere_C10"

# Some gmsh issues, so we'll use a pre-built one
mesh_file = "Sample_Meshes_Gmsh/test_mesh_sphere_at_res_005_c.msh"
res = 0.3
r_o = 1.0
r_i = 0.0

Rayleigh = 1.0  # Doesn't actually matter to the solution pattern, 
                # choose 1 to make re-scaling simple

iic_radius = 0.0
iic_delta_eta = 100.0
import os

os.makedirs(output_dir, exist_ok=True)

# +
# Imports here seem to be order dependent again (pygmsh / gmsh v. petsc

import pygmsh
import meshio

import petsc4py
from petsc4py import PETSc
import mpi4py

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy


# +
# meshball = uw.meshes.MeshFromGmshFile(dim=3, filename=mesh_file, verbose=True, simplex=True)
# meshball.meshio.remove_lower_dimensional_cells()

meshball = uw.meshes.SphericalShell(dim=3, degree=1, 
                                    radius_inner=r_i, 
                                    centre_point=False,
                                    radius_outer=r_o, 
                                    cell_size=res, 
                                    cell_size_upper=res,
                                    verbose=False)
# -

meshball.dm.view()

# +
v_soln   = uw.mesh.MeshVariable('U', meshball, meshball.dim, degree=2 )
v_soln_1 = uw.mesh.MeshVariable('U_1', meshball, meshball.dim, degree=2 )
p_soln   = uw.mesh.MeshVariable('P', meshball, 1, degree=1 )
t_soln   = uw.mesh.MeshVariable("T", meshball, 1, degree=3 )
r        = uw.mesh.MeshVariable('R', meshball, 1, degree=1 )

# vorticity = uw.mesh.MeshVariable('omega',  meshball, 1, degree=1 )

swarm = uw.swarm.Swarm(mesh=meshball)
v_star     = uw.swarm.SwarmVariable("Vs", swarm, meshball.dim, proxy_degree=2)
remeshed   = uw.swarm.SwarmVariable("Vw", swarm, 1, dtype='int', _proxy=False)
X_0        = uw.swarm.SwarmVariable("X0", swarm, meshball.dim, _proxy=False)

swarm.populate(fill_param=3)

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre 
# of the sphere to (say) 1 at the surface

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10+radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff 

x = meshball.N.x
y = meshball.N.y
z = meshball.N.z

# r  = sympy.sqrt(x**2+y**2+z**2)  # cf radius_fn which is 0->1 
th = sympy.atan2(y+1.0e-10,x+1.0e-10)
ph = sympy.acos(z / (r.fn+1.0e-10))

hw = 1000.0 / res 
surface_fn = sympy.exp(-((r.fn - r_o) / r_o)**2 * hw)

## Buoyancy (T) field

t_forcing_fn = 1.0 * ( sympy.exp(-5.0*(x**2+(y-0.8)**2+z**2)) + 
                       sympy.exp(-10.0*((x-0.8)**2+y**2+z**2)) +
                       sympy.exp(-15.0*(x**2+y**2+(z-0.8)**2))        
                     )


# +
"""
lons = uw.function.evaluate(th, t_soln.coords)
lats = uw.function.evaluate(ph, t_soln.coords)

ic_raw = np.loadtxt("./qtemp_6000.xyz")  # Data: lon, lat, ?, ?, ?, T
ic_data = ic_raw[:,5].reshape(181,361)

## Map heights/ages to the even-mesh grid points

def map_raster_to_mesh(lons, lats, raster):

    latitudes_in_radians  = lats
    longitudes_in_radians = lons 
    latitudes_in_degrees  = np.degrees(latitudes_in_radians) 
    longitudes_in_degrees = np.degrees(longitudes_in_radians)

    dlons = (longitudes_in_degrees + 180)
    dlats = latitudes_in_degrees 

    ilons = raster.shape[0] * dlons / 360.0
    ilats = raster.shape[1] * dlats / 180.0

    icoords = np.stack((ilons, ilats))

    from scipy import ndimage

    mvals = ndimage.map_coordinates(raster, icoords , order=3, mode='nearest').astype(float)
    
    return mvals

qt_vals = map_raster_to_mesh(lons, lats, ic_data.T)

with meshball.access(t_soln):
    t_soln.data[...] = (uw.function.evaluate(radius_fn**2, t_soln.coords) * qt_vals).reshape(-1,1)
"""

pass
# +
orientation_wrt_z = sympy.atan2(y+1.0e-10,x+1.0e-10)
v_rbm_z_x = -r.fn * sympy.sin(orientation_wrt_z) * meshball.N.i
v_rbm_z_y =  r.fn * sympy.cos(orientation_wrt_z) * meshball.N.j
v_rbm_z   =  v_rbm_z_x + v_rbm_z_y

orientation_wrt_x = sympy.atan2(z+1.0e-10,y+1.0e-10)
v_rbm_x_y = -r.fn * sympy.sin(orientation_wrt_x) * meshball.N.j
v_rbm_x_z =  r.fn * sympy.cos(orientation_wrt_x) * meshball.N.k
v_rbm_x   =  v_rbm_x_z + v_rbm_x_y 

orientation_wrt_y = sympy.atan2(z+1.0e-10,x+1.0e-10)
v_rbm_y_x = -r.fn * sympy.sin(orientation_wrt_y) * meshball.N.i
v_rbm_y_z =  r.fn * sympy.cos(orientation_wrt_y) * meshball.N.k
v_rbm_y   =  v_rbm_y_z + v_rbm_y_x 


# vx = -vtheta*sympy.sin(th)
# vy =  vtheta*sympy.cos(th)
# +
# Create NS object

navier_stokes = uw.systems.NavierStokesSwarm(meshball, 
                velocityField=v_soln, 
                pressureField=p_soln, 
                velocityStar_fn=v_star.fn,
                u_degree=v_soln.degree, 
                p_degree=p_soln.degree, 
                rho=1.0,
                theta=0.5,
                verbose=False,
                projection=True,
                solver_name="navier_stokes")

navier_stokes.petsc_options.delValue("ksp_monitor") # We can flip the default behaviour at some point
navier_stokes.petsc_options["snes_rtol"]=3.0e-3 # We can flip the default behaviour at some point

navier_stokes._u_star_projector.petsc_options.delValue("ksp_monitor")
navier_stokes._u_star_projector.petsc_options["snes_rtol"] = 1.0e-2
navier_stokes._u_star_projector.petsc_options["snes_type"] = "newtontr"
navier_stokes._u_star_projector.smoothing = 0.0 # navier_stokes.viscosity * 1.0e-6
navier_stokes._u_star_projector.penalty = 0.0

navier_stokes.theta=0.5
navier_stokes.viscosity = 1.0

# pseudo-steady-state
navier_stokes.UF0 =  -navier_stokes.rho * (v_soln.fn - v_soln_1.fn) / navier_stokes.delta_t

# thermal buoyancy force
buoyancy_force = Rayleigh * gravity_fn * t_forcing_fn 

# Free slip condition by penalizing radial velocity at the surface (non-linear term)
free_slip_penalty  = Rayleigh * 1.0e6 *  v_soln.fn.dot(unit_rvec) * unit_rvec * surface_fn
solid_body_penalty = 1.0e2   * (v_rbm_x * v_soln.fn.dot(v_rbm_x) + 
                                v_rbm_y * v_soln.fn.dot(v_rbm_y) +
                                v_rbm_z * v_soln.fn.dot(v_rbm_z))

navier_stokes.bodyforce = unit_rvec * buoyancy_force 
navier_stokes.bodyforce -= free_slip_penalty + solid_body_penalty     

navier_stokes._Ppre_fn = 1.0 / (navier_stokes.viscosity + navier_stokes.rho / navier_stokes.delta_t + Rayleigh * 1.0e1 * surface_fn)

# Velocity boundary conditions

# navier_stokes.add_dirichlet_bc( (0.0, 0.0, 0.0), "Upper", (0,1,2))
navier_stokes.add_dirichlet_bc( (0.0, 0.0, 0.0), "Centre", (0,1,2))


v_theta = navier_stokes.theta * navier_stokes.u.fn + (1.0 - navier_stokes.theta) * navier_stokes.u_star_fn
# v_theta = navier_stokes.u.fn

# +
with meshball.access(r):
    r.data[:,0]   = uw.function.evaluate(sympy.sqrt(x**2+y**2+z**2), meshball.data)  # cf radius_fn which is 0->1 
    
with meshball.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(t_forcing_fn, t_soln.coords).reshape(-1,1)


# +
navier_stokes.solve(timestep=10000.0)

with meshball.access(v_soln_1):
    v_inertial = v_soln.data.copy()
    v_soln_1.data[...] = v_soln.data[...]

with swarm.access(v_star, remeshed, X_0):
    v_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data) 
    X_0.data[...] = swarm.data[...] 


# +
# Mesh restore function for advection of points
# Which probably should be a feature of the mesh type ...

def points_in_sphere(coords): 
    r = np.sqrt(coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2).reshape(-1,1)
    outside = np.where(r>1.0)
    coords[outside] *= 0.99 / r[outside]
    return coords    

swarm.advection(v_soln.fn, 
                delta_t=navier_stokes.estimate_dt(),
                restore_points_to_domain_func=points_in_sphere,
                corrector=False)


# -

def plot_V_mesh(filename):
# # +
# check the mesh if in a notebook / serial

    import mpi4py

    if mpi4py.MPI.COMM_WORLD.size==1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = 'white'
        pv.global_theme.window_size = [1250, 1250]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = 'panel'
        pv.global_theme.smooth_shading = True

        pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TETRA)

        with meshball.access():
            usol = navier_stokes.u.data.copy()
            print("usol - magnitude {}".format(np.sqrt((usol**2).mean())))        


        pvmesh.point_data["T"]  = uw.function.evaluate(t_forcing_fn, meshball.data)
        # pvmesh.point_data["S"]  = uw.function.evaluate(surface.fn, meshball.data)

        arrow_loc = np.zeros((navier_stokes.u.coords.shape[0],3))
        arrow_loc[...] = navier_stokes.u.coords[...]

        arrow_length = np.zeros((navier_stokes.u.coords.shape[0],3))
        arrow_length[...] = usol[...] 


        pl = pv.Plotter()

        # pl.add_mesh(pvmesh,'Black', 'wireframe')

        pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                      use_transparency=False, opacity=1.0)

        pl.add_arrows(arrow_loc, arrow_length, mag=30.0)

        pl.screenshot(filename="{}.png".format(filename), window_size=(2560,2560),
                      return_img=False)
        
        pl.close()
        
        del(pl)



ts = 0
swarm_loop = 5

# +
# PSEUDO T-STEP LOOP

for step in range(0,50):
    
    Omega = 1.0 * meshball.N.k * min(ts/5, 1.0) 
    navier_stokes.bodyforce = unit_rvec * buoyancy_force 
    navier_stokes.bodyforce -= free_slip_penalty + solid_body_penalty     
    navier_stokes.bodyforce -= 2.0 * navier_stokes.rho * sympy.vector.cross(Omega, v_theta)
       
    delta_t = 3.0 * navier_stokes.estimate_dt()
    
    navier_stokes.solve(timestep=delta_t, 
                        zero_init_guess=False)
    
    dv_fn = v_soln.fn - v_soln_1.fn
    _,_,_,_,_,_,deltaV = meshball.stats(dv_fn.dot(dv_fn))


    with meshball.access(v_soln,v_soln_1):
        v_soln.data[...] = 0.5 * v_soln_1.data[...] + 0.5 * v_soln.data[...] 
        v_soln_1.data[...] = v_soln.data[...] 

    with swarm.access(v_star):
        v_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data)

    swarm.advection(v_soln.fn, 
                delta_t=navier_stokes.estimate_dt(),
                restore_points_to_domain_func=points_in_sphere,
                corrector=False)
    
    # Restore a subset of points to start
    offset_idx = step%swarm_loop
    
    with swarm.access(swarm.particle_coordinates, remeshed):
        remeshed.data[...] = 0
        remeshed.data[offset_idx::swarm_loop,:] = 1
        swarm.data[offset_idx::swarm_loop,:] = X_0.data[offset_idx::swarm_loop,:]
        
    # re-calculate v history for remeshed particles 
    # Note, they may have moved procs after the access manager closed
    # so we re-index 
    
    with swarm.access(v_star, remeshed):
        idx = np.where(remeshed.data == 1)[0]
        v_star.data[idx] = uw.function.evaluate(v_soln.fn, swarm.data[idx]) 

    if uw.mpi.rank==0:
        print("Timestep {}, dt {}, deltaV {}".format(ts, delta_t, deltaV))
                
    if ts%1 == 0:
        # nodal_vorticity_from_v.solve()
        plot_V_mesh(filename="output/{}_step_{}".format(expt_name,ts))
        
        # savefile = "output/{}_ts_{}.h5".format(expt_name,step) 
        # meshball.save(savefile)
        # v_soln.save(savefile)
        # p_soln.save(savefile)
        # vorticity.save(savefile)
        # meshball.generate_xdmf(savefile)
        
    # navier_stokes._u_star_projector.smoothing = navier_stokes.viscosity * 1.0e-6
    
    ts += 1


# +
## Save data

# savefile = "{}/free_slip_sphere.h5".format(output_dir)
# meshball.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshball.generate_xdmf(savefile)

# +
# OR

# # +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 600]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TETRA)

    with meshball.access():
        usol = uw.function.evaluate(v_soln.fn, navier_stokes.u.coords) - v_inertial
        print("usol - magnitude {}".format(np.sqrt((usol**2).mean())))        
        
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_forcing_fn, meshball.data)
    # pvmesh.point_data["S"]  = uw.function.evaluate(surface.fn, meshball.data)

    arrow_loc = np.zeros((navier_stokes.u.coords.shape[0],3))
    arrow_loc[...] = navier_stokes.u.coords[...]
    
    arrow_length = np.zeros((navier_stokes.u.coords.shape[0],3))
    arrow_length[...] = usol[...] 
    
    clipped = pvmesh.clip(origin=(0.001,0.0,0.0), normal=(0, 0, 1), invert=False)

    pl = pv.Plotter()
    pl.add_axes()
    

    pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                  use_transparency=False, opacity=1.0)

    
    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
    #               use_transparency=False, opacity=1.0)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=1.0e2)
    
    pl.show(cpos="xy")

# -
meshball.stats(v_soln.fn.dot(v_rbm_y)), meshball.stats(v_star.fn.dot(v_rbm_y))

meshball.stats(v_soln.fn.dot(v_rbm_z)), meshball.stats(v_star.fn.dot(v_rbm_z))

meshball.stats(v_soln.fn.dot(v_rbm_x)), meshball.stats(v_star.fn.dot(v_rbm_x))





usol

v_inertial


