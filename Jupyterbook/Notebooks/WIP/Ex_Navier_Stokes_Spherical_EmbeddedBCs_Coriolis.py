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
expt_name="NS_Sphere_C50_iii"

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
# Build this one by hand 

csize_local = 0.3
cell_size_lower = 0.3
cell_size_upper = 0.2
radius_outer = 1.0
radius_inner = 0.0

import pygmsh
import meshio

# Generate local mesh.
with pygmsh.geo.Geometry() as geom:
    
    geom.characteristic_length_max = csize_local
    
    if radius_inner > 0.0:
        inner  = geom.add_ball((0.0,0.0,0.0),radius_inner, with_volume=False, mesh_size=cell_size_lower)
        outer = geom.add_ball((0.0,0.0,0.0), radius_outer, with_volume=False, mesh_size=cell_size_upper)
        domain = geom.add_ball((0.0,0.0,0.0), radius_outer*1.25, mesh_size=cell_size_upper, holes=[inner.surface_loop])

        geom.add_physical(inner.surface_loop.surfaces,  label="Lower")
        geom.add_physical(outer.surface_loop.surfaces,  label="Upper")
        geom.add_physical(domain.surface_loop.surfaces, label="Celestial_Sphere")
        geom.add_physical(domain.volume, label="Elements")

    else:
        centre = geom.add_point((0.0,0.0,0.0), mesh_size=cell_size_lower)
        outer = geom.add_ball((0.0,0.0,0.0), radius_outer, with_volume=False, mesh_size=cell_size_upper)
        domain = geom.add_ball((0.0,0.0,0.0), radius_outer*1.25, mesh_size=cell_size_upper)
        geom.in_volume(centre, domain.volume)
        
        for surf in outer.surface_loop.surfaces:
            geom.in_volume(surf, domain.volume)
      
        geom.add_physical(centre,  label="Centre")
        geom.add_physical(outer.surface_loop.surfaces, label="Upper")
        geom.add_physical(domain.surface_loop.surfaces, label="Celestial_Sphere")
        geom.add_physical(domain.volume, label="Elements")    


        
    geom.generate_mesh(dim=3, verbose=True)
    geom.save_geometry("ignore_celestial_3d.msh")
    geom.save_geometry("ignore_celestial_3d.vtk")
        
    
meshball = uw.meshes.MeshFromGmshFile(dim=3, degree=1, filename="ignore_celestial_3d.msh", label_groups=[], simplex=True)
# -

meshball.dm.view()

# +
v_soln   = uw.mesh.MeshVariable('U',  meshball, meshball.dim, degree=2 )
v_soln_1 = uw.mesh.MeshVariable('U_1',meshball, meshball.dim, degree=2 )
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

# 
mask_fn     = sympy.Piecewise( ( 1.0,  r.fn <= radius_outer ),                               ( 0, True ))        
i_mask_fn   = sympy.Piecewise( ( 1.0,  r.fn <  radius_outer ),                               ( 0, True ))
surface_fn  = sympy.Piecewise( ( 1.0, (r.fn - radius_outer)**2 < 0.05*cell_size_upper**2  ), ( 0, True ))
sky_mask_fn = sympy.Piecewise( ( 1.0,  r.fn > radius_outer ),                                ( 0,True) )


## Buoyancy (T) field

t_forcing_fn = 1.0 * ( sympy.exp(-10.0*(x**2+(y-0.8)**2+z**2)) + 
                       sympy.exp(-10.0*((x-0.8)**2+y**2+z**2)) +
                       sympy.exp(-10.0*(x**2+y**2+(z-0.8)**2))        
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
# Rigid body rotations that are null-spaces for this set of bc's 

# We can remove these after the fact, but also useful to double check
# that we are not adding anything to excite these modes in the forcing terms. 

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
# -

volume = sympy.vector.divergence(sympy.vector.cross(meshball.N.k, v_soln.fn))

vorticity = meshball.N.k.dot(sympy.vector.curl(v_soln.fn))





0/0

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
navier_stokes._u_star_projector.petsc_options["snes_rtol"] = 1.0e-3
navier_stokes._u_star_projector.petsc_options["snes_type"] = "newtontr"
navier_stokes._u_star_projector.smoothing = 0.0 # navier_stokes.viscosity * 1.0e-6
navier_stokes._u_star_projector.penalty = 0.0

navier_stokes.theta=0.5
# Constant visc

navier_stokes.rho=0.01 + mask_fn * 1.0 
navier_stokes.theta=0.5
navier_stokes.penalty=0.0
navier_stokes.viscosity = 0.1 + mask_fn * 1.0 

# Free slip condition by penalizing radial velocity at the surface (non-linear term)
free_slip_penalty  =  1.0e4 * Rayleigh * v_soln.fn.dot(unit_rvec) * unit_rvec * surface_fn 

# Velocity boundary conditions

navier_stokes.add_dirichlet_bc( (0.0, 0.0), "Celestial_Sphere",  (0,1))
# navier_stokes.add_dirichlet_bc( (0.0, 0.0), "Centre", (0,1))

v_theta = navier_stokes.theta * navier_stokes.u.fn + (1.0 - navier_stokes.theta) * navier_stokes.u_star_fn


# +
v_proj = navier_stokes._u_star_projector.u
free_slip_penalty_p  =  100 * v_proj.fn.dot(unit_rvec) * unit_rvec * surface_fn 
navier_stokes._u_star_projector.F0 =  free_slip_penalty_p


navier_stokes.bodyforce = Rayleigh * unit_rvec * t_forcing_fn * i_mask_fn # minus * minus
navier_stokes.bodyforce -= free_slip_penalty # + solid_body_penalty 

# navier_stokes._Ppre_fn = 1.0 / (navier_stokes.viscosity + navier_stokes.rho / navier_stokes.delta_t + Rayleigh * surface_fn)



# +
with meshball.access(r):
    r.data[:,0]   = uw.function.evaluate(sympy.sqrt(x**2+y**2+z**2), meshball.data)  # cf radius_fn which is 0->1 
    
with meshball.access(t_soln):
    t_soln.data[...] = uw.function.evaluate(t_forcing_fn, t_soln.coords).reshape(-1,1)


# +
navier_stokes.solve(timestep=100.0)

_,x_ns,_,_,_,_,_ = meshball.stats(v_soln.fn.dot(v_rbm_x))
_,y_ns,_,_,_,_,_ = meshball.stats(v_soln.fn.dot(v_rbm_y))
_,z_ns,_,_,_,_,_ = meshball.stats(v_soln.fn.dot(v_rbm_z))

print("Rigid body: {}, {}, {} (x,y,z axis)".format(x_ns, y_ns, z_ns))

with meshball.access(v_soln_1):
    v_soln_1.data[...] = v_soln.data[...]

with swarm.access(v_star, X_0):
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
            usol  = uw.function.evaluate(navier_stokes.u.fn * mask_fn,
                                         navier_stokes.u.coords) 
            print("usol - magnitude {}".format(np.sqrt((usol**2).mean())))        


        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn*mask_fn, meshball.data)
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn*mask_fn, meshball.data)
        # pvmesh.point_data["S"]  = uw.function.evaluate(surface.fn, meshball.data)

        arrow_loc = np.zeros((navier_stokes.u.coords.shape[0],3))
        arrow_loc[...] = navier_stokes.u.coords[...]

        arrow_length = np.zeros((navier_stokes.u.coords.shape[0],3))
        arrow_length[...] = usol[...] 


        pl = pv.Plotter(window_size=[1000,1000])

        # pl.add_mesh(pvmesh,'Black', 'wireframe')

        pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
                      use_transparency=False, opacity=1.0)

        pl.add_arrows(arrow_loc, arrow_length, mag=30.0)

        pl.screenshot(filename="{}.png".format(filename), window_size=(2560,2560),
                      return_img=False)
        
        pl.close()
        
        del(pl)

print("timestep_0 = {}".format(navier_stokes.estimate_dt()))

ts = 0
swarm_loop = 5

# +
# PSEUDO T-STEP LOOP

for step in range(0,35):
    
    # navier_stokes.petsc_options["snes_type"] = "newtontr"
    
    Omega_0 = 10.0 * min(ts/30, 1.0) 
    Omega = meshball.N.k * Omega_0    
    
    navier_stokes.bodyforce = Rayleigh * unit_rvec * t_init * i_mask_fn # minus * minus
    navier_stokes.bodyforce -= free_slip_penalty 
    navier_stokes.bodyforce -= 2.0 * navier_stokes.rho * sympy.vector.cross(Omega, v_theta) * mask_fn
    
    
    navier_stokes._Ppre_fn = 1.0 / (navier_stokes.viscosity + 
                                    navier_stokes.rho / navier_stokes.delta_t + 
                                    Omega_0 * mask_fn + 
                                    Rayleigh * surface_fn)
       
    delta_t = 3.0 * navier_stokes.estimate_dt()
    
    navier_stokes.solve(timestep=delta_t, zero_init_guess=False)
    
    _,x_ns,_,_,_,_,_ = meshball.stats(v_soln.fn.dot(v_rbm_x))
    _,y_ns,_,_,_,_,_ = meshball.stats(v_soln.fn.dot(v_rbm_y))
    _,z_ns,_,_,_,_,_ = meshball.stats(v_soln.fn.dot(v_rbm_z))

    print("Rigid body: {}, {}, {} (x,y,z axis)".format(x_ns, y_ns, z_ns))

    
    dv_fn = v_soln.fn - v_soln_1.fn
    _,_,_,_,_,_,deltaV = meshball.stats(dv_fn.dot(dv_fn))


    with meshball.access(v_soln,v_soln_1):
        v_soln_1.data[...] = v_soln.data[...] 

    with swarm.access(v_star):
        v_star.data[...] = 0.5 * uw.function.evaluate(v_soln.fn, swarm.data) + 0.5 * v_star.data[...]

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
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    

    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TETRA)
    
    coriolis_term = -sympy.vector.cross(Omega, v_theta) + 1.0e-8 * meshball.N.k

    with meshball.access():
        usol  = uw.function.evaluate(v_soln.fn, navier_stokes.u.coords) # - v_inertial
        corio = uw.function.evaluate(coriolis_term * (surface_fn),
                                     navier_stokes.u.coords)        
  
    pvmesh.point_data["T"]  = uw.function.evaluate(t_forcing_fn, meshball.data)
    pvmesh.point_data["P"]  = uw.function.evaluate(p_soln.fn, meshball.data)

    
    # pvmesh.point_data["S"]  = uw.function.evaluate(surface.fn, meshball.data)

    arrow_loc = np.zeros((navier_stokes.u.coords.shape[0],3))
    arrow_loc[...] = navier_stokes.u.coords[...]
    
    arrow_length = np.zeros((navier_stokes.u.coords.shape[0],3))
    arrow_length[...] = usol[...] 
    
    clipped = pvmesh.clip(origin=(0.001,0.0,0.0), normal=(0, 0, 1), invert=False)

    pl = pv.Plotter(window_size=[1000,1000])
    pl.add_axes()
    

    pl.add_mesh(clipped, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="P",
                  use_transparency=False, opacity=1.0)

    
    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T",
    #               use_transparency=False, opacity=1.0)
    
    pl.add_arrows(arrow_loc, arrow_length, mag=100)
    
    pl.show(cpos="xy")

# -
meshball.stats(v_soln.fn.dot(v_rbm_y)), meshball.stats(v_star.fn.dot(v_rbm_y))

meshball.stats(v_soln.fn.dot(v_rbm_z)), meshball.stats(v_star.fn.dot(v_rbm_z))

meshball.stats(v_soln.fn.dot(v_rbm_x)), meshball.stats(v_star.fn.dot(v_rbm_x))

meshball.stats(sympy.vector.cross(Omega, v_soln.fn).dot(v_rbm_z))




