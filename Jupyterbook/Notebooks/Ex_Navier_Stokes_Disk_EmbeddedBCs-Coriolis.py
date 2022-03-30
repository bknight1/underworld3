# # Cylindrical Stokes with Coriolis term (out of plane)

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
import numpy as np
import sympy

# -


expt_name = "NS_ES_flow_coriolis_disk_100_i"

# +
import meshio
import pygmsh

csize_local = 0.1
cell_size_upper = 0.075
cell_size_lower = 0.1


radius_outer = 1.0
radius_inner = 0.0

if uw.mpi.rank==0:

    with pygmsh.geo.Geometry() as geom:

        geom.characteristic_length_max = csize_local
        outer  = geom.add_circle((0.0,0.0,0.0),radius_outer, make_surface=False, mesh_size=cell_size_upper)

        if radius_inner > 0.0:   
            inner  = geom.add_circle((0.0,0.0,0.0),radius_inner, make_surface=False, mesh_size=cell_size_upper)
            domain = geom.add_circle((0.0,0.0,0.0), radius_outer*1.25, mesh_size=cell_size_upper, holes=[inner])
            geom.add_physical(inner.curve_loop.curves, label="Centre")       
            for l in inner.curve_loop.curves:
                geom.set_transfinite_curve(l, num_nodes=7, mesh_type="Progression", coeff=1.0)

        else:
            centre = geom.add_point((0.0,0.0,0.0), mesh_size=cell_size_lower)       
            domain = geom.add_circle((0.0,0.0,0.0), radius_outer*1.25, mesh_size=cell_size_upper)
            geom.in_surface(centre, domain.plane_surface)
            geom.add_physical(centre, label="Centre")

        for l in outer.curve_loop.curves:
            geom.in_surface(l, domain.plane_surface)
            geom.set_transfinite_curve(l, num_nodes=40, mesh_type="Progression", coeff=1.0)

        for l in domain.curve_loop.curves:
            geom.set_transfinite_curve(l, num_nodes=40, mesh_type="Progression", coeff=1.0)

        geom.add_physical(outer.curve_loop.curves, label="Upper")       
        geom.add_physical(domain.curve_loop.curves, label="Celestial_Sphere")

        # This is not really needed in the label list - it's everything else
        geom.add_physical(domain.plane_surface, label="Elements")

        geom.generate_mesh(dim=2, verbose=True)
        geom.save_geometry("ignore_celestial.msh")
        geom.save_geometry("ignore_celestial.vtk")


meshball = uw.meshes.MeshFromGmshFile(dim=2, degree=1, filename="ignore_celestial.msh", label_groups=[], simplex=True)


# meshball = uw.meshes.SphericalShell(dim=2, radius_outer=1.0, radius_inner=0.0, cell_size=0.075, degree=1, verbose=False)                       
# -

meshball.dm.view()

# +
v_soln = uw.mesh.MeshVariable('U',meshball, 2, degree=2 )
p_soln = uw.mesh.MeshVariable('P',meshball, 1, degree=1 )
t_soln = uw.mesh.MeshVariable("T",meshball, 1, degree=3 )
r        = uw.mesh.MeshVariable('R', meshball, 1, degree=1 )


v_soln_1  = uw.mesh.MeshVariable('U_1',    meshball, meshball.dim, degree=2 )
vorticity = uw.mesh.MeshVariable('omega',  meshball, 1, degree=1 )


# +
swarm = uw.swarm.Swarm(mesh=meshball)
v_star     = uw.swarm.SwarmVariable("Vs", swarm, meshball.dim, proxy_degree=3)
remeshed   = uw.swarm.SwarmVariable("Vw", swarm, 1, dtype='int', _proxy=False)
X_0        = uw.swarm.SwarmVariable("X0", swarm, meshball.dim, _proxy=False)

swarm.populate(fill_param=4)

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre 
# of the sphere to (say) 1 at the surface

import sympy

radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec)) # normalise by outer radius if not 1.0
unit_rvec = meshball.rvec / (1.0e-10+radius_fn)
gravity_fn = radius_fn

# Some useful coordinate stuff 

x = meshball.N.x
y = meshball.N.y

# r  = sympy.sqrt(x**2+y**2)
th = sympy.atan2(y+1.0e-5,x+1.0e-5)

# 
Rayleigh = 1.0e2

# 
# hw = 1000.0 / cell_size_upper
# surface_fn = sympy.exp(-((r.fn - 1.0) / 1.0)**2 * hw)


mask_fn = sympy.Piecewise( ( 1.0,  r.fn <= radius_outer ),( 0, True ))        
i_mask_fn = sympy.Piecewise( ( 1.0, r.fn <  radius_outer ),( 0, True ))
surface_fn = sympy.Piecewise( ( 1.0,  (r.fn - radius_outer)**2 < 0.05*cell_size_upper**2  ), (0, True ))

sky_mask_fn = sympy.Piecewise( ( 1.0, r.fn > radius_outer ),( 0,True) )

# 
# mask_fn = 0.5 - 0.5 * sympy.tanh(1000.0*(r.fn-1.003)) 
# i_mask_fn = 0.5 - 0.5 * sympy.tanh(1000.0*(r.fn-0.997)) 
# sky_mask_fn = 1.0 - mask_fn

# surface_fn = mask_fn - i_mask_fn
# -
orientation_wrt_z = sympy.atan2(y+1.0e-10,x+1.0e-10)
v_rbm_z_x = -r.fn * sympy.sin(orientation_wrt_z) * meshball.N.i
v_rbm_z_y =  r.fn * sympy.cos(orientation_wrt_z) * meshball.N.j
v_rbm_z   =  v_rbm_z_x + v_rbm_z_y

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
navier_stokes._u_star_projector.petsc_options.delValue("ksp_monitor")
navier_stokes._u_star_projector.petsc_options["snes_rtol"] = 1.0e-2
navier_stokes._u_star_projector.petsc_options["snes_type"] = "newtontr"
navier_stokes._u_star_projector.smoothing = 0.0 # navier_stokes.viscosity * 1.0e-6
navier_stokes._u_star_projector.penalty = 0.0

# navier_stokes.UF0 =  -navier_stokes.rho * (v_soln.fn - v_soln_1.fn) / navier_stokes.delta_t


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

# -

t_init = sympy.cos(3*th) * i_mask_fn

# +
with meshball.access(r):
    r.data[:,0]   = uw.function.evaluate(sympy.sqrt(x**2+y**2), meshball.data)  # cf radius_fn which is 0->1 

# Write density into a variable for saving

with meshball.access(t_soln):
    t_soln.data[:,0] = uw.function.evaluate(t_init, t_soln.coords)

# +
navier_stokes.bodyforce = Rayleigh * unit_rvec * t_init * i_mask_fn # minus * minus
navier_stokes.bodyforce -= free_slip_penalty # + solid_body_penalty 

# navier_stokes._Ppre_fn = 1.0 / (navier_stokes.viscosity + navier_stokes.rho / navier_stokes.delta_t + Rayleigh * surface_fn)

navier_stokes._u_star_projector.smoothing = navier_stokes.viscosity * 1.0e-6
v_proj = navier_stokes._u_star_projector.u
free_slip_penalty_p  =  10 * v_proj.fn.dot(unit_rvec) * unit_rvec * surface_fn + 100 * v_proj.fn * sky_mask_fn
navier_stokes._u_star_projector.F0 =  free_slip_penalty_p 


# +
navier_stokes.solve(timestep=10.0)

with meshball.access():
    v_inertial = v_soln.data.copy()
    
with swarm.access(v_star, remeshed, X_0):
    v_star.data[...] = uw.function.evaluate(v_soln.fn, swarm.data) 
    X_0.data[...] = swarm.data[...] 

# -

swarm.advection(v_soln.fn, 
                delta_t=navier_stokes.estimate_dt(),
                corrector=False)


# +
# check the mesh if in a notebook / serial

def plot_V_mesh(filename):
    
    if uw.mpi.size==1:

        import pyvista as pv
        import vtk

        pv.global_theme.background = 'white'
        pv.global_theme.window_size = [1250, 1250]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = 'panel'
        pv.global_theme.smooth_shading = True

        pvmesh = pv.read_meshio("ignore_celestial.vtk")

        with meshball.access():
            pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn*mask_fn, meshball.data)
            pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn*mask_fn, meshball.data)


        with meshball.access():
            usol  = uw.function.evaluate(navier_stokes.u.fn * mask_fn,
                                         navier_stokes.u.coords) 

        arrow_loc = np.zeros((navier_stokes.u.coords.shape[0],3))
        arrow_loc[:,0:2] = navier_stokes.u.coords[...]

        arrow_length = np.zeros((navier_stokes.u.coords.shape[0],3))
        arrow_length[:,0:2] = usol[...] 

        pl = pv.Plotter()
        pl.camera.SetPosition(0.0001,0.0001,4.0)

        # pl.add_mesh(pvmesh,'Black', 'wireframe')
        pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="P",
                      use_transparency=False, opacity=0.5)
        pl.add_arrows(arrow_loc, arrow_length, mag=0.03)
        
        pl.screenshot(filename="{}.png".format(filename), window_size=(2560,2560),
                      return_img=False)
        
        pl.close()
        
        del(pl)


# -


ts = 0
swarm_loop = 5

for step in range(0,50):
    
    Omega_0 = 100.0 * min(ts/30, 1.0) 
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
    
    # _,z_ns,_,_,_,_,_ = meshball.stats(v_soln.fn.dot(v_rbm_z))
    # print("Rigid body: {}".format(z_ns))

    
    dv_fn = v_soln.fn - v_soln_1.fn
    _,_,_,_,_,_,deltaV = meshball.stats(dv_fn.dot(dv_fn))


    with meshball.access(v_soln_1):
        v_soln_1.data[...] = v_soln.data[...] 

    with swarm.access(v_star):
        v_star.data[...] = 0.5 * uw.function.evaluate((v_soln.fn + v_star.fn) * mask_fn, swarm.data)  

    swarm.advection(v_soln.fn, 
                    delta_t=delta_t,
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
        
    navier_stokes._u_star_projector.smoothing = navier_stokes.viscosity * 1.0e-3
    
    ts += 1





# +
# check the mesh if in a notebook / serial


if uw.mpi.size==1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [1280, 640]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True
    
    pvmesh = pv.read_meshio("ignore_celestial.vtk")

    
    with meshball.access():
        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn*mask_fn, meshball.data)
        pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn*mask_fn, meshball.data)
        pvmesh.point_data["S"] = uw.function.evaluate(surface_fn, meshball.data)


    coriolis_term = -sympy.vector.cross(Omega, v_theta) 
    modified_coriolis_term = coriolis_term 

    with meshball.access():
        usol  = uw.function.evaluate(navier_stokes.u.fn * mask_fn,
                                     navier_stokes.u.coords) 
        corio = uw.function.evaluate(modified_coriolis_term,
                                     navier_stokes.u.coords) 

    arrow_loc = np.zeros((navier_stokes.u.coords.shape[0],3))
    arrow_loc[:,0:2] = navier_stokes.u.coords[...]
    
    arrow_length = np.zeros((navier_stokes.u.coords.shape[0],3))
    arrow_length[:,0:2] = usol[...] 
    
    pl = pv.Plotter(window_size=[1000,1000])
    
    

    # pl.add_mesh(pvmesh,'Black', 'wireframe')
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S",
                  use_transparency=False, opacity=0.5)
    pl.add_arrows(arrow_loc, arrow_length, mag=0.00050)
    
    pl.show(cpos="xy")
# -


meshball.stats(sympy.vector.cross(Omega, v_soln.fn).dot(sympy.vector.cross(Omega, v_soln.fn)))

meshball.stats(v_soln.fn.dot(v_rbm_z))

meshball.stats(v_soln.fn.dot(v_soln.fn))

meshball.stats( v_soln.fn.dot(v_rbm_z))

meshball.stats((v_soln.fn+0.015*v_rbm_z).dot(v_rbm_z))

meshball.stats(sympy.vector.cross(Omega, v_soln.fn).dot(v_rbm_z))

sympy.vector.cross(Omega, v_soln.fn)

# +
_,z_ns,_,_,_,_,_ = meshball.stats(v_soln.fn.dot(v_rbm_z))
print("Rigid body: {}".format(z_ns))



# -

p_soln.stats()


