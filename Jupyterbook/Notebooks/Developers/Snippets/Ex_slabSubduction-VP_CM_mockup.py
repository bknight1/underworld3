#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Slab subduction
#
#
# #### [From Dan Sandiford](https://github.com/dansand/uw3_models/blob/main/slabsubduction.ipynb)
#
#
#
# UW2 example ported to UW3 

# %%
import numpy as np
import os
import math
from petsc4py import PETSc
import underworld3 as uw


from underworld3.utilities import generateXdmf


from sympy import Piecewise, ceiling, Abs, Min, sqrt, eye, Matrix, Max


# %%
expt_name = 'output/slabSubduction/'

if uw.mpi.rank==0:
    ### delete previous model run
    if os.path.exists(expt_name):
        for i in os.listdir(expt_name):
            os.remove(expt_name + i)
            
    ### create folder if not run before
    if not os.path.exists(expt_name):
        os.makedirs(expt_name)


# %%
### For visualisation
render = True


# %%


options = PETSc.Options()

options["snes_converged_reason"] = None
options["snes_monitor_short"] = None



# %%
n_els     =  30
dim       =   2
boxLength = 4.0
boxHeight = 1.0
ppcell    =   5

# %% [markdown]
# ### Create mesh and mesh vars

# %%
mesh = uw.meshing.StructuredQuadBox(elementRes=(    4*n_els,n_els), 
                    minCoords =(       0.,)*dim, 
                    maxCoords =(boxLength,boxHeight) )


v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

strain_rate_inv2 = uw.discretisation.MeshVariable("SR", mesh, 1, degree=1)
node_viscosity   = uw.discretisation.MeshVariable("Viscosity", mesh, 1, degree=1)
# materialField    = uw.discretisation.MeshVariable("Material", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)





# %% [markdown]
# ### Create swarm and swarm vars
# - 'swarm.add_variable' is a traditional swarm, can't be used to map material properties. Can be used for sympy operations, similar to mesh vars.
# - 'uw.swarm.IndexSwarmVariable', creates a mask for each material and can be used to map material properties. Can't be used for sympy operations.
#

# %%
swarm  = uw.swarm.Swarm(mesh)

# %%
## # Add index swarm variable for material
material              = uw.swarm.IndexSwarmVariable("M", swarm, indices=5) 

# %%


swarm.populate(3)

# Add some randomness to the particle distribution
import numpy as np
np.random.seed(0)

with swarm.access(swarm.particle_coordinates):
    factor = 0.5*boxLength/n_els/ppcell
    swarm.particle_coordinates.data[:] += factor*np.random.rand(*swarm.particle_coordinates.data.shape)
      


# %% [markdown]
# #### Project fields to mesh vars
# Useful for visualising stuff on the mesh (Viscosity, material, strain rate etc) and saving to a grouped xdmf file


# %%
# material.info()
# """
# you have 5 materials
# if you want to have material variable rheologies, density
# """
# phi 1 = material.piecewise([m1_visc,2,3,4,5])
# phi_2 = material.piecewise([m1_rho, m2_rho, m3_rho])

# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes._Einv2
# nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
# nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")

# meshMat = uw.systems.Projection(mesh, materialField)
# meshMat.uw_function = material.sym
# # meshMat.smoothing = 1.0e-3
# meshMat.petsc_options.delValue("ksp_monitor")

def updateFields():
    ### update strain rate
    nodal_strain_rate_inv2.uw_function = stokes._Einv2
    nodal_strain_rate_inv2.solve()

    ### update viscosity
    nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
    nodal_visc_calc.solve(_force_setup=True)
    
    # ### update material field from swarm
    # meshMat.uw_function = material.sym
    # meshMat.solve(_force_setup=True)



# %% [markdown]
# ## Setup the material distribution


# %%
import matplotlib.path as mpltPath

### initialise the 'material' data to represent two different materials. 
upperMantleIndex = 0
lowerMantleIndex = 1
upperSlabIndex   = 2
lowerSlabIndex   = 3
coreSlabIndex    = 4

### Initial material layout has a flat lying slab with at 15\degree perturbation
lowerMantleY   = 0.4
slabLowerShape = np.array([ (1.2,0.925 ), (3.25,0.925 ), (3.20,0.900), (1.2,0.900), (1.02,0.825), (1.02,0.850) ])
slabCoreShape  = np.array([ (1.2,0.975 ), (3.35,0.975 ), (3.25,0.925), (1.2,0.925), (1.02,0.850), (1.02,0.900) ])
slabUpperShape = np.array([ (1.2,1.000 ), (3.40,1.000 ), (3.35,0.975), (1.2,0.975), (1.02,0.900), (1.02,0.925) ])


# %%
slabLower  = mpltPath.Path(slabLowerShape)
slabCore   = mpltPath.Path(slabCoreShape)
slabUpper  = mpltPath.Path(slabUpperShape)


# %% [markdown]
# ### Update the material variable of the swarm

# %%
with swarm.access(swarm.particle_coordinates, material):

    ### for the symbolic mapping of material properties
    material.data[:] = upperMantleIndex
    material.data[swarm.particle_coordinates.data[:,1] < lowerMantleY]           = lowerMantleIndex
    material.data[slabLower.contains_points(swarm.particle_coordinates.data[:])] = lowerSlabIndex
    material.data[slabCore.contains_points(swarm.particle_coordinates.data[:])]  = coreSlabIndex
    material.data[slabUpper.contains_points(swarm.particle_coordinates.data[:])] = upperSlabIndex
    
    
    


# %%
def plot_mat():

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True


    mesh.vtk("tempMsh.vtk")
    pvmesh = pv.read("tempMsh.vtk") 

    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]
        points[:,2] = 0.0

    point_cloud = pv.PolyData(points)


    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()



    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)       



    pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="M",
                        use_transparency=False, opacity=0.95)



    pl.show(cpos="xy")
 
if render == True:
    plot_mat()


# %% [markdown]
# ### Function to save output of model
# Saves both the mesh vars and swarm vars

# %%
def saveData(step, outputPath):

    mesh.petsc_save_checkpoint(meshVars=[v, p, strain_rate_inv2, node_viscosity], index=step, outputPath=outputPath)
    
    swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=outputPath)
    


# %% [markdown]
# #### Density

# %%
mantleDensity = 0.5
slabDensity   = 1.0 

density_fn = material.createMask([mantleDensity, 
                                 mantleDensity,
                                 slabDensity,
                                 slabDensity,
                                 slabDensity])





stokes.bodyforce =  Matrix([0, -1 * density_fn])


# %% [markdown]
# ### Boundary conditions
#
# Free slip by only constraining one component of velocity 

# %%
#free slip
stokes.add_dirichlet_bc( (0.,0.), 'Left',   (0) ) # left/right: function, boundaries, components
stokes.add_dirichlet_bc( (0.,0.), 'Right',  (0) )

stokes.add_dirichlet_bc( (0.,0.), 'Top',    (1) )
stokes.add_dirichlet_bc( (0.,0.), 'Bottom', (1) )# top/bottom: function, boundaries, components 

# %% [markdown]
# ###### initial first guess of constant viscosity

# %%
if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

stokes.petsc_options["snes_max_it"] = 500

stokes.tolerance = 1e-6

# %%
### initial linear solve
stokes.constitutive_model.Parameters.viscosity  = 1.

stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity

stokes.solve(zero_init_guess=True)


# %% [markdown]
# #### add in NL rheology for solve loop
#
# CM-VP mockup

# %%
### viscosity from UW2 example
upperMantleViscosity =    1.0
lowerMantleViscosity =  100.0
slabViscosity        =  500.0
coreViscosity        =  500.0


strainRate_2ndInvariant = stokes._Einv2



# %%
import sympy

# %%
### set background/initial viscoisity
shear_viscosity_0 = sympy.Matrix([upperMantleViscosity, lowerMantleViscosity, slabViscosity, slabViscosity, coreViscosity])

# %%
### define yield stress
yield_stress = sympy.Matrix([0,0,0.06, 0.06, 0])
yield_visc   = 0.5 * (yield_stress / (stokes._Einv2 + 1.0e-18))

# %%
example_yield_visc = 0.06 / (2*stokes._Einv2+1.0e-18)
example_shear_visc_0 = 500.

example_viscosity_HA = 1. / ((1./example_yield_visc) + (1./example_shear_visc_0))

example_viscosity_Min = sympy.Min(example_yield_visc, example_shear_visc_0)



# %%
example_viscosity_HA

# %%
example_viscosity_Min

# %%
averaging_method = 'min'

# %%
#### create the viscosity for each material
viscosity = []

if averaging_method.casefold() == 'min':
    for i in range(len(yield_visc)):
        # print(yield_visc[i])
        if yield_visc[i] != 0:
            viscosity.append(sympy.Min(yield_visc[i], shear_viscosity_0[i]))
        else:
            viscosity.append(shear_viscosity_0[i])   
else: ### Think the harmonic average is best as the default due to better solvability
    for i in range(len(yield_visc)):
        # print(yield_visc[i])
        if yield_visc[i] != 0:
            viscosity.append( 1./((1./yield_visc[i])+ (1./shear_viscosity_0[i]) ) )
        else:
            viscosity.append(shear_viscosity_0[i])
        
    
    

# %%
viscosity_fn = material.sym.T.dot( sympy.Matrix( viscosity ) )

# %%
viscosity_fn


# %%
stokes.constitutive_model.Parameters.viscosity = viscosity_fn

# %%
stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity

# %% [markdown]
# ### Main loop
# Stokes solve loop

# %%
step      = 0
max_steps = 50
time      = 0


#timing setup
#viewer.getTimestep()
#viewer.setTimestep(1)


while step<max_steps:
    
    print(f'\nstep: {step}, time: {time}')
          
    #viz for parallel case - write the hdf5s/xdmfs 
    if step%10==0:
        if uw.mpi.rank == 0:
            print(f'\nSave data: ')
            
        ### updates projection of fields to the mesh
        updateFields()
        
        ### saves the mesh and swarm
        saveData(step, expt_name)
        

            
    
    if uw.mpi.rank == 0:
        print(f'\nStokes solve: ')  
        
    stokes.solve(zero_init_guess=False)
    
    ### get the timestep
    dt = stokes.estimate_dt()
 
    ### advect the particles according to the timestep
    swarm.advection(V_fn=stokes.u.sym, delta_t=dt, corrector=False)
        
    step += 1
    
    time += dt




# %%
