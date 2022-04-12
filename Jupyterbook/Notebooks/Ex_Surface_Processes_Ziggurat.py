# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: uw3-venv
#     language: python
#     name: uw3-venv
# ---

# # Surface Processes

import underworld3 as uw
import numpy as np

mesh = uw.util_mesh.Annulus(radiusInner=0., radiusOuter=7.0, cellSize=0.05)

heightvar = uw.mesh.MeshVariable( mesh=mesh, num_components=1, name="height", vtype=uw.VarType.SCALAR, degree=1 )

# ## Initialize Height Field

x = mesh.data[:,0]
y = mesh.data[:,1]

# +
from scipy.interpolate import LinearNDInterpolator

n = 10
npts = 100
theta = np.linspace(0, n * np.pi, n * npts )
s1 = 0.30 * theta 
x1 = s1 * np.cos(theta)
y1 = s1 * np.sin(theta)
s2 = 0.25 * (theta + 1e-6)
x2 = s2 * np.cos(theta)
y2 = s2 * np.sin(theta)

rmean = (s1 + s2) / 2.0
z = np.exp(-rmean**2.0 / 20)

h2 = (1.0 - s1 / s1.max()) * 500
h1 = (1.0 - s2 / s1.max()) * 550

x0 = np.hstack( [x1,x2] )
y0 = np.hstack( [y1,y2] )
h0 = np.hstack( [h1,h2] )

interp = LinearNDInterpolator(list(zip(x0, y0)), h0)
height = interp(list(zip(x,y)))

height = height + (1.0 + 0.01 * np.random.random(size=height.shape))
# -

with mesh.access(heightvar):
    heightvar.data[:,0] = height

# +
# Validate

from mpi4py import MPI

if MPI.COMM_WORLD.size==1:

    import numpy as np
    import pyvista as pv

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [500, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'pythreejs'
    pv.global_theme.smooth_shading = True
    
    mesh.vtk("mesh.tmp.vtk")
    pvmesh = pv.read("mesh.tmp.vtk")

    with mesh.access(heightvar):
        pvmesh.point_data["height"]  = heightvar.data[:]
    
    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="height",
                  use_transparency=False, opacity=0.5)
    
    pl.camera_position="xy"
     
    pl.show(cpos="xy")
    # pl.screenshot(filename="test.png")  
# -

import os
savefile = "Ziggurat.h5" 
mesh.save(savefile)
heightvar.save(savefile)
mesh.generate_xdmf(savefile)
