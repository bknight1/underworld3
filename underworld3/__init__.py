from petsc4py import PETSc
#PETSc.Log().begin()
import underworld3.mesh
import underworld3.meshes
import underworld3.maths
import underworld3.swarm
import underworld3.systems
import underworld3.tools
import underworld3.algorithms
import underworld3.mpi

from enum import Enum as _Enum
class VarType(_Enum):
    SCALAR=1
    VECTOR=2
    OTHER=3  # add as required 


import numpy as _np
# Info for JIT modules.
# These dicts should be populated by submodules
# which define cython/c based classes.
# We use ordered dictionaries because the
# ordering can be important when linking in libraries.
# Note that actually what we want is an ordered set (which Python
# doesn't natively provide). Hence for the key/value pair,
# the value is always set to `None`.
from collections import OrderedDict as _OD
_libfiles = _OD()
_libdirs  = _OD()
_incdirs  = _OD({_np.get_include():None})
