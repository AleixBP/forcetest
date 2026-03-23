from dolfin import *
import numpy as np

#import scitools

#def to_array(mesh, a, Nx, Ny):
#    a2 = a if a.ufl_element().degree() == 1 else project(a, FunctionSpace(mesh, 'Lagrange', 1))
#    a_box = scitools.BoxField.dolfin_function2BoxField(a2, mesh, (Nx,Ny), uniform_mesh=True)
#    return a_box.values

# y axis is reversed in Dolfin compared to Numpy:
# 0 at bottom for Dolfin, at top for Numpy
def fix_ysign(a):
    return -a

# transforms an image in a matrix form into a dolfin function defined on a rectangular mesh.
# the mesh has the same number of square divisions than the image has pixels. As the dolfin function values
# are ordered in a vector fashion we just have to charge the pixel values repeating them as there are two triangles per square.
def array_to_scalar(mesh, a):
    VI = FunctionSpace(mesh, "DG", 0)
    A = Function(VI)
    
    if False:
        vals = a.flatten().repeat(2).astype(np.float32)

        dofmap = VI.dofmap()                                                             
        my_first, my_last = dofmap.ownership_range()      

        unowned = dofmap.local_to_global_unowned()
        dofs = list(filter(lambda dof: dofmap.local_to_global_index(dof) not in unowned, 
                    range(my_last-my_first)))
        
        A.vector().set_local(vals[dofs])
        A.vector().apply('insert')

    elif False:
        A.vector().set_local(a.flatten().repeat(2).astype(np.float32))
    else:
        A.vector()[:] = a.flatten().repeat(2).astype(np.float32)

#alsoworks    A.vector.set_local(a.flatten().repeat(2).astype(np.float32))
   
    return A

def scalar_to_array(dest_mesh, Nx, Ny, v):
    Vu = FunctionSpace(dest_mesh, "DG", 0)

    if True:
        a = project(v, Vu).vector().get_local()
        va = 0.5*(a[::2] + a[1::2]).reshape((Ny, Nx))

    return va

# same thing as array to scalar but the two values of each vector are represented as consecutive values in the vector form
# of the dolphin function
def arrays_to_vector(mesh, ax, ay):
    VgradI = VectorFunctionSpace(mesh, "DG", 0)
    A = Function(VgradI)
    
    b = np.empty(VgradI.dim())
    b[::2] = ax.flatten().repeat(2).astype(np.float32)
    b[1::2] = ay.flatten().repeat(2).astype(np.float32)

    if False:
        
        dofmap = VgradI.dofmap()                                                             
        my_first, my_last = dofmap.ownership_range()      

        unowned = dofmap.local_to_global_unowned()
        dofs = list(filter(lambda dof: dofmap.local_to_global_index(dof) not in unowned, 
                    range(my_last-my_first)))
        
        A.vector().set_local(b[dofs])
        A.vector().apply('insert')
    elif False: 
        A.vector().set_local(b)
        
    else:
        A.vector()[:] = b
        
    #could also try forcing one stram in main as
    #from mpi4py import MPI
    #comm = MPI.COMM_WORLD
    #if comm.rank == 0:
    #https://fenicsproject.discourse.group/t/running-a-parallel-sub-routine-within-the-main-routine-which-is-run-in-series/5581
    
    return A

def vector_to_arrays(dest_mesh, Nx, Ny, v):
    Vu = FunctionSpace(dest_mesh, "DG", 0)

    vx, vy = v.split()
    ax = project(vx, Vu).vector().get_local()
    vx = 0.5*(ax[::2] + ax[1::2]).reshape((Ny, Nx))
    ay = project(vy, Vu).vector().get_local()
    vy = 0.5*(ay[::2] + ay[1::2]).reshape((Ny, Nx))
    #vy = fix_ysign(vy)

    return vx, vy
