import numpy as np
from dolfin import *
from mshr import *
from optimisation.contours import resample

def meshPyramid(vertices, Nx, Ny, Lx, Ly, factor=1.5, min_Nx=5, max_Nx=None, resampleVertices=True):
    
    Nx_H, Ny_H = Nx, Ny

    if max_Nx:
        # do not compute on finest mesh anyway
        if Nx_H>max_Nx:
            Ny_H = Ny_H*max_Nx/Nx_H
            Nx_H = max_Nx

    meshes = []
    areas = []
    
    while Nx_H >= min_Nx:
        Dx_H = Lx/Nx_H
        Dy_H = Ly/Ny_H
        area = Dx_H*Dy_H/2.
        mesh = Mesh()
        # the cell_size parameter is an upper bound on the length of the longest edge of triangles
        #cell_size = 2.*np.sqrt(area)
        cell_size = np.sqrt(Dx_H**2 + Dy_H**2)

        if resampleVertices:
             contour_factor = np.sqrt((Nx_H*Ny_H)/(Nx*Ny))
             resampled_vertices = resample(vertices, contour_factor)
        else:
             resampled_vertices = vertices

        #generator = CSGCGALMeshGenerator2D()
        #generator.parameters['cell_size'] = cell_size
        #generator.generate(resampled_vertices, mesh)

        geo = Polygon(resampled_vertices)
        geoo = CSGCGALDomain2D(geo)
        generator = CSGCGALMeshGenerator2D()
        generator.parameters['cell_size'] = cell_size
        generator.parameters['mesh_resolution'] = 0.
        mesh = generator.generate(geoo)

#        num_refinements=1
#        if Nx_H<50:
#            for j in range(num_refinements):
#                mesh.init(1,2)
#                markers=CellFunction('bool', mesh, False)
#                for c in cells(mesh):
#                    for f in facets(c):
#                        if f.exterior():
#                            markers[c]=True
#                            break
#                mesh=refine(mesh,markers)

        #mesh = CSGCGALMeshGenerator2D(geo)
        #mesh = generate_mesh(geo, cell_size)
        #plot(mesh); interactive()
# does not work for fenics 2016.2        #PolygonalMeshGenerator.generate(mesh, resampled_vertices, cell_size)

        meshes += [mesh]
        areas += [area]

        Ny_H, Nx_H = Ny_H/factor, Nx_H/factor

    print("=> successfully generated mesh pyramid")
    return meshes, areas#meshes[:5][-2:], areas[:5][-2:]
