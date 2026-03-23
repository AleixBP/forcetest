from dolfin import *
import scipy.signal

# convert from a contour in pixel units to a list of vertices in dolfin format and length units
def contour_to_vertices(contour, Dx, Dy):
    #vertices = [Point(Dx*x, Dy*(Ny-1-y)) for [x, y, z] in contour]
    vertices = [Point(Dx*x, Dy*y) for [x, y, z] in contour]
    return vertices

# resample a closed contour
def resample(vertices, factor):
    vertices_x = [vertex.x() for vertex in vertices]
    vertices_y = [vertex.y() for vertex in vertices]

    resampled_vertices_x = scipy.signal.resample(vertices_x, int(round(len(vertices_x)*factor)))
    resampled_vertices_y = scipy.signal.resample(vertices_y, int(round(len(vertices_y)*factor)))

    resampled_vertices = []
    for x, y in zip(resampled_vertices_x, resampled_vertices_y):
        resampled_vertices.append(Point(x, y))

    return resampled_vertices
