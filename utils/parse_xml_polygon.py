if False:
    from xml.etree import cElementTree as ET

    poly_file = "/home/boquet/tests/hypothesis_testing/polygon8TFM.xml"

    from pathlib import Path
    txt = Path(poly_file).read_text()
    root = ET.fromstring(txt)


    tree = ET.parse(poly_file)
    root = tree.getroot()
    
    #### Read Icy xml into Python array of vertices
    file_path = "/home/boquet/tests/hypothesis_testing/"
    file_name = "polygon8TFMb"
    npy_from_polygon_xml(file_path, file_name)

def npy_from_polygon_xml(file_path, file_name):
    poly_file = file_path + file_name + ".xml"
    from pathlib import Path
    txt = Path(poly_file).read_text()

    from bs4 import BeautifulSoup
    y = BeautifulSoup(txt)

    y.points.findAll("pos_y")
    y.points.findAll("pos_y")[0].contents


    poys = []
    for poy in y.points.findAll("pos_y"):
        poys.append(float(poy.contents[0]))
        
    poxs = []
    for pox in y.points.findAll("pos_x"):
        poxs.append(float(pox.contents[0]))

    import numpy as np
    Nx = 800.; Ny=800.
    poss = np.array([poxs,poys])/np.array([Nx,Ny])[:,np.newaxis]

    import matplotlib.pyplot as plt    
    plt.plot(*poss)

    np.save(file_path + file_name + ".npy", poss)

    #### Turn vertices into mesh
    import numpy as np

    poss_loaded = np.load(file_path + file_name + ".npy")
    plt.plot(*poss_loaded)
    
    return 0


def mesh_from_polygon_npy(verticesnpy, cell_size):
    import numpy as np
    import mshr
    import dolfin as dl
    poly = np.load(verticesnpy)[:,::-1] #change clockwiseness, otherwise mshr.Polygon throws cryptic error
    vertices = [dl.Point(x, y) for [x, y] in poly.T]
    #print(vertices[0].x(), vertices[0].y(), poly[:,0])
    geo = mshr.Polygon(vertices) 
    geoo = mshr.CSGCGALDomain2D(geo, 0.)
    generator = mshr.CSGCGALMeshGenerator2D()
    generator.parameters['cell_size'] = cell_size
    generator.parameters['mesh_resolution'] = 0.
    mesh = generator.generate(geoo, [(1, geoo)] )
    return mesh
    
    
#poly_mesh = mesh_from_polygon_npy("/home/boquet/tests/hypothesis_testing/polygon8TFM.npy", 1./50)
#dl.plot(poly_mesh)