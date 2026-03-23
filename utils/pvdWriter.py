# import 'with' statement (needed for Jython 2.5, can be removed with Jython 2.7)
from __future__ import with_statement

pvdHeader = """<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1">
  <Collection>
"""

pvdFooter = """  </Collection>
</VTKFile>
"""

def writePvdHeader(f):
	f.write(pvdHeader)

def writePvdFooter(f):
	f.write(pvdFooter)

def addVtuFile(f, t, name):
	f.write("<DataSet timestep=\"%d\" part=\"0\" file=\"%s_t%03d000000.vtu\" />\n" %(t, name, t))

def writePvdFile(filename, name, times):
	with open(filename, 'w') as f:
		writePvdHeader(f)
		for t in times:
			addVtuFile(f, t, name)
		writePvdFooter(f)
