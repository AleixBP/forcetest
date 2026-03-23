import os, errno

def make_sure_path_exists(path):
   try:
       os.makedirs(path)
   except OSError, exception:
       if exception.errno <> errno.EEXIST:
           raise
