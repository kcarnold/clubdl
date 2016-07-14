#!/usr/bin/env python
from osgeo import gdal
import numpy as np
import sys
import path

np.save(path.Path(sys.argv[1]).namebase + '.npy', gdal.Open(sys.argv[1]).ReadAsArray())
