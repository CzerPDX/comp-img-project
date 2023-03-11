# This file is for testing the Spectral Cube library

import os
from utilities.dispersionImg import DispersionImg

# Normalize the path for whatever operating system
imgLocation = os.path.normpath('img/DSC_5984.NEF')
dispersedNEF = DispersionImg(imgLocation)