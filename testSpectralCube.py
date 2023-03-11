# This file is for separating a 3-channel RGB image into more channels

# References
# https://letmaik.github.io/rawpy/api/rawpy.RawPy.html
# https://pillow.readthedocs.io/en/stable/handbook/concepts.html
# https://pillow.readthedocs.io/en/stable/reference/ImageShow.html

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import rawpy
from utilities.dispersionImg import rawToNumpy, tifToNumpy

from utilities.dispersionImg import DispersionImg

# class RGBimg:
#     def __init__(self, imgData, sourceImgType):
#         self.image = imgData
#         self.sourceImageType = sourceImgType



# RGB Numpy from TIF file (mostly used for testing)
# Uses an interleaved TIF





def testGradient():
    imgLocation = 'img\\DSC_5984.NEF'

    

    # # Load the test gradient
    # numpyImgFromTIF = tifToNumpy('img\\rainbowGradient.tif')
    # plt.title(f'image from TIF: {numpyImgFromTIF.shape}')
    # plt.imshow(numpyImgFromTIF)
    # plt.show()

    
    # # Load the raw file
    # rawImg = rawToNumpy(imgLocation)
    # print()
    # print(f'Raw Shape: {rawImg.shape}')
    # print()
    # plt.title(f' image from NEF: {numpyImgFromNEF.shape}')
    # plt.imshow(numpyImgFromNEF)
    # plt.show()





testGradient()