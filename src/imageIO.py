# Handles the reading and writing of image files to the disk
# It will prepare the RAW data into numpy arrays and will pull out the metadata
# It will handle loading data to the disk
# It will also handle saving data to the disk

import rawpy
import os
import matplotlib.pyplot as plt
import numpy as np

class imageIO:
    def __init__(self, outputImagesFolder, imgLocation):
        self.imgError = None                          # Whether or not there is an error in image processing

        self.__imgLocation = None                     # Image location on the computer
        self.__rawImg = None                          # Raw image as numpy array
        self.__parameters = None                      # Parameters taken from rawpy object

        self.__imgLocation = imgLocation              # Location of input image
        self.__outputImgsFolder = outputImagesFolder  # Location of output images

        # Initialize the image data
        self.__initializeData(self, self.__imgLocation)        

    # Overwrites class data/Initializes class information
    def __initializeData(self, imgLocationUnNormed):
        self.imgError = False

        # Normalize the path for whatever operating system
        imgLocation = os.path.normpath(imgLocationUnNormed)
        print()
        print(f'Loading image information at location: {imgLocation}')
        
        self.__imgLocation = imgLocation
        try:
            # Open the raw image. It will close on its own after going out of scope due to the "with"
            # Get information about the image
            with rawpy.imread(self.__imgLocation) as rawImg:

                # Save the raw image
                self.__rawImg = rawImg.raw_image.copy()

                # Save the rawpy metadata
                # Eventually this should use getattr() so it can accommodate other camers besides mine
                # but I'm working with this unsafe version right now because it's safe for my camera!
                self.__parameters = {
                    'raw_type': rawImg.raw_type,
                    'black_level_per_channel': rawImg.black_level_per_channel,
                    'camera_white_level_per_channel': rawImg.camera_white_level_per_channel,
                    'camera_whitebalance': rawImg.camera_whitebalance,
                    'color_desc': rawImg.color_desc,
                    'color_matrix': rawImg.color_matrix,
                    'daylight_whitebalance': rawImg.daylight_whitebalance,
                    'num_colors': rawImg.num_colors,
                    'raw_colors': rawImg.raw_colors,
                    'raw_pattern': rawImg.raw_pattern,
                    'raw_image_visible': rawImg.raw_image_visible.copy(),
                    'rgb_xyz_matrix': rawImg.rgb_xyz_matrix,
                    'sizes': rawImg.sizes,
                    'tone_curve': rawImg.tone_curve,
                    'white_level': rawImg.white_level
                }

            if self.imgError is False:
                print('Image data successfully Loaded!')

        except Exception as err:
            print(f'Error! Unable to load image information: {err}')
            self.imgError = True

    def saveFig(self, incomingFig, filename):
        plt.savefig(f'{self.__outputImgsFolder}\\6-channel-multispectral-approximation.png')

