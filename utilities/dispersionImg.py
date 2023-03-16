# This file is for the DispersionImg class
# The DispersionImg class takes in a rawpy-compatible Raw image dispersed by a prism
# It then gathers and manages the initial information and processes it and then attempts to undisperse the image

import numpy as np
import rawpy
import cv2
import matplotlib.pyplot as plt


# Takes in a rawpy-compatible Raw dispersed image
class DispersionImg:
    def __init__(self, imgLocation):
        self.imgError = None            # Whether or not there is an error in the image processing process
        self.imgLocation = None         # Image location on the computer
        self.rawImg = None              # Raw image as numpy array
        self.parameters = None          # Parameters taken from rawpy object
        self.processedImg = None        # Processed image from RAW image (often creates RGB) as numpy array
        self.smallerImg = None          # Smaller image for processing easier as numpy array
        self.maxDimensionPx = 600       # Max in either dimension for smaller image

        # Save image information
        self.resetImg(imgLocation)


    # Public Methods ###############################

    # Displays a window with the smaller version of the processed image in it
    def displaySmallerImg(self):
        if self.smallerImg is None:
            print('Error! Cannot display smaller image as it has not been generated')
        else:
            plt.imshow(self.smallerImg)
            plt.title(f'Smaller processed image: {self.smallerImg.shape}')
            plt.show()
        
    # Overwrites class data
    def resetImg(self, imgLocation):
        self.imgError = False
        print()
        print(f'Loading image information at location: {imgLocation}')
        self.imgLocation = imgLocation
        try:
            # Open the raw image. It will close on its own after going out of scope due to the "with"
            # Get information about the image
            with rawpy.imread(self.imgLocation) as rawImg:

                # Save the raw image
                self.rawImg = rawImg.raw_image.copy()

                # Save the rawpy metadata
                self.parameters = {
                    'raw_type': rawImg.raw_type,
                    'black_level_per_channel': rawImg.black_level_per_channel,
                    'camera_white_level_per_channel': rawImg.camera_white_level_per_channel,
                    'camera_whitebalance': rawImg.camera_whitebalance,
                    'color_desc': rawImg.color_desc,
                    'color_matrix': rawImg.color_matrix,
                    'daylight_whitebalance': rawImg.daylight_whitebalance,
                    'num_colors': rawImg.num_colors,
                    'raw_colors': rawImg.raw_colors,
                    'raw_image_visible': rawImg.raw_image_visible.copy(),
                    'rgb_xyz_matrix': rawImg.rgb_xyz_matrix,
                    'sizes': rawImg.sizes,
                    'tone_curve': rawImg.tone_curve,
                    'white_level': rawImg.white_level
                }

                # Process the image and save the processed version (often this makes an RGB image)
                self.processedImg = rawImg.postprocess()

                # Make a smaller image to work with, if either dimension is larger than 512 pixels
                self.smallerImg = self.__reduceProcessedImg()

            if self.imgError is False:
                print('Image data successfully Loaded!')

        except Exception as err:
            print(f'Error! Unable to load image information: {err}')
            self.imgError = True

    # Prints all the information gathered from the image to the terminal.
    def printImageInformation(self):
        if self.imgError is False:
            # Image dimensions
            print()
            print(f'Raw Image Dimensions: {self.rawImg.shape}')
            print(f'Processed Image Dimensions: {self.processedImg.shape}')
            print(f'Smaller version Dimensions: {self.smallerImg.shape}')
            print()
            print('Raw Image Parameters:')
            for parameterName in self.parameters:
                print(f'{parameterName}')
                print(f'{self.parameters[parameterName]}')
                print()
        else:
            print('Error! Unable to print image information as there was an error when loading this image has not been properly loaded')


    # Private Methods ################################

    # Returns a reduced-size image if the image is larger than the maxPixelSize
    # Otherwise just points to the original processed image
    def __reduceProcessedImg(self):

        try:
            # Point to the original processed image and set initial height and width value
            height = self.processedImg.shape[0]
            width = self.processedImg.shape[1]

            # Reduce by 1/2 until under maxPixelSize to help there be less artifacting
            while (height > self.maxDimensionPx) or (width > self.maxDimensionPx):
                height = int(height * 0.5)
                width = int(width * 0.5)

            # Return the resized image using openCV
            retImg = cv2.resize(self.processedImg, (width, height), interpolation=cv2.INTER_AREA)
            print(f'Image successfully reduced to size ({height}, {width})')
            return retImg
        except Exception as err:
            raise(f'Error! Could not reduce size of processed image: {err}')



