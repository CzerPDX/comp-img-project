# This file is for the DispersionImg class
# The DispersionImg class takes in a rawpy-compatible Raw image dispersed by a prism
# It then gathers and manages the initial information and processes it and then attempts to undisperse the image

# References: 
# https://numpy.org/doc/stable/reference/arrays.indexing.html#basic-slicing-and-indexing

import numpy as np
import rawpy
import cv2
import matplotlib.pyplot as plt
import os


# Takes in a rawpy-compatible Raw dispersed image
class DispersionImg:
    def __init__(self, imgLocation, maxDimensionPx):
        self.imgError = None                    # Whether or not there is an error in the image processing process
        self.imgLocation = None                 # Image location on the computer
        self.rawImg = None                      # Raw image as numpy array
        self.parameters = None                  # Parameters taken from rawpy object
        self.processedImg = None                # Processed image from RAW image (often creates RGB) as numpy array
        self.smallerImg = None                  # Smaller image for processing easier as numpy array
        self.maxDimensionPx = maxDimensionPx    # Max in either dimension for smaller image
        self.rawChannels = {}                   # Will hold the raw, unaligned input from each type of image sensor separated into 1/4 size numpy arrays
        self.manuallyDemosaicedRaw = None       #Image is demosaiced manually into each channel keeping raw values

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

    def displayRawChannels(self):
        # Display the RAW data from each photosite
        # Add Green2 channel to upper left to match rgbg Bayer pattern of:
        # G2 B
        # R G1
        plt.suptitle(f'RAW photosite by sensor type for {self.parameters["color_desc"]} Bayer pattern', fontweight = 'bold')
        plt.subplot(2, 2, 1)
        plt.title('Green2')
        plt.imshow(self.channels['green2'], cmap='Greens')

        plt.subplot(2, 2, 2)
        plt.title('Blue')
        plt.imshow(self.channels['blue'], cmap='Blues')

        plt.subplot(2, 2, 3)
        plt.title('Red')
        plt.imshow(self.channels['red'], cmap='Reds')

        plt.subplot(2, 2, 4)
        plt.title('Green1')
        plt.imshow(self.channels['green1'], cmap='Greens')

        plt.show() 
        
    # Overwrites class data
    def resetImg(self, imgLocationUnNormed):
        self.imgError = False

        # Normalize the path for whatever operating system
        imgLocation = os.path.normpath(imgLocationUnNormed)
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
                # Eventually this should use getattr() so it can accommodate other camers besides mine
                # but I'm working with this unsafe version right now because it's safe for my camera!
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
                    'raw_pattern': rawImg.raw_pattern,
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

                # Separate the raw data into channels by photosite type (ie all Reds in one, Blues in another, etc...)
                self.__manualDemosaicKeepRawData()

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
        
    # Separate each photosite on the image sensor into its own channel based on the Bayer pattern
    # This will only work for b'RGBG' bayer pattern for now. 
    # Future work will attempt to expand this area to more Bayer patterns programmatically
    # This is used temporarily for getting the proof of concept out
    def __manualDemosaicKeepRawData(self):

        try:
            self.channels = {}
            self.channels['green2'] = self.rawImg[0::2, 0::2].copy()   # From every other row starting at row 0, choose every other pixel starting at 0
            self.channels['blue'] = self.rawImg[0::2, 1::2].copy()     # From every other row starting at row 0, choose every other pixel starting at 1
            self.channels['red'] = self.rawImg[1::2, 0::2].copy()      # From every other row starting at row 1, choose every other pixel starting at 0
            self.channels['green1'] = self.rawImg[1::2, 1::2].copy()   # From every other row starting at row 1, choose every other pixel starting at 1
            
            print('Photosites successfully separated into their own images under "channels" object!')

        except Exception as err:
            raise Exception(f'Error! Failed to demosaic raw data:\n{err}')


def testDispersionImg():
    imgLocation = 'img/DSC_5984.NEF'
    
    dispersedImgObj = DispersionImg(imgLocation, 512)

    dispersedImgObj.displayRawChannels()

    # # Print the information about the object to the console
    # dispersedImgObj.printImageInformation()

    # print(dispersedImgObj.parameters['raw_colors'].shape)



testDispersionImg()