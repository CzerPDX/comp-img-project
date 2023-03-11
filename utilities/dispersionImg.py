# References
# https://towardsdatascience.com/raw-image-processing-in-python-238e5d582761

import numpy as np
import rawpy
import cv2


# Takes in a rawpy-compatible Raw dispersed image
class DispersionImg:
    def __init__(self, imgLocation):
        self.imgError = False
        self.imgLocation = imgLocation
        self.rawImg = None
        self.rawType = None
        self.parameters = None
        self.processedImg = None
        self.maxDimensionPx = 512

        # Save image location information
        self.resetImg(imgLocation)
        
    # Overwrites class object data with that at the imgLocation if the 
    def resetImg(self, imgLocation):
        print()
        print(f'Loading image information at location: {imgLocation}')
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
                self.reduceProcessedImg(self.maxDimensionPx)
            
            # Print image information to terminal to confirm initialization to user
            self.printImageInformation()

        except Exception as err:
            print(f'Error! Unable to load image information: {err}')
            self.imgError = True

    # Prints all the information gathered from the image to the terminal.
    def printImageInformation(self):
        if self.imgError != True:
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

    # Returns a reduced-size image if the image is larger than the maxPixelSize
    # Otherwise just points to the original processed image
    def reduceProcessedImg(self, maxPixelSize):
        # Point to the original processed image and set initial height and width values
        self.smallerImg = self.processedImg
        height = self.processedImg.shape[0]
        width = self.processedImg.shape[1]

        # Only resize if the image is larger than the max pixel size for the smallest version
        if (height > maxPixelSize) or (width > maxPixelSize):
            # Set new height and width sizes
            if height > width:
                width = int(width * (maxPixelSize/height))
                height = maxPixelSize
            else:
                height = int(height * (maxPixelSize/width))
                width = maxPixelSize

            # Resize the image using openCV
            self.smallerImg = cv2.resize(self.processedImg, (height, width), interpolation=cv2.INTER_AREA)



imgLocation = 'img\\DSC_5984.NEF'
dispersedNEF = DispersionImg(imgLocation)



# def rawToNumpy(rawImgLocation):
#     # Open the RAW image. Using "with" closes the file after
#     with rawpy.imread(rawImgLocation) as rawImg:
#         rawNumpy = rawImg.raw_image.copy()

#         return rawNumpy

# def tifToNumpy(imgLocation):
#     # Load TIF
#     tifImg = Image.open(imgLocation)
#     # Convert to RGB
#     rgbImg = tifImg.convert('RGB')
#     return (np.asarray(rgbImg))