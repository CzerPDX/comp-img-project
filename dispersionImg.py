# This file is for the DispersionImg class
# The DispersionImg class takes in a rawpy-compatible Raw image dispersed by a prism
# It then gathers and manages the initial information and processes it and then attempts to undisperse the image

# References: 
# https://numpy.org/doc/stable/reference/arrays.indexing.html#basic-slicing-and-indexing
# https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html
# https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html

import numpy as np
import rawpy
import cv2
import matplotlib.pyplot as plt
import os
import math
from scipy import signal
from scipy import datasets
from scipy import stats as st
from tqdm import tqdm
from scipy.ndimage import zoom
from skimage import exposure
import collections


# Takes in a rawpy-compatible Raw dispersed image
class DispersionImg:
    def __init__(self, imgLocation, maxDimensionPx):
        self.imgError = None                    # Whether or not there is an error in the image processing process
        self.imgLocation = None                 # Image location on the computer
        self.rawImg = None                      # Raw image as numpy array
        self.parameters = None                  # Parameters taken from rawpy object
        self.processedImg = None                # Processed image from RAW image (often creates RGB) as numpy array
        self.processedImg_smaller = None        # Smaller raw image for processing easier as numpy array
        
        self.maxDimensionPx = maxDimensionPx    # Max in either dimension for smaller image
        self.rawChannels = {}                   # Will hold the raw, unaligned input from each type of image sensor separated into 1/4 size numpy arrays
        self.rawChannels_smaller = {}           # Smaller versions of the rawChannels that makes alignment more reasonable
        self.rawChannels_smaller_uint8 = {}     # 8-bit version of the rawChannels_smaller
        self.manuallyDemosaicedRaw = None       #Image is demosaiced manually into each channel keeping raw values
        self.reductionPercent = 0.25            # Percentage to reduce the raw channels by as a float
        self.correlationValues = {}
        self.shiftedImgs = {}
        self.stackedTest = None
        self.outputImgsFolder = os.path.normpath('img/v2_dispersed_output/')
        self.rawShiftAmounts = {}               # How much to shift each raw channel image

        # Save image information
        self.resetImg(imgLocation)


    # Overwrites class data/Initializes class information
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
                self.processedImg_smaller = self.__reduceProcessedImg()

                # Separate the raw data into channels by photosite type (ie all Reds in one, Blues in another, etc...)
                self.__manualDemosaicKeepRawData()

                # Reduce the size of the demosaiced raw channels so they can be aligned more reasonably
                self.__makeSmallerRawChannels(self.reductionPercent)
                
                # Convert smaller raw images into 8-bit images to work with
                for channel in self.rawChannels_smaller:
                    self.rawChannels_smaller_uint8[channel] = cv2.convertScaleAbs(self.rawChannels_smaller[channel], alpha=(255.0/16383.0))

                # Align the raw channels
                # self.__alignRawChannels()
                # Combine the green into one channel
                print('starting sift alignment test')
                self.__siftAlignmentTest('green1')

            if self.imgError is False:
                print('Image data successfully Loaded!')

        except Exception as err:
            print(f'Error! Unable to load image information: {err}')
            self.imgError = True




    # Public Methods ###############################

    # Displays a window with the smaller version of the processed image in it
    def displayprocessedImg_smaller(self):
        if self.processedImg_smaller is None:
            print('Error! Cannot display smaller image as it has not been generated')
        else:
            plt.imshow(self.processedImg_smaller)
            plt.title(f'Smaller processed image: {self.processedImg_smaller.shape}')
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
        


    # Prints all the information gathered from the image to the terminal.
    def printImageInformation(self):
        if self.imgError is False:
            # Image dimensions
            print()
            print(f'Raw Image Dimensions: {self.rawImg.shape}')
            print(f'Processed Image Dimensions: {self.processedImg.shape}')
            print(f'Smaller version Dimensions: {self.processedImg_smaller.shape}')
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
            self.rawChannels = {}
            self.rawChannels['green2'] = self.rawImg[0::2, 0::2].copy()   # From every other row starting at row 0, choose every other pixel starting at 0
            self.rawChannels['blue'] = self.rawImg[0::2, 1::2].copy()     # From every other row starting at row 0, choose every other pixel starting at 1
            self.rawChannels['red'] = self.rawImg[1::2, 0::2].copy()      # From every other row starting at row 1, choose every other pixel starting at 0
            self.rawChannels['green1'] = self.rawImg[1::2, 1::2].copy()   # From every other row starting at row 1, choose every other pixel starting at 1
            
            print('Photosites successfully separated into their own images under "channels" object!')

        except Exception as err:
            raise Exception(f'Error! Failed to demosaic raw data:\n{err}')
        
    # Aligning the raw channels will take a million years if you don't make them smaller, unfortunately.
    # reductionPercent should be represented as float value
    def __makeSmallerRawChannels(self, reductionPercent):
        print(f'Creating smaller raw channel versions for alignment')
        try:
            for channel in self.rawChannels:
                smallerSize = zoom(self.rawChannels[channel], (reductionPercent, reductionPercent), order = 3)

                # Increase the contrast to allow better alignment in future steps
                # clipLimit = 2.0 means the contrast is limited to twice the original contrast
                # tileGridSize
                contrastSettings = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
                contrastedImg = contrastSettings.apply(smallerSize)

                #Now smooth it a little to reduce noise
                kernelSize = 3
                smoothedImg = cv2.GaussianBlur(contrastedImg, (kernelSize, kernelSize), 0)

                self.rawChannels_smaller[channel] = smoothedImg
        except Exception as err:
            raise Exception(f'Error! Failed to demosaic raw data:\n{err}')

    # Takes in two tuples (ogCenter and convolvedCenter) and returns the shift value to align the images
    def __getCorrelationShiftValues(self, ogCenter, convolvedCenter):
        try:
            # Get the size shift for the actual raw images, not the smaller versions so multiply by 1/reduction percent
            rowShift = (1 / self.reductionPercent) * (convolvedCenter[0] - ogCenter[0])
            colShift = (1 / self.reductionPercent) * (-1 * (convolvedCenter[1] - ogCenter[1]))
        except Exception as err:
            raise Exception(f'Error! Failed to calculate image shift:\n{err}')
        
        return (rowShift, colShift)

    def __correlateChannels(self, mainChannel, secondaryChannel, secondaryChannelName):
        # Replace NaN values with zeros so it doesn't error if there is invalid data
        mainChannel = np.nan_to_num(mainChannel)
        secondaryChannel = np.nan_to_num(secondaryChannel)

        # signal.correlate2d returns an array of correlations between the first img and the second img based on shifting the image around
        # The index of the highest correlation in the output will be used to calculate the (row, col) shift that we need to do on the second image
        correlatedImgs = np.zeros_like(mainChannel).astype(np.float32)
        for i in tqdm(range(mainChannel.shape[0])):
            correlatedImgs[i,:] = signal.fftconvolve(mainChannel[i:i+1,:], secondaryChannel[::-1,::-1], mode='same')
        
        # Find the index of the highest value in the correlatedImgs array. Returns as a flat index.
        # (Flat indicies are returned as a single number. Imagine the array is transformed into a 1d array with all 
        # rows starting at the end of the last row. The flat index is the number in that array.)
        maxCorrelationFlatIdx = np.argmax(correlatedImgs)

        # Translate the flat index into a tuple which we can use to calculate the shift value in (rows, col) format.
        correlatedPos = np.unravel_index(maxCorrelationFlatIdx, correlatedImgs.shape)

        # Get center of the main image (using floor) into a tuple and use that with the correlatedPos to get the shift amount for the layer
        og_rows = (mainChannel.shape[0] // 2)
        og_col = (mainChannel.shape[1] // 2)
        ogCenter = (og_rows, og_col)
        shiftAmt = self.__getCorrelationShiftValues(ogCenter, correlatedPos)
        
        # Save the value to the class data
        self.correlationValues[secondaryChannelName] = shiftAmt

    # Takes in a set of matches and a verticalThresholdPercent and returns only those that are vertically similar
    # Our prism should only be dispersing in the x-direction, so we know matches that are vertically similar to each other
    def __maskMatches(self, imgShape, matches, keypoints1, keypoints2, verticalThresholdPercent, horizontalThresholdPercent):
        
        # Filter the matches based on vertical displacement
        goodMatches = []

        for m, n in matches:
            # Get the coordinates of the matched keypoints
            pt1 = keypoints1[m.queryIdx].pt
            nearestNeighborTo_pt1 = keypoints2[m.trainIdx].pt         # Nearest neighbor to pt1
            secondNearestNeighborTo_pt1 = keypoints2[n.trainIdx].pt         # Second nearest neighbor to pt1

            # Get the vertical displacement between the keypoints
            # Displacement between point1 and nearestNeighborTo_pt1 
            displacement_y1 = abs(pt1[1] - nearestNeighborTo_pt1[1])    
            # Displacement between point1 and secondNearestNeighborTo_pt1
            displacement_y2 = abs(pt1[1] - secondNearestNeighborTo_pt1[1])

            # Displacement between point1 and nearestNeighborTo_pt1 
            displacement_x1 = abs(pt1[0] - nearestNeighborTo_pt1[0])    
            # Displacement between point1 and secondNearestNeighborTo_pt1
            displacement_x2 = abs(pt1[0] - secondNearestNeighborTo_pt1[0])
            

            appendMatch = False         # Whether or not we will be appending m to our goodMatches
            # Figure out if the match is vertically similar enough to go on our goodMatches list
            # If both y1 and y2 displacements are less than the allowed percentage of the image height, it will be considered a good match
            if (displacement_y1 < (verticalThresholdPercent * imgShape[0])) and (displacement_y2 < (verticalThresholdPercent * imgShape[0])):
                # Now figure out if the matches are horizontally similar enough
                if (displacement_x1 < (horizontalThresholdPercent * imgShape[1])) and (displacement_x2 < (horizontalThresholdPercent * imgShape[1])):
                    appendMatch = True
                        
            # Only append if it's good enough to append
            if (appendMatch):
                goodMatches.append((m, n)) 

        return goodMatches


    # References for this section:
    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    # https://www.mathworks.com/discovery/ransac.html
    # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    # https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object

    # Gives shift values for the image channels using SIFT

    # Eventually it would be best if this didn't even take in any arguments and instead,
    # dynamically finds the number of channels to align based on the contents of self.rawChannels.
    # This change will allow it to function for larger numbers of raw channels.
    # but in the interest of getting a functional version out, we are doing it explicitly here using
    # red, green1, green2, and blue channels

    # Get the shiftValues for the x-direction using keypoints and matches
    def __getKeypointShiftValues(sef, matches, keypoints1, keypoints2):
        shiftValues = []
        for match in matches:
            keypoint1_x = keypoints1[match[0].queryIdx].pt[0]
            keypoint2_x = keypoints2[match[0].trainIdx].pt[0]

            
            shiftValues.append(int(keypoint1_x) - int(keypoint2_x))

        return shiftValues
    
        
    def __siftAlignmentTest(self, mainChannelName):
        sift = cv2.SIFT_create()                # Initialize SIFT detector
        keypointsAndDescriptors = {}            # Hold keypoints and descriptors for channels
        checksToMake = 300                      # Number of checks to make in FLANN Matching

        # Set up the datastructure for the FLANN index (efficient nearest neighbor search)
        # Number of times that a candidate for the FLANN (Fast Library for Approximate Nearest Neighbor)
        # checks its nearest neighbors to make sure it has the actual nearest neighbor
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = checksToMake)
        
        # Set up the matcher
        flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Get keypoints and descriptors from 8-bit smaller versions
        for channelName in self.rawChannels_smaller_uint8:
            keypointsAndDescriptors[channelName] = sift.detectAndCompute(self.rawChannels_smaller_uint8[channelName], None)

        # Get shift amounts for each raw channel (full-size shift amounts calculated from smaller versions)
        for channelName in self.rawChannels_smaller_uint8:
            # We don't want to try to shift the mainChannel (the one we are basing the others off of)
            # We only want to try to shift the other channels in relation to mainChannel
            if (channelName.lower() == mainChannelName.lower()): 
                self.rawShiftAmounts[channelName] = 0
            else:
                # Give the smaller versions, you will get back shift values equal to the originals
                self.rawShiftAmounts[channelName] = self._getShiftValues_sift(flannMatcher,
                                                                                mainChannelName,
                                                                                self.rawChannels_smaller_uint8[mainChannelName],
                                                                                keypointsAndDescriptors[mainChannelName][0],
                                                                                keypointsAndDescriptors[mainChannelName][1],
                                                                                channelName,
                                                                                self.rawChannels_smaller_uint8[channelName],
                                                                                keypointsAndDescriptors[channelName][0],
                                                                                keypointsAndDescriptors[channelName][1])
        
        # Shift each raw channel
        for channelName in self.rawShiftAmounts:
            print(f'Shift channel {channelName} by {self.rawShiftAmounts[channelName]}')
            self.shiftedImgs[channelName] = self.__shiftImgs(self.rawChannels[channelName], self.rawShiftAmounts[channelName], 0).copy()

        # Stack the images into RGB
        greenChannel = (self.shiftedImgs['green1'] + self.shiftedImgs['green2']) / 2
        self.__combineToRGB(self.shiftedImgs['red'], greenChannel, self.shiftedImgs['blue'])
                
                                                

    def _getShiftValues_sift(self,
                            flannMatcher,
                            mainChannelName,
                            mainChannelImg_smaller_uint8, 
                            keypoints_mainChannel, 
                            descriptors_mainChannel, 
                            shiftChannelName,
                            shiftChannelImg_smaller_uint8, 
                            keypoints_shiftChannel,
                            descriptors_shiftChannel):
        # Get the matches between the descriptors of the main channel and the shift channel
        matches = flannMatcher.knnMatch(descriptors_mainChannel, descriptors_shiftChannel, k=2)

        # Filter out any matches that are too vertically or horizontally dissimilar.
        # Vertical threshold can be much lower because the dispersion from our prism is assumed to be in the x-direction.
        # Horizontal threshold can also be low, but not as low. The x-direction dispersion will not be terribly large
        verticalThresholdPercent = 0.01         # Percentage as float in allowable vertical difference between keypoints matches
        horizontalThresholdPercent = 0.05        # Percentage as float of allowable horizontal difference between keypoint matches
        matches_filtered = self.__maskMatches(mainChannelImg_smaller_uint8.shape,
                                              matches,
                                              keypoints_mainChannel,
                                              keypoints_shiftChannel,
                                              verticalThresholdPercent,
                                              horizontalThresholdPercent)
        
        # Visualize the matches made between main and shift channels as an image and save to a file
        matchImg = cv2.drawMatchesKnn(mainChannelImg_smaller_uint8,
                                      keypoints_mainChannel,
                                      shiftChannelImg_smaller_uint8,
                                      keypoints_shiftChannel,
                                      matches_filtered,
                                      None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(matchImg)
        plt.title(f'{mainChannelName} to {shiftChannelName} points')
        plt.savefig(f'{self.outputImgsFolder}\{mainChannelName}_{shiftChannelName}_matches.png')

        # Get all the shift values between mainChannel and shiftChannel
        shifts = self.__getKeypointShiftValues(matches_filtered, keypoints_mainChannel, keypoints_shiftChannel)
        # Then find the value that occurs most
        counter = collections.Counter(shifts)
        try:
            shiftAmt = int(counter.most_common(1)[0][0] * (1 / self.reductionPercent))
        except Exception as err:
            raise(f'No value appeared more than once. Try increasing the SIFT parameters to be more permissive.')
        
        return shiftAmt
        


    
    # We will try to use scipy's  correlate2d because the images will only ever be offset in one directional axis.
    def __alignRawChannels(self):
        # Find the cross-correlation between the arrays
        print(f'Beginning alignment of images using signal.correlate2d')

        # Correlate all positions to a single channel (I chose red for now)
        # Correlates the blue channel to the red channel
        print('Getting correlations for green1 channel in relation to red channel')
        self.__correlateChannels(self.rawChannels_smaller['red'], self.rawChannels_smaller['green1'], 'green1')
        print('Getting correlations for blue channel in relation to red channel')
        self.__correlateChannels(self.rawChannels_smaller['red'], self.rawChannels_smaller['blue'], 'blue')
        print('Getting correlations for green2 channel in relation to red channel')
        self.__correlateChannels(self.rawChannels_smaller['red'], self.rawChannels_smaller['green2'], 'green2')

        # Display the shift amounts
        print('Shift values:')
        for color in self.correlationValues:
            print(f'{color}: {self.correlationValues[color]}')
            print(f'Shifting {color}...')
            self.shiftedImgs[color] = self.__shiftImgs(self.rawChannels[color], self.correlationValues[color][0], self.correlationValues[color][1]).copy()

        greenChannel = (self.shiftedImgs['green1'] + self.shiftedImgs['green2']) / 2

        self.__combineToRGB(self.rawChannels['red'], greenChannel, self.shiftedImgs['blue'])
        
        
        # Transpose the images so the # of channels comes last
        # stackedRGB = np.transpose(stackedRGB, (1, 2, 0))

        

    def __combineToRGB(self, red, green, blue):
        # Run some equalization on the image since we skipped that by working with the raw data
        redEq = exposure.equalize_hist(red)
        greenEq = exposure.equalize_hist(green)
        blueEq = exposure.equalize_hist(blue)


        stackedRGB = np.stack((redEq, greenEq, blueEq), axis=-1)

        # Scale the pixel values to the range [0, 255]
        stackedRGB = (stackedRGB - stackedRGB.min()) / (stackedRGB.max() - stackedRGB.min()) * 255.0
        stackedRGB = stackedRGB.astype(np.uint8)

        # 
        plt.imshow(stackedRGB)
        plt.show()

        
    # Takes a numpy image to shift and shiftValues as a tuple with (rows, col) format
    def __shiftImgs(self, imgToShift, rowShift, colShift):

        # Shift the rows
        shiftedImg = np.roll(imgToShift, shift=rowShift, axis=0)

        # Shift the columns
        shiftedImg = np.roll(shiftedImg, shift=colShift, axis=1)

        return shiftedImg

    

def testDispersionImg():
    imgLocation = 'img/v2_dispersed.NEF'
    
    dispersedImgObj = DispersionImg(imgLocation, 512)
    # dispersedImgObj.printImageInformation()

    # dispersedImgObj.displayRawChannels()




testDispersionImg()