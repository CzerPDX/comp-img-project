# This file is for the DispersionImg class
# The DispersionImg class takes in a rawpy-compatible Raw image dispersed by a prism
# It then gathers and manages the initial information and processes it and then attempts to undisperse the image

# References: 
# https://numpy.org/doc/stable/reference/arrays.indexing.html#basic-slicing-and-indexing
# https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html
# https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
# https://www.dynamsoft.com/blog/insights/image-processing/image-processing-101-color-space-conversion/

import numpy as np
import rawpy
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy.ndimage import zoom
from skimage import exposure




# Takes in a rawpy-compatible Raw dispersed image
class DispersionImg:
    def __init__(self, 
                 outputImagesFolder, 
                 imgLocation, 
                 reductionPercent, 
                 colorWeights,
                 verticalThresholdPercent,
                 horizontalThresholdPercent,
                 gaussianKernelSize,
                 clipLimit,
                 tileGridSize
                 ):
        self.imgError = None                        # Whether or not there is an error in the image processing process
        self.imgLocation = None                     # Image location on the computer
        self.rawImg = None                          # Raw image as numpy array
        self.parameters = None                      # Parameters taken from rawpy object
        
        self.imgLocation = imgLocation              # Location of input image
        self.outputImgsFolder = outputImagesFolder  # Location of output images

        self.processedImg = None                    # Processed image from RAW image (often creates RGB) as numpy array
        self.processedImg_smaller = None            # Smaller raw image for processing easier as numpy array
        
        self.rawChannels = {}                       # Will hold the raw, unaligned input from each type of image sensor separated into 1/4 size numpy arrays
        self.rawChannels_smaller = {}               # Smaller versions of the rawChannels that makes alignment more reasonable
        self.rawChannels_smaller_uint8 = {}         # 8-bit version of the rawChannels_smaller
        self.manuallyDemosaicedRaw = None           # Image is demosaiced manually into each channel keeping raw values
        self.shiftedImgs = {}                       # Raw channels after shifting

        self.reductionPercent = reductionPercent    # Percentage to reduce the raw channels by as a float
        self.verticalThresholdPercent = verticalThresholdPercent        # Percentage as float in allowable vertical difference between keypoints matches
        self.horizontalThresholdPercent = horizontalThresholdPercent    # Percentage as float of allowable horizontal difference between keypoint matches        
        self.gaussianKernelSize = gaussianKernelSize
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

        self.rawShiftAmounts = {}                   # How much to shift each raw channel image
        self.calculatedHyperspectral = {}           
        self.colorWeights = colorWeights            # Color weight tuples in format (r, g, b). Keys accessed by nanometer

        # Save image information
        self.resetImg(self.imgLocation)


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
                self.rawChannels_smaller = self.__makeSmallerChannels(self.rawChannels, )

                self.rawChannels_smaller_uint8 = self.__getSmallerUint8(self.rawChannels_smaller)

                # The image sensor in a Nikon D90 and Nikon D5000 are the same
                # I was unable to find the spectral response data for the Nikon D5000, but there is a useful
                # Researchgate paper that details the spectral response of the D90. I will use this below
                # To get a better estimate of the colorweights
                
                
                # Start sift alignment test. Aligning all layers to green1
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

            plt.cla()   # Clear axis
            plt.clf()   # Clear figure
            plt.close() # Close a figure window

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

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window



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

    def displaycalculatedHyperspectral(self):
        # Added this because I was having issues with artifacts from previous figures.
        # This is a messy but temporary solution
        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window

        # Some points we will decide are the representations of these colors
        red = 690
        orange = 590
        yellow = 570
        green = 530
        blue = 470
        violet = 410

        # Define the custom yellow colormap for white to yellow because there doesn't seem to be one like for 'Blues' and 'Reds' etc
        yellowCmap = LinearSegmentedColormap.from_list(
            'yellows',
            [
                (1.0, 1.0, 1.0),   # Pure white
                (1.0, 1.0, 0.0)    # Pure yellow
            ],
        )

        # create figure
        fig = plt.figure(figsize=(10, 5))

        # setting values to rows and column variables
        rows = 2
        columns = 3

        plt.suptitle(f'Approximate 6-channel Multispectral Images', fontweight = 'bold')
        
        # Create red plot
        fig.add_subplot(rows, columns, 1)
        plt.imshow(self.calculatedHyperspectral[red], cmap='Reds')
        plt.axis('off')
        plt.title("Red")
        
        # Create orange plot
        fig.add_subplot(rows, columns, 2)
        plt.imshow(self.calculatedHyperspectral[orange], cmap='Oranges')
        plt.axis('off')
        plt.title("Orange")
        
        # Create yellow plot
        fig.add_subplot(rows, columns, 3)
        plt.imshow(self.calculatedHyperspectral[yellow], cmap=yellowCmap)
        plt.axis('off')
        plt.title("Yellow")
        
        # Create green plot
        fig.add_subplot(rows, columns, 4)
        plt.imshow(self.calculatedHyperspectral[green], cmap='Greens')
        plt.axis('off')
        plt.title("Green")

        # Create blue plot
        fig.add_subplot(rows, columns, 5)
        plt.imshow(self.calculatedHyperspectral[blue], cmap='Blues')
        plt.axis('off')
        plt.title("Blue")

        # Create violet plot
        fig.add_subplot(rows, columns, 6)
        plt.imshow(self.calculatedHyperspectral[violet], cmap='Purples')
        plt.axis('off')
        plt.title("Violet")
        
        # plt.show()
        plt.savefig(f'{self.outputImgsFolder}\\6-channel-multispectral-approximation.png')

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window


    # Private Methods ################################

    # Returns a reduced-size image if the image is larger than the maxPixelSize
    # Otherwise just points to the original processed image
    def __reduceProcessedImg(self):

        try:
            # Point to the original processed image and set initial height and width value
            height = int(self.processedImg.shape[0] * self.reductionPercent)
            width = int(self.processedImg.shape[1] * self.reductionPercent)

            # Return the resized image using openCV
            retImg = cv2.resize(self.processedImg, (width, height), interpolation=cv2.INTER_AREA)
            print(f'Image successfully reduced to size ({height}, {width})')
        except Exception as err:
            raise(f'Error! Could not reduce size of processed image: {err}')
        
        
        return retImg
        
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
    def __makeSmallerChannels(self, channelsDict):
        print(f'Creating {self.reductionPercent} channel versions for alignment')
        channelsDict_smaller = {}
        try:
            for channel in channelsDict:
                smallerSize = zoom(channelsDict[channel], (self.reductionPercent, self.reductionPercent), order = 3)

                # Increase the contrast to allow better alignment in future steps
                # clipLimit = 2.0 means the contrast is limited to twice the original contrast
                # tileGridSize
                contrastSettings = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
                contrastedImg = contrastSettings.apply(smallerSize)

                #Now smooth it a little to reduce noise
                smoothedImg = cv2.GaussianBlur(contrastedImg, (self.gaussianKernelSize, self.gaussianKernelSize), 0)

                channelsDict_smaller[channel] = smoothedImg
        except Exception as err:
            raise Exception(f'Error! Failed to reduce channel sizes:\n{err}')
        
        return channelsDict_smaller


    # Takes in a set of matches and a verticalThresholdPercent and returns only those that are vertically and horizontally similar
    # Our prism should only be dispersing in the x-direction, so we know matches that are vertically similar to each other
    def __maskMatches(self, imgShape, matches, keypoints1, keypoints2, verticalThresholdPercent, horizontalThresholdPercent):
        
        # Filter the matches based on vertical displacement
        goodMatches = []            # Good matches we will be sending back
        ratioThreshold = 0.75       # Threshold for Lowe's ratio test
        appendMatch = False         # Whether or not we will be appending m to our goodMatches

        # m = nearest neighbor
        # n = second-nearest neighbor
        # matches are in (col, row) aka (x, y) format
        for match in matches:
            # Get the coordinates of the matched keypoints
            # match[0].queryIdx = index of the matched point from keypoints1
            # match[0].trainIdx = index of the nearest neighbor matched point from keypoints2
            # match[1].trainIdx = index of the second-nearest neighbor matched point from keypoints2
            # Use the above to dereference the keypoints1 and keypoints2

            # # First perform Lowe's ratio test
            # if match[0].distance < (ratioThreshold * match[1].distance):
                
            pt1 = keypoints1[match[0].queryIdx].pt
            nearestNeighborTo_pt1 = keypoints2[match[0].trainIdx].pt         # Nearest neighbor to pt1
            # secondNearestNeighborTo_pt1 = keypoints2[match[1].trainIdx].pt   # Second nearest neighbor to pt1

            # Get the vertical displacement between the keypoints
            # Displacement between point1 and nearestNeighborTo_pt1 
            displacement_y1 = pt1[1] - nearestNeighborTo_pt1[1]   
            # Displacement between point1 and secondNearestNeighborTo_pt1
            # displacement_y2 = pt1[1] - secondNearestNeighborTo_pt1[1]

            # Displacement between point1 and nearestNeighborTo_pt1 
            displacement_x1 = pt1[0] - nearestNeighborTo_pt1[0]
            # Displacement between point1 and secondNearestNeighborTo_pt1
            # displacement_x2 = pt1[0] - secondNearestNeighborTo_pt1[0]
            
    
            absDisplacement_x1 = abs(displacement_x1)
            # absDisplacement_x2 = abs(displacement_x2)
            absDisplacement_y1 = abs(displacement_y1)
            # absDisplacement_y2 = abs(displacement_y2)
            maxVerticalDisp = (verticalThresholdPercent * imgShape[0])
            maxHorizontalDisp = (horizontalThresholdPercent * imgShape[1])
            # Figure out if the match is vertically similar enough to go on our goodMatches list
            # If both y1 and y2 displacements are less than the allowed percentage of the image height, it will be considered a good match
            if (absDisplacement_y1 < maxVerticalDisp):
                # Now figure out if the matches are horizontally similar enough
                if (absDisplacement_x1 < maxHorizontalDisp):
                    appendMatch = True
                        
            # Only append if it's good enough to append
            if (appendMatch):
                goodMatches.append((match[0], match[0])) 
                appendMatch = False

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

    # Get the shiftValues for the x (col) and y (row) directions using keypoints and matches
    # Matches are in the (col, row) aka (x, y) format
    # Returns (col, row)/(x, y) format as well
    def __getKeypointShiftValues(self, matches, keypoints1, keypoints2):
        shiftValues_col = []
        shiftValues_row = []
        
        # Get all the shift values between mainChannel and shiftChannel
        try:
            for match in matches:
                keypoint1_col = keypoints1[match[0].queryIdx].pt[0]
                keypoint2_col = keypoints2[match[0].trainIdx].pt[0]
                shift_col = int(keypoint1_col) - int(keypoint2_col)
                shiftValues_col.append(shift_col)

                keypoint1_row = keypoints1[match[0].queryIdx].pt[1]
                keypoint2_row = keypoints2[match[0].trainIdx].pt[1]
                shift_row = int(keypoint1_row) - int(keypoint2_row)
                shiftValues_row.append(shift_row)
        except Exception as err:
            raise Exception(f'Error! Failed to get keypoint shift values:\n{err}')
        
        # Then find the value that occurs most in each direction

        # x direction / cols
        try:
            shiftAmt_col = int(self.__getMostCommonOrMediumValue(shiftValues_col) * (1 / (self.reductionPercent)))
        except Exception as err:
            raise(f'No value appeared more than once in the x-direction. Try udpating the SIFT parameters to be more permissive.')
        
        
        # y direction / rows
        try:
            shiftAmt_row = int(self.__getMostCommonOrMediumValue(shiftValues_row) * (1 / (self.reductionPercent)))
        except Exception as err:
            raise(f'No value appeared more than once in the x-direction. Try udpating the SIFT parameters to be more permissive.')

        
        # Return (col, row)/(x, y) shift
        return (shiftAmt_col, shiftAmt_row)
    
    # Reference
    # https://www.freecodecamp.org/news/sort-dictionary-by-value-in-python
    def __getMostCommonOrMediumValue(self, intArray):
        retValue = 0
        counts = {}

        # If there are no entries in intArray
        if len(intArray) < 1:
            raise Exception(f'Error! No entries in the intArray')
        # Otherwise there is at least 1 entry so start counting!
        else:
            for number in intArray:
                # If the number already exists as a key in counts dictionary, increment it
                if number in counts:
                    counts[number] += 1
                # Otherwise we need to initialize the entry to 1 count
                else:
                    counts[number] = 1

        # If there is only one entry in counts
        if (len(counts) == 1):
            retValue = intArray[0]
        # Otherwise there are multiple entries in counts
        else:
            # Sort the keys from lowest count to highest because we will need it sorted in a minute
            sortedCounts = dict(sorted(counts.items(), key=lambda item: item[0]))

            # Use list comprehension and max() to get the keys that correspond with the highest values
            maxValue = max(sortedCounts.values())
            maxKeys = [key for key, value in sortedCounts.items() if value == maxValue]

            # Now we know all the numbers tied for max and they are sorted in order
            lengthOfMaxKeys = len(maxKeys)

            # If there's only one entry
            if (lengthOfMaxKeys == 1):
                retValue = maxKeys[0]
            # Otherwise there are multiple entries we need to find the tie-breaker
            else:
                # If the number of maxKeys is odd
                if (lengthOfMaxKeys % 2) != 0:
                    retValue = maxKeys[lengthOfMaxKeys // 2]
                else:
                    retValue = maxKeys[lengthOfMaxKeys // 2 - 1]

        
        return retValue

    
    def __getSmallerUint8(self, channelsDict):
        channels_uint8 = {}
        try:
            # Convert images into 8-bit images to work with
            for channel in channelsDict:
                channels_uint8[channel] = cv2.convertScaleAbs(channelsDict[channel], alpha=(255.0/16383.0))
        except Exception as err:
            raise Exception(f'Error! Failed to create uint8 images:\n{err}')
        
        return channels_uint8

    
        
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
                self.rawShiftAmounts[channelName] = (0, 0)
            else:
                # Give the smaller versions, you will get back shift values equal to the original size
                # flannMatcher returns values of (col, row) aka (x, y)
                # __getShiftValues_sift also returns values of (col, row)/(x, y)
                self.rawShiftAmounts[channelName] = self._getShiftValues_sift(flannMatcher,
                                                                                mainChannelName,
                                                                                self.rawChannels_smaller_uint8[mainChannelName],
                                                                                keypointsAndDescriptors[mainChannelName][0],
                                                                                keypointsAndDescriptors[mainChannelName][1],
                                                                                channelName,
                                                                                self.rawChannels_smaller_uint8[channelName],
                                                                                keypointsAndDescriptors[channelName][0],
                                                                                keypointsAndDescriptors[channelName][1])
        # # SIMPLY FOR TESTING FOR THE LOVE OF GOD PLS REMOVE THIS
        # self.rawShiftAmounts['red'] = (self.rawShiftAmounts['red'][0], 0)
        # self.rawShiftAmounts['green1'] = (self.rawShiftAmounts['green1'][0], 0)

        # self.rawShiftAmounts['green2'] = (self.rawShiftAmounts['green2'][0], 0)
        # self.rawShiftAmounts['blue'] = (self.rawShiftAmounts['blue'][0], 0)

        # Shift each raw channel by (col, row)/(x, y) in self.rawShiftAmounts[channelName]
        for channelName in self.rawShiftAmounts:
            print(f'Shift channel {channelName} by {self.rawShiftAmounts[channelName][0]} cols and {self.rawShiftAmounts[channelName][1]} rows')
            self.shiftedImgs[channelName] = self.__shiftImgs(self.rawChannels[channelName], self.rawShiftAmounts[channelName])

        greenChannel = (self.shiftedImgs['green1'] + self.shiftedImgs['green2']) / 2
        # Create a simple RGB image of the 3 raw channels in alignment
        self.__combineToRGB(self.shiftedImgs['red'], greenChannel, self.shiftedImgs['blue'])

        # self.__makecalculatedHyperspectral(self.shiftedImgs)
        self.__makeHyperspectral(self.shiftedImgs)

    # Ok now we are going to try slicing these up into more channels.
    # We will be making some assumptions here about the amounts of red, green, and blue that will be in the image

    def __makeHyperspectral(self, channelDict):
        # Slice up the colors by weight:
        for wavelengthRange in self.colorWeights:
            self.calculatedHyperspectral[wavelengthRange] = self.__approximateIntermediateColors(self.colorWeights[wavelengthRange])

        self.displaycalculatedHyperspectral()


        
        
    # Gets a 6-channel multispectral approximation
    def __approximateIntermediateColors(self, colorWeights):
        greenChannel = (self.shiftedImgs['green1'] + self.shiftedImgs['green2']) / 2
        return (self.shiftedImgs['red'] * colorWeights[0]) + (greenChannel * colorWeights[1]) + (self.shiftedImgs['blue'] * colorWeights[2])       

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
        # Matches come back as (col, row) aka (x, y) from the flannMatcher
        matches = flannMatcher.knnMatch(descriptors_mainChannel, descriptors_shiftChannel, k=1)

        # Filter out any matches that are too vertically or horizontally dissimilar.
        # Vertical threshold can be much lower because the dispersion from our prism is assumed to be in the x-direction.
        # Horizontal threshold can also be low, but not as low. The x-direction dispersion will not be terribly large
        matches_filtered = self.__maskMatches(mainChannelImg_smaller_uint8.shape,
                                              matches,
                                              keypoints_mainChannel,
                                              keypoints_shiftChannel,
                                              self.verticalThresholdPercent,
                                              self.horizontalThresholdPercent)
        
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

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window

        
        return self.__getKeypointShiftValues(matches_filtered, keypoints_mainChannel, keypoints_shiftChannel)
        

    def __combineToRGB(self, red, green, blue):
        # Run some equalization on the image since we skipped that by working with the raw data
        redEq = exposure.equalize_hist(red)
        greenEq = exposure.equalize_hist(green)
        blueEq = exposure.equalize_hist(blue)


        stackedRGB = np.stack((redEq, greenEq, blueEq), axis=-1)

        # Scale the pixel values to the range [0, 255]
        stackedRGB = (stackedRGB - stackedRGB.min()) / (stackedRGB.max() - stackedRGB.min()) * 255.0
        stackedRGB = stackedRGB.astype(np.uint8)

        # Save the image
        plt.title(f'3-Channel image stacked to RGB')
        plt.imshow(stackedRGB)
        plt.savefig(f'{self.outputImgsFolder}\\3-channel-stacked-RGB.png')

        plt.cla()   # Clear axis
        plt.clf()   # Clear figure
        plt.close() # Close a figure window

        
    # Takes a numpy image to shift and shiftValues as a tuple with (col, row) format
    def __shiftImgs(self, imgToShift, colRowShiftTuple):
        # colRowShiftTuple is in format (col, row)/(x, y)
        rowShift = colRowShiftTuple[1]          # Number of rows to shift the image
        colShift = colRowShiftTuple[0]          # Number of columns to shift the image
        rowAxis = 0             # Row/x axis is 0 for np.roll
        colAxis = 1             # Col/y axis is 1 for np.roll


        # Shift the rows
        shiftedImg = np.roll(imgToShift, shift=rowShift, axis=rowAxis)

        # Shift the columns
        shiftedImg = np.roll(shiftedImg, shift=colShift, axis=colAxis)

        return shiftedImg
    