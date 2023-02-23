# References 
# https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
# https://pypi.org/project/rawpy/0.3.5/

# This code mostly comes from:
# https://letmaik.github.io/rawpy/api/rawpy.RawPy.html

# import numpy as np
import rawpy

def returnNumpyFromNEF(nefLocation):
  # Read the raw image and its metadata into the software
  rawImg = rawpy.imread(nefLocation)
  # Extract the raw image (it should be in numpy array format)
  numpyImg = rawImg.raw_image
  
  return numpyImg

# Get the minimum and maximum pixel values for a numpy image
def getMinMaxPixelValues(numpyImg):
  min_pixel_value = numpyImg.min()
  max_pixel_value = numpyImg.max()

  return min_pixel_value, max_pixel_value

# Get the metadata from a raw image using rawpy
def printMetadata(rawImg):
  # Descriptive comments in this section are mostly from https://letmaik.github.io/rawpy/api/rawpy.RawPy.html

  # Raw Type (the type of raw image we are working with)
  rawType = rawImg.raw_type
  print(rawType)
  
  # Raw Pattern
  # The smallest possible Bayer pattern of this image. Only usable for flat RAW images.
  print()
  if rawType.name != 'Flat':
    print(f'raw_pattern can only be used on flat RAW images. This images is format: {rawType.name}')
  else:
    print('The smallest possible Bayer pattern for this image is:')
    print(rawImg.raw_pattern)

  # Color Description
  # String description of colors numbered from 0 to 3 (RGBG,RGBE,GMCY, or GBTG).
  # Note that same letters may not refer strictly to the same color. There are cameras with two different greens for example.
  print()
  print('String description of colors numbered from 0 to 3 (RGBG,RGBE,GMCY, or GBTG) used in the Bayer pattern of the camera.')
  print('Note that same letters may not refer strictly to the same color. There are cameras with two different greens for example.')
  print()
  print(f'Color description is: {rawImg.color_desc}')
  
  # Raw Colors
  # An array of color indices for each pixel in the RAW image. 
  # Equivalent to calling raw_color(y,x) for each pixel. Only usable for flat RAW images (see raw_type property).
  # Each pixel will either be red, green, or blue (Often there are 2 greens)
  print()
  if rawType.name != 'Flat':
    print(f'raw_colors can only be used on flat RAW images. This images is format: {rawType.name}')
  else:
    print('The color indices for each pixel in the RAW image.')
    print(rawImg.raw_colors)

  # Color Matrix
  # read from file for some cameras, calculated for others.
  print()
  print('The color matrix for this image is:')
  print(rawImg.color_matrix)
  

# if __name__ == "__main__":
def readRawImg():
  imgLocation = 'DSC_5984.NEF'
  
  # Read the raw image file into the software using rawpy
  # The "with" format closes the file automatically releases access when finished.
  with rawpy.imread(imgLocation) as rawImg:
    printMetadata(rawImg)


  # try:
  #   thumb = rawImg.extract_thumb()
  # except rawpy.LibRawNoThumbnailError:
  #   print('no thumbnail found')
  # except rawpy.LibRawUnsupportedThumbnailError:
  #   print('unsupported thumbnail')
  # else:
  #   if thumb.format == rawpy.ThumbFormat.JPEG:
  #     with open('thumb.jpg', 'wb') as f:
  #       f.write(thumb.data)
  #   elif thumb.format == rawpy.ThumbFormat.BITMAP:
  #     imageio.imsave('thumb.tiff', thumb.data)

readRawImg()