# References 
# https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
# https://pypi.org/project/rawpy/0.3.5/

# This code mostly comes from:
# https://letmaik.github.io/rawpy/api/rawpy.RawPy.html

from PIL import Image
import rawpy
import imageio
from matplotlib import pyplot as plt

def returnNumpyFromNEF(nefLocation):
  # Read the raw image and its metadata into the software
  rawImg = rawpy.imread(nefLocation)
  # Extract the raw image (it should be in numpy array format)
  numpyImg = rawImg.raw_image
  
  return numpyImg


# if __name__ == "__main__":
def testing():
  imgLocation = 'DSC_5875.NEF'
  
  # Read the raw image file into the software
  rawImg = rawpy.imread(imgLocation)

  try:
    thumb = rawImg.extract_thumb()
  except rawpy.LibRawNoThumbnailError:
    print('no thumbnail found')
  except rawpy.LibRawUnsupportedThumbnailError:
    print('unsupported thumbnail')
  else:
    if thumb.format == rawpy.ThumbFormat.JPEG:
      with open('thumb.jpg', 'wb') as f:
        f.write(thumb.data)
    elif thumb.format == rawpy.ThumbFormat.BITMAP:
      imageio.imsave('thumb.tiff', thumb.data)

  # Get some data from the raw image
  # Raw image data in numpy format
  numpyImg = rawImg.raw_image
  print(numpyImg)
  print(f'Image dimensions: {numpyImg.shape}')

  # Black level per channel
  print(f'Black levels per channel: {rawImg.black_level_per_channel}')

  # Print the number of colors in the image
  print(f'Number of colors in image: {rawImg.num_colors}')
  print(rawImg.raw_type)

  # Raw data in default cmap mode
  plt.axis('off')
  plt.title('Rawpy test of raw image data')
  plt.imshow(numpyImg)
  plt.show()

  # Output to normal rgb using postprocess
  processedImg = rawImg.postprocess()
  plt.imshow(processedImg)
  plt.show()
