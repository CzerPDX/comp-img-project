# This entire code snippet was generated using ChatGPT (Jan 30 2023 version, Free Research Preview) along with the following description:

# This code takes a 2D numpy array dispersed_image as input and returns a 2D numpy array rectified_image 
# that has the different wavelengths aligned in a single row. The idea is to find the center of the spectrum 
# in each column of the dispersed image, and then shift the pixels in that column so that the center of the 
# spectrum is at the center of the row in the rectified image.

import numpy as np

def align_wavelengths(image, theta, wavelengths):
  # Calculate the dispersion factor for each wavelength using the constant theta value
  dispersion_factor = np.sin(theta) / wavelengths
  
  # Create a 2D matrix with the dispersion factors
  dispersion_matrix = np.outer(np.ones(image.shape[0]), dispersion_factor)
  
  # Multiply each row of the input image with the dispersion matrix element-wise
  dispersed_image = image * dispersion_matrix
  
  # Find the sum of each row in the dispersed image and find the row with the maximum sum
  dispersed_row_sums = np.sum(dispersed_image, axis=1)
  max_row_index = np.argmax(dispersed_row_sums)
  
  # Shift all the rows in the dispersed image such that the row with the maximum sum is at the center of the image
  center_index = int(image.shape[0] / 2)
  row_shift = center_index - max_row_index
  dispersed_image = np.roll(dispersed_image, row_shift, axis=0)
  
  return dispersed_image






# import numpy as np
# from scipy.sparse import csr_matrix


# def align_wavelengths(image, theta, wavelengths):
#   # Calculate the dispersion factor for each wavelength using the constant theta value
#   dispersion_factor = np.sin(theta) / np.array(wavelengths)
  
#   # Create a sparse matrix with the dispersion factors
#   row_indices = np.arange(len(wavelengths))
#   col_indices = np.zeros(len(wavelengths))
#   values = dispersion_factor
#   dispersion_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(len(wavelengths), image.shape[1]))
  
#   # Multiply each row of the input image with the dispersion matrix to get the dispersed image
#   dispersed_image = np.dot(image, dispersion_matrix.toarray().T)
  
#   # Finally, we align the dispersed image by adding the maximum value of each row to each row.
#   max_dispersed = np.amax(dispersed_image, axis=1)
#   for i in range(len(max_dispersed)):
#     dispersed_image[i,:] = dispersed_image[i,:] + max_dispersed[i]
  
#   return dispersed_image
