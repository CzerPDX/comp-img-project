import rawpyTests
import alignWavelengths
import numpy as np
from matplotlib import pyplot as plt

# Violet: 380–450 nm (688–789 THz frequency)
# Blue: 450–495 nm.
# Green: 495–570 nm.
# Yellow: 570–590 nm.
# Orange: 590–620 nm.
# Red: 620–750 nm (400–484 THz frequency)

# floor of (upper - lower)/2 + lower of above wavelength ranges from wikipedia
# https://en.wikipedia.org/wiki/Visible_spectrum#Spectral_colors
red = 690
orange = 605
yellow = 580
green = 532
blue = 472
violet = 415

wavelengths = [red, orange, yellow, green, blue, violet]

testImg = rawpyTests.returnNumpyFromNEF('DSC_5911.NEF')
# plt.imshow(testImg)
# plt.show()

# The ray deviation for the prism is 60 degrees. We need to input in radians
theta = np.deg2rad(60)
# The refractive index of optical glass is about 1.5
refractiveIndex = 1.5

alignedImg = alignWavelengths.align_wavelengths(testImg, theta, wavelengths)

print('Unaltered raw numpy array:')
print(testImg)

print()
print('Aligned raw numpy array:')
print(alignedImg)

print(alignedImg.shape)
plt.imshow(alignedImg, cmap='gray')
plt.show()