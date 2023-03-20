# I had a hard time figuring out the output from fftconvolve. 
# I did a lot of testing to figure out how to get a shfit, but I suspect there is still an easier way
# https://stackoverflow.com/questions/24768222/how-to-detect-a-shift-between-images

import numpy as np
from scipy import signal
import math

def testFftConvolve():
    img_1x1 = np.array([
        [0]
    ])

    img_2x2 = np.array([
        [0, 0],
        [1, 0]
    ])

    img_3x3 = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])

    img_4x4 = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    img_5x5 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])

    img_6x6 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]
    ])

    img_7x7 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])
    # Rows x columns
    img_7x4 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    

    corr_1x1 = signal.fftconvolve(img_1x1, img_1x1[::-1,::-1], mode='same')
    center_1x1 = np.unravel_index(np.argmax(corr_1x1), corr_1x1.shape)

    corr_2x2 = signal.fftconvolve(img_2x2, img_2x2[::-1,::-1], mode='same')
    center_2x2 = np.unravel_index(np.argmax(corr_2x2), corr_2x2.shape)

    corr_3x3 = signal.fftconvolve(img_3x3, img_3x3[::-1,::-1], mode='same')
    center_3x3 = np.unravel_index(np.argmax(corr_3x3), corr_3x3.shape)

    corr_4x4 = signal.fftconvolve(img_4x4, img_4x4[::-1,::-1], mode='same')
    center_4x4 = np.unravel_index(np.argmax(corr_4x4), corr_4x4.shape)

    corr_5x5 = signal.fftconvolve(img_5x5, img_5x5[::-1,::-1], mode='same')
    center_5x5 = np.unravel_index(np.argmax(corr_5x5), corr_5x5.shape)

    corr_6x6 = signal.fftconvolve(img_6x6, img_6x6[::-1,::-1], mode='same')
    center_6x6 = np.unravel_index(np.argmax(corr_6x6), corr_6x6.shape)

    corr_7x7 = signal.fftconvolve(img_7x7, img_7x7[::-1,::-1], mode='same')
    center_7x7 = np.unravel_index(np.argmax(corr_7x7), corr_7x7.shape)

    corr_7x4 = signal.fftconvolve(img_7x4, img_7x4[::-1,::-1], mode='same')
    center_7x4 = np.unravel_index(np.argmax(corr_7x4), corr_7x4.shape)

    print()
    print(f'1x1:    {center_1x1},   center of original img: ({math.floor(img_1x1.shape[0] / 2), math.floor(img_1x1.shape[1] / 2)} ')
    print(f'2x2:    {center_2x2},   center of original img: ({math.floor(img_2x2.shape[0] / 2), math.floor(img_2x2.shape[1] / 2)} ')
    print(f'3x3:    {center_3x3},   center of original img: ({math.floor(img_3x3.shape[0] / 2), math.floor(img_3x3.shape[1] / 2)} ')
    print(f'4x4:    {center_4x4},   center of original img: ({math.floor(img_4x4.shape[0] / 2), math.floor(img_4x4.shape[1] / 2)} ')
    print(f'5x5:    {center_5x5},   center of original img: ({math.floor(img_5x5.shape[0] / 2), math.floor(img_5x5.shape[1] / 2)} ')
    print(f'6x6:    {center_6x6},   center of original img: ({math.floor(img_6x6.shape[0] / 2), math.floor(img_6x6.shape[1] / 2)} ')
    print(f'7x7:    {center_7x7},   center of original img: ({math.floor(img_7x7.shape[0] / 2), math.floor(img_7x7.shape[1] / 2)} ')
    print(f'7x4:    {center_7x4},   center of original img: ({math.floor(img_7x4.shape[0] / 2), math.floor(img_7x4.shape[1] / 2)} ')
    print()

    # Results of testing:
    # Even-numbered arrays will get an image center that is the floor of the array size divided by 2 in each direction.
    # However, it's important to note that numpy displays arrays in a rows x column format, which caused me considerable confusion at first!!!

    # 1 is at (1, 3)
    img1 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    # down 2 and right 1
    img2 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ])

    # up 2 and left 1
    img3 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    # up 2 and left 0
    img4 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])



    correlation2 = signal.fftconvolve(img1, img2[::-1,::-1], mode='same')
    img2Center = np.unravel_index(np.argmax(correlation2), correlation2.shape)

    correlation3 = signal.fftconvolve(img1, img3[::-1,::-1], mode='same')
    img3Center = np.unravel_index(np.argmax(correlation3), correlation3.shape)

    correlation4 = signal.fftconvolve(img1, img4[::-1,::-1], mode='same')
    img4Center = np.unravel_index(np.argmax(correlation4), correlation4.shape)

    og_rows = (img1.shape[0] // 2)
    og_col = (img1.shape[1] // 2)
    

    print(f'img1 base center:   ({og_rows}, {og_col})')
    print(f'down 2, right 1:    {img2Center}:  Moves: ({img2Center[0] - og_rows}, {-1 * (img2Center[1] - og_rows)})')
    print(f'up 2,   left 1:     {img3Center}:  Moves: ({img3Center[0] - og_rows}, {-1 * (img3Center[1] - og_rows)})')
    print(f'up 2,   left 0:     {img4Center}:  Moves: ({img4Center[0] - og_rows}, {-1 * (img4Center[1] - og_rows)})')

    # So if (3, 3) is the original
    # And "down 2 right 1" gives us (1, 2) = (3, 3) + (-2, -1)   (3 + rowShift = 1  =>  rowShift = -2)  and ()
    # And "up 2 left 1" gives us (5, 4) = (3, 3) + (2, 1)
    # And "up 2 left 0" gives us (5, 3) = (3, 3) + (2, 0)

    # So we should be able to calculate the shift by:
    # rowShift = convolvedRows - og_rows
    # colShift = -1 * (convolvedCol - og_col)

testFftConvolve()