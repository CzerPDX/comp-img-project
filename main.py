from src.dispersionImg import DispersionImg
import os

if __name__ == "__main__":
    
    # Spectral response dictionary. Keys accessed by nanometer
    colorWeights = {}
    colorWeights[400] = (0.0,   0.0,    0.0)
    colorWeights[405] = (0.015, 0.005,  0.11)
    colorWeights[410] = (0.02,  0.005,  0.2)
    colorWeights[415] = (0.039, 0.01,   0.53)
    colorWeights[420] = (0.05,  0.01,   0.42)
    colorWeights[425] = (0.065, 0.015,  0.53)
    colorWeights[430] = (0.08,  0.02,   0.61)
    colorWeights[435] = (0.095, 0.025,  0.68)
    colorWeights[440] = (0.11,  0.03,   0.73)
    colorWeights[445] = (0.15,  0.034,  0.625)
    colorWeights[450] = (0.105, 0.035,  0.67)
    colorWeights[455] = (0.1,   0.039,  0.62)
    colorWeights[460] = (0.085, 0.05,   0.595)
    colorWeights[465] = (0.069, 0.085,  0.64)
    colorWeights[470] = (0.035, 0.13,   0.7)
    colorWeights[475] = (0.015, 0.185,  0.775)
    colorWeights[480] = (0.0,   0.245,  0.815)
    colorWeights[485] = (0.02,  0.345,  0.828)
    colorWeights[490] = (0.04,  0.47,   0.83)
    colorWeights[495] = (0.075, 0.57,   0.828)
    colorWeights[500] = (0.095, 0.68,   0.8)
    colorWeights[505] = (0.1,   0.74,   0.68)
    colorWeights[510] = (0.09,  0.77,   0.45)
    colorWeights[515] = (0.085, 0.815,  0.24)
    colorWeights[520] = (0.075, 0.83,   0.075)
    colorWeights[525] = (0.072, 0.815,  0.071)
    colorWeights[530] = (0.07,  0.78,   0.06)
    colorWeights[535] = (0.065, 0.73,   0.059)
    colorWeights[540] = (0.06,  0.7,    0.06)
    colorWeights[545] = (0.06,  0.705,  0.065)
    colorWeights[550] = (0.06,  0.72,   0.071)
    colorWeights[555] = (0.075, 0.73,   0.08)
    colorWeights[560] = (0.085, 0.72,   0.08)
    colorWeights[565] = (0.23,  0.655,  0.085)
    colorWeights[570] = (0.44,  0.57,   0.09)
    colorWeights[575] = (0.63,  0.48,   0.09)
    colorWeights[580] = (0.78,  0.4,    0.09)
    colorWeights[585] = (0.85,  0.325,  0.85)
    colorWeights[590] = (0.915, 0.24,   0.08)
    colorWeights[595] = (0.965, 0.16,   0.075)
    colorWeights[600] = (0.990, 0.1,    0.072)
    colorWeights[605] = (0.985, 0.085,  0.072)
    colorWeights[610] = (0.965, 0.072,  0.072)
    colorWeights[615] = (0.93,  0.07,   0.07)
    colorWeights[620] = (0.85,  0.06,   0.07)
    colorWeights[625] = (0.84,  0.05,   0.065)
    colorWeights[630] = (0.77,  0.04,   0.05)
    colorWeights[635] = (0.71,  0.038,  0.049)
    colorWeights[640] = (0.63,  0.03,   0.04)
    colorWeights[645] = (0.52,  0.028,  0.038)
    colorWeights[650] = (0.44,  0.025,  0.035)
    colorWeights[655] = (0.35,  0.026,  0.032)
    colorWeights[660] = (0.27,  0.025,  0.029)
    colorWeights[665] = (0.19,  0.021,  0.024)
    colorWeights[670] = (0.13,  0.015,  0.019)
    colorWeights[685] = (0.08,  0.013,  0.014)
    colorWeights[680] = (0.045, 0.011,  0.009)
    colorWeights[685] = (0.027, 0.009,  0.007)
    colorWeights[690] = (0.015, 0.003,  0.002)
    colorWeights[695] = (0.01,  0.001,  0.001)
    colorWeights[700] = (0.0,   0.0,    0.0)


    # Test a good image with broad spectral content (will provide an easier set of matches)
    # Image location and name
    imgsFolder = 'img/'
    imgFileName = 'v2_dispersed.NEF'

    # Get the paths for the operating system we're on
    imgName = os.path.splitext(imgFileName)[0]
    imgLocation = os.path.normpath(f'{imgsFolder}{imgFileName}')
    outputImagesFolder = f'{imgsFolder}{imgName}_output/'

    # Parameters for the program
    reductionPercent = 0.5              # Reduced size for processing keypoints and matches
    clipLimit = 1.0                     # For CLAHE contrast 
    tileGridSize=(10, 10)               # For CLAHE contrast 
    gaussianKernelSize = 5              # For Smoothing
    verticalThresholdPercent = 0.01     # Vertical threshold for allowable vertical change in matches
    horizontalThresholdPercent = 0.05   # Horizontal threshold for allowable horizontal change in matches

    
    dispersedImgObj = DispersionImg(outputImagesFolder, 
                                    imgLocation, 
                                    reductionPercent,
                                    colorWeights,
                                    verticalThresholdPercent,
                                    horizontalThresholdPercent,
                                    gaussianKernelSize,
                                    clipLimit,
                                    tileGridSize
                                    )



    # Test on image with less spectral content
    # Image location and name
    imgsFolder = 'img/'
    imgFileName = 'DSC_5911.NEF'

    # Get the paths for the operating system we're on
    imgName = os.path.splitext(imgFileName)[0]
    imgLocation = os.path.normpath(f'{imgsFolder}{imgFileName}')
    outputImagesFolder = f'{imgsFolder}{imgName}_output/'

    # Parameters for the program
    reductionPercent = 0.5              # Reduced size for processing keypoints and matches
    clipLimit = 1.0                     # For CLAHE contrast 
    tileGridSize=(10, 10)               # For CLAHE contrast 
    gaussianKernelSize = 5              # For Smoothing
    verticalThresholdPercent = 0.99     # Vertical threshold for allowable vertical change in matches
    horizontalThresholdPercent = 0.99   # Horizontal threshold for allowable horizontal change in matches

    
    dispersedImgObj = DispersionImg(outputImagesFolder, 
                                    imgLocation, 
                                    reductionPercent,
                                    colorWeights,
                                    verticalThresholdPercent,
                                    horizontalThresholdPercent,
                                    gaussianKernelSize,
                                    clipLimit,
                                    tileGridSize
                                    )
    