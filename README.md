# Hyperspectral Image Undispersion Tool

This software allows the user to take a raw NEF file from a Nikon D5000 that has been dispersed through a prism. Using the spectral response of that camera's particular CMOS sense as a parameter, the `dispersionImg` class will align the content and then provide an approximation of the hyperspectral content of the scene without the need for a diffraction grating or hyperspectral sensor.

## Usage
- Place the raw NEF image you want to undisperse into the `/img` directory.
- Set the software in main.py to look for that image
- Run the software using `Python main.py`

It will populate the image directory with an _output folder that is prefaced with the name of the input file. This directory will include 5 images:
- `3-channel-stacked-RGB`: This is the image after having 3 of its channels aligned.
- `6-channel-multispectral-approximation`: This is a set of 6 images that breaks down a few intermediate colors based on the aligned image and the provided color weights.
- A series of three images that provides a visualization of the matchpoints made. Each match is made from the `green_1` channel to the other 3 channels (`blue`, `green_2`, and `red`).

You may need to adjust the parameters to be more or less permissive in `main.py`. Ideally you want a small number of accurate matches from green_1 to the other channels. (though green_1 to green_2 will typically produce many matches as they are very similar sensors)

## Roadmap

This software is deeply in-progress. It was produced under a tight time constraint and has accumulated some technical debt I am working my way through. Though this project has had to be put on the backburner for the moment I will eventually be able to address the following concerns:

- Provide more "Pythonic" error handling to use exceptions rather than error flags.
- Provide the ability to apply to more than one type of RAW file taken by one type of camera (currently only .NEF from a Nikon D5000).
- Provide a graphical user interface.
- Separate the concerns as the main class `dispersionImg` is fairly large. If possible, I would like to refactor it to several classes.
- Provide functionality to look at the number of matches and try lower/higher parameter values to get better outputs.

## Installation and Environment

Python 3.9.13 is necessary. Newer versions of Python will not work with the software.

The libraries below will also be required.

- numpy
- rawpy
- cv2 (opencv-python)
- matplotlib
- scipy

If you are working in a Windows environment Anaconda is suggested!