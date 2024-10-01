# Brain MRI Segmentation

This project implements a brain MRI segmentation algorithm using Python. The main functionality is provided by the `BrainSegmenter` class, which preprocesses the input MRI images and applies segmentation techniques to isolate brain regions.

## Features

- Image preprocessing using CLAHE (Contrast Limited Adaptive Histogram Equalization).
- Otsu's thresholding for binary segmentation.
- Removal of small objects and holes in the segmented mask.
- Evaluation metrics for segmentation performance, including Intersection over Union (IoU) and Dice coefficient.

## Usage

1. Place your MRI image in the project directory.
2. Run the script to generate the segmentation mask.
3. The output will be saved as `segmentation_result.png`.

## Requirements

- numpy
- opencv-python
- scikit-image
