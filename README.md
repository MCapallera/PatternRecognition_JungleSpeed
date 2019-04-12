# PatternRecognition_JungleSpeed
The app folder contains the models we used for the different tasks

The report folder contains the reports for the exercices, one report per task with the results in it (% accuracy, loss...)

The results folder contains the results of each task

## Keyword spotting task
### step one
 - rid of alpha channel (skimage.exposure.equalize_adapthist)
 - light homogenisation (exposure)
 - binarization (Adaptive Gaussian Thresholding, openCV)
 - scale
 - crop

### feature selection
 - window based
    - hog 
    - cumulative black
    -
 
### Comparison
 - Constrained Dynamic Time Warp (https://github.com/lukauskas/dtwco)
 
### compute statistics
 - by hand :)
 
 