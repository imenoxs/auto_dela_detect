# Repository Overview
This repo contains several approaches to detect delaminations in cfrp images from ultrasound images or ultrasound radio frequency data.
The several pipeline python files each contain one approach.
Each pipeline also has a optimizer python file with the corresponding number.
The optimizers use the optuna framework to optimize the parameters of the respective approach.
To see the results of the optuna training you can start the environment with `optuna-dashboard sqlite:///optuna.db`
`mlflow ui --backend-store-uri sqlite:///MLFlow.db`
in the configs folder there is also a yaml file containing the configuration of the pipelines.
If the pipeline is run the configuration gets read from the associated file.

- pipeline01.py: This approach uses simple image processing techniques automatically classifying if the image is showing a delamination or not using the fallowing steps
    - Threshold: The grascale input image gets converted into a black and white image which already lowerse noise levels
    - Morphological Erodsion is applied to erase remaining noise
    - Morphological Delation is used afterwards to make the remaining details clearer
    - The Pixelvalues of each row get added up which results in a vector showing the pixl distribution over the image hight
    - After smoothing out the curve all local maxima get extracted and if there are more then two peaks it is assumed that there is a defect present.
- pipeline02.py: Here a one layer neuronal network is being used to do a classification of the image data
- pipeline03.py. here a deep neuronal network is used to do image classification

