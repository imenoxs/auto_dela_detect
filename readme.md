# Repository Overview
This repo contains several approaches to detect delaminations in cfrp images from ultrasound images or ultrasound radio frequency data.
The python scripts starting with `pipeline0[...]` contain the code for the models.
The python scripts starting with `optimizer0[...]` contain the optimizer function to optimize the parameters of the respective pipeline.

- pipeline01_IMG_Manual is a deterministic approach applying image processing techniques for classification. optimizer01 is the optimizer for the parameters for the pipeline
- pipeline02_IMG_MLP is a classification MLP.
- pipeline03_IMG_CNNOptPooling is a classification CNN.

The results are saved inside the MLFlow.db and optuna.db for pipeline 1 and and E1 and E2 of pipeline 2.
The files MLFlow_PC.db and optuna_PC.db contai  the results of pipeline 3 and pipeline2 E3

To see the results of the optuna training you can start the environment with 

`optuna-dashboard sqlite:///optuna.db`
or
`mlflow ui --backend-store-uri sqlite:///MLFlow.db`

Make sure to modify the file name accordingly.

The configs folder contains yaml files holding the configuration of the pipelines.
If the pipeline is run the configuration gets read from the associated file.

- pipeline01.py: This approach uses simple image processing techniques automatically classifying if the image is showing a delamination or not using the fallowing steps
    - Threshold: The grascale input image gets converted into a black and white image which already lowerse noise levels
    - Morphological Erodsion is applied to erase remaining noise
    - Morphological Delation is used afterwards to make the remaining details clearer
    - The Pixelvalues of each row get added up which results in a vector showing the pixl distribution over the image hight
    - After smoothing out the curve all local maxima get extracted and if there are more then two peaks it is assumed that there is a defect present.
- pipeline02.py: Here a one layer neuronal network is being used to do a classification of the image data
- pipeline03.py. here a deep neuronal network is used to do image classification

# MLflow Troubleshooting
When starting mlflow ui returns the error "alembic.util.exc.CommandError: Can't locate revision identified by [...]", make sure your environment uses Python 3.9.16.

If copying the mlflow runs folder to another location the paths in the database have to bechanged.
This can be done with the fallowing db querries:
```SQL
UPDATE tablename SET columnname = replace( columnname, '/original/Path/', '/new/Path/' ) WHERE columnname LIKE '/original/Path/%';
```
Make sure to keep the "%" following original path after "LIKE".

This has to be done for the **artifact_uri** column in the **runs** table and the **artifact_location** in the **experiments** table.