import optuna
import mlflow
import yaml
import pipeline01_IMG_Manual
from Scripts import dsutils
import os

configpath = "configs/pipeline05config.yaml"
experiment_name= "pipeline01_IMG_Manual_NEW_RUN1Final"


def objective(trial):
    # function to be optimized by optuna
    global configpath
    global experiment_name

    #open pipeline config and store new values
    with open('configs/pipeline01config.yaml') as f:
        config=yaml.safe_load(f)
    config["Processing"]["thrval"]=trial.suggest_int('thrval', 0, 255)
    config["Processing"]["movinge_av"]=trial.suggest_int('movinge_av', 1, 20)
    config["Processing"]["smoothing_val"]=trial.suggest_int('smoothing_val', 0, 100)
    config["Processing"]["dil_iters"]=trial.suggest_int('dil_iters', 0, 50)
    config["Processing"]["ero_iters"]=trial.suggest_int('ero_iters', 0, 50)
    config["Processing"]["peakthr"]=trial.suggest_int('peakthr', 0, 10)
    with open('configs/pipeline01config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # prepare directory where artefacts are saved to
    dstpath = config['Paths']['dstpath']
    dsutils.clean_dir(os.path.join(dstpath,'temp'))

    # setting up mlflow tracking
    mlflow.set_tracking_uri(f"sqlite:///MLFlow.db") #configures local sqlite database as logging target
    mlflow.set_experiment(experiment_name=experiment_name) # creating experiment under which future runs will get logged
    experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id # extracting experiment ID to be able to manually start runs in that scope
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Trial {trial.number}"):
        # log hyperparameters and start pipeline
        mlflow.log_params(config["Processing"])
        full_eval = pipeline01_IMG_Manual.main(trial) 
        mlflow.log_metrics(full_eval)
        mlflow.log_artifacts(os.path.join(dstpath,'temp'))

    return full_eval['acc']

if __name__ == '__main__': #run optuna experiment
    study = optuna.create_study(study_name=experiment_name, direction='maximize', storage="sqlite:///optuna.db", load_if_exists=True)
    study.optimize(objective, n_trials=3000)
    study.best_params