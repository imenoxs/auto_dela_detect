import optuna
import yaml
import pipeline01_IMG_Manual

def objective(trial):
    with open("configs/pipeline01config.yaml") as f:
        config=yaml.safe_load(f)
    config["Thresholding"]["thrval"]=trial.suggest_int('thrval', 0, 255)
    config["Processing1"]["movingaverage"]=trial.suggest_int('movingaverage', 1, 20)
    config["Processing1"]["threshold"]=trial.suggest_int('threshold', 0, 100)
    config["Erodil"]["dil_iters"]=trial.suggest_int('dil_iters', 0, 50)
    config["Erodil"]["ero_iters"]=trial.suggest_int('ero_iters', 0, 50)
    with open('config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    pipeline = pipeline01_IMG_Manual.pipe_deladetect()
    pipeline.run_pipeline()

    return pipeline.acc

study = optuna.create_study(study_name="AccumPxlRun1", direction='maximize', storage="sqlite:///optuna.db", load_if_exists=True)
study.optimize(objective, n_trials=200)
study.best_params