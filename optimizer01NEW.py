import optuna
import yaml
import pipeline01_IMG_Manual_NEW

def objective(trial):
    with open('configs/pipeline01config.yaml') as f:
        config=yaml.safe_load(f)
    config["Processing"]["thrval"]=trial.suggest_int('thrval', 0, 255)
    config["Processing"]["movinge_av"]=trial.suggest_int('movinge_av', 1, 20)
    config["Processing"]["smoothing_val"]=trial.suggest_int('smoothing_val', 0, 100)
    config["Processing"]["dil_iters"]=trial.suggest_int('dil_iters', 0, 50)
    config["Processing"]["ero_iters"]=trial.suggest_int('ero_iters', 0, 50)
    with open('config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    acc = pipeline01_IMG_Manual_NEW.main()

    return acc

study = optuna.create_study(study_name="AccumPxlRun1", direction='maximize', storage="sqlite:///optuna.db", load_if_exists=True)
study.optimize(objective, n_trials=200)
study.best_params