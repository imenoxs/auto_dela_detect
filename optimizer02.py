import optuna
import yaml
import pipeline02

configpath = "configs/pipeline02config.yaml"
def objective(trial):
    with open(configpath) as f:
        config=yaml.safe_load(f)
    #config["Hyperparameters"]["loss"]=trial.suggest_categorical('loss', ["binary_crossentropy"])
    config["Hyperparameters"]["optimizer"]=trial.suggest_categorical('optimizersdg',["sdg"])
    config["Hyperparameters"]["neurons"]=trial.suggest_int('neurons', 64, 129*57+3/2)
    config["Hyperparameters"]["batch_size"]=trial.suggest_int('batch_size', 4, 128)
    #config["Hyperparameters"]["epochs"]=trial.suggest_int('epochs', 5, 200)
    #config["Hyperparameters"]["lr_adam"]=trial.suggest_float('lr_adam', 0.00001, 0.1)
    config["Hyperparameters"]["lr_sdg"]=trial.suggest_float('lr_sdg', 0.00001, 0.1)
    config["Hyperparameters"]["from_logits"]=trial.suggest_categorical('from_logits',["true","false"])
    with open(configpath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    pipeline = pipeline02.pipe_deladetect()
    pipeline.run_pipeline()
    return   pipeline.acc


study = optuna.create_study(study_name="Ml1", direction='maximize', storage="sqlite:///optunaml1sdg.sqlite3", load_if_exists=True)
study.optimize(objective, n_trials=120)
study.best_params