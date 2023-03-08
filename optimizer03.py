import optuna
import yaml
import pipeline03

configpath = "configs/pipeline03config.yaml"
best_acc = 0.0
def objective(trial):
    with open(configpath) as f:
        config=yaml.safe_load(f)
    #config["Hyperparameters"]["loss"]=trial.suggest_categorical('loss', ["binary_crossentropy"])
    config["Hyperparameters"]["neurons1"]=trial.suggest_int('neurons1', 64, 128*57+3/2)
    config["Hyperparameters"]["neurons2"]=trial.suggest_int('neurons2', 32, 128*57+3/2)
    config["Hyperparameters"]["batch_size"]=trial.suggest_int('batch_size', 2, 128)
    config["Hyperparameters"]["layers"]=trial.suggest_int('layers', 1, 10)
    config["Hyperparameters"]["lr_adam"]=trial.suggest_float('lr_adam', 0.00000001, 0.1)
    with open(configpath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    pipeline = pipeline03.pipe_deladetect(trial=trial)
    pipeline.run_pipeline()
    if best_acc < pipeline.acc:
        pipeline.model.save(f'dst/2303_pez_dnn/model.hdf5')
    return  pipeline.acc
study = optuna.create_study(study_name="Dnn1", direction='maximize', storage="sqlite:///optunaml2dnnv3.sqlite3", load_if_exists=True)
study.optimize(objective, n_trials=1000)
study.best_params