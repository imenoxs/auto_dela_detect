import optuna
import yaml
import pipeline03_IMG_MLP
import mlflow
import tensorflow as tf
import datetime

configpath = "configs/pipeline03config.yaml"
experiment_name= "Pipeline03_IMG_MLP_MaximizeRun7DELETE"

best_loss = 0.0
def objective(trial):
    global experiment_name
    global best_loss
    with open(configpath) as f:
        config=yaml.safe_load(f)
    mlflow.set_tracking_uri(f"sqlite:///MLFlow.db") #configures local sqlite database as logging target
    mlflow.set_experiment(experiment_name=experiment_name) # creating experiment under which future runs will get logged
    experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id # extracting experiment ID to be able to manually start runs in that scope
    cb_mlflow =  tf.keras.callbacks.LambdaCallback(
                    on_epoch_begin=None,
                    on_epoch_end=lambda epoch, logs:mlflow.log_metrics(metrics=logs, step=epoch),
                    on_batch_begin=None,
                    on_batch_end=None,
                    on_train_begin=None,
                    on_train_end=None
                )

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Trial {trial.number}"):
        mlflow.set_tag("pruned",True)
        config["Hyperparameters"]["neurons"]=trial.suggest_int('neurons', 64, 128*57*2/3)
        config["Hyperparameters"]["layers"]=trial.suggest_int('layers', 0, 8)
        config["Hyperparameters"]["batch_size"]=trial.suggest_int('batch_size', 2, 128)
        config["Hyperparameters"]["lr_adam"]=trial.suggest_float('lr_adam', 0.000000001, 0.1)
        with open(configpath, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        mlflow.log_params(config["Hyperparameters"])
        pipeline = pipeline03_IMG_MLP.pipe_deladetect(trial=trial)
        pipeline.callbackslst.append(cb_mlflow)
        pipeline.run_pipeline()
        if best_loss > pipeline.loss:
            best_loss = pipeline.loss
            pipeline.model.save(f'dst/2303_pez_dnn/model.hdf5')
        mlflow.log_metrics(pipeline.finalmetrics)
        mlflow.log_artifact("/Users/tilmanseesselberg/Nextcloud2/WIP/Bachelorarbeit/Automation/dst/2303_pez_dnn/confusion.png")
        mlflow.log_artifact("/Users/tilmanseesselberg/Nextcloud2/WIP/Bachelorarbeit/Automation/dst/2303_pez_dnn/roccurve.png")
        mlflow.log_artifact("/Users/tilmanseesselberg/Nextcloud2/WIP/Bachelorarbeit/Automation/dst/2303_pez_dnn/prereccurve.png")
        mlflow.log_artifact("/Users/tilmanseesselberg/Nextcloud2/WIP/Bachelorarbeit/Automation/dst/2303_pez_dnn/detcurve.png")
        if trial.should_prune():
            mlflow.set_tag("pruned",True)
        else:
            mlflow.set_tag("pruned",False)
        return  pipeline.finalmetrics["final_acc"]
study = optuna.create_study(study_name=experiment_name, direction='maximize', storage="sqlite:///optuna.db", load_if_exists=True)
study.optimize(objective, n_trials=500)
bestparams = study.best_params
with open(configpath) as f:
        config=yaml.safe_load(f)
        config["Hyperparameters"]["neurons"] = bestparams["neurons"]
        config["Hyperparameters"]["layers"] = bestparams["layers"]        
        config["Hyperparameters"]["batch_size"] = bestparams["batch_size"]        
        config["Hyperparameters"]["lr_adam"] = bestparams["lr_adam"]
with open(f"{configpath[:-5]}best.yaml", 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)

