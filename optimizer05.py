import optuna
import yaml
import pipeline05_IMG_CNN as pipe
import mlflow
import tensorflow as tf
import datetime
import os
import Scripts.dsutils as dsutils

configpath = "configs/pipeline05config.yaml"
experiment_name= "pipeline05_IMG_CNN_MaximizeRun1"

best_loss = None
best_acc = None
def objective(trial):
    global experiment_name
    global best_loss
    global best_acc
    cb_lst = []
    final_metrics = None
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
    cb_lst.append(cb_mlflow)

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"Trial {trial.number}"):
        mlflow.set_tag("pruned",True)
        config["Hyperparameters"]["initialfilternr"]=trial.suggest_categorical('initialfilternr', [8,16,32,64,128])
        config["Hyperparameters"]["cnnlyrs"]=trial.suggest_int('layers', 0, 5)
        config["Hyperparameters"]["batch_size"]=trial.suggest_categorical('batch_size', [280,140,70,56,40,35,28,20,14,10,8,7,5,4,2,1])
        config["Hyperparameters"]["lr_adam"]=trial.suggest_float('lr_adam', 0.000000001, 0.1)
        config["Hyperparameters"]["dropout"]=trial.suggest_categorical('dropout', [True, False])
        config["Hyperparameters"]["normalization"]=trial.suggest_categorical('normalization', [True, False])

        with open(configpath, 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        mlflow.log_params(config["Hyperparameters"])

        #run pipeline
        srcpath = config["Paths"]["srcpath"]
        batchsize = config["Hyperparameters"]["batch_size"]
        cnnlyrs = config["Hyperparameters"]["cnnlyrs"]
        initialfilternr = config["Hyperparameters"]["initialfilternr"]
        dropout= config["Hyperparameters"]["dropout"]
        normalization = config["Hyperparameters"]["normalization"]
        lr=config["Hyperparameters"]["lr_adam"]

        dsutils.clean_dir(os.path.join(srcpath,"temp"))
        #get one image to extract image dimensions
        for dirpath, _, filenames in os.walk(srcpath):
            for filename in filenames:
                if filename.endswith(".png"):
                    with tf.keras.preprocessing.image.load_img(os.path.join(dirpath, filename)) as img:
                        image_size = (img.height, img.width)
                break

        ds_train, ds_val = pipe.load_ds(os.path.join(srcpath,"data"), batch_size=batchsize, image_size=image_size)
        try:
            model=pipe.create_model(input_size=image_size, cnnlyrs=cnnlyrs, initialfilternr=initialfilternr, dropout=dropout, normalization=normalization)
            history = pipe.train(model, ds_train, ds_val, lr=lr, trial=trial, cb_lst=cb_lst)
            final_metrics=pipe.analyse_model(model=model, val_dataset=ds_val, dstpath=srcpath)
            if best_loss == None or best_loss > final_metrics["final_loss"] or best_acc< final_metrics["final_acc"]:
                best_loss = final_metrics["final_loss"]
                best_acc = final_metrics["final_acc"]
                model.save(os.path.join(srcpath,"temp","model.hdf5"))
                mlflow.set_tag("model",True)
            mlflow.log_metrics(final_metrics)
        except Exception as e: 

            if not trial.should_prune():
                print("*************************************")
                print("CRASHED")
                print("*************************************")
                with open(os.path.join(srcpath,"temp","exception.txt"), "w") as f:
                    f.write(str(e))
                mlflow.set_tag("crashed",True)
                mlflow.log_artifacts(os.path.join(srcpath,"temp"))            
                raise optuna.TrialPruned()



        
        mlflow.log_artifacts(os.path.join(srcpath,"temp"))
        if trial.should_prune():
            mlflow.set_tag("pruned",True)
        else:
            mlflow.set_tag("pruned",False)
        if final_metrics != None:
            return  final_metrics["final_acc"]
        else: 
            return 0
        

study = optuna.create_study(study_name=experiment_name, direction='maximize', storage="sqlite:///optuna.db", load_if_exists=True)

study.optimize(objective, n_trials=500)


bestparams = study.best_params
with open(configpath) as f:
        config=yaml.safe_load(f)
        config["Hyperparameters"]["cnnlyrs"] = bestparams["cnnlyrs"] 
        config["Hyperparameters"]["initialfilternr"]= bestparams["initialfilternr"]    
        config["Hyperparameters"]["batch_size"] = bestparams["batch_size"]        
        config["Hyperparameters"]["lr_adam"] = bestparams["lr_adam"]
        config["Hyperparameters"]["dropout"] = bestparams["dropout"]
        config["Hyperparameters"]["normalization"] = bestparams["normalization"]
with open(f"{configpath[:-5]}best.yaml", 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)