import os
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, average_precision_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import pickle
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently import ColumnMapping
import boto3
from dotenv import load_dotenv


@task
def read_data(file_path:str, file_path_airline_codes:str = "data/L_UNIQUE_CARRIERS.csv"):

    # Open the ZIP file
    with zipfile.ZipFile(file_path, 'r') as z:
        # Identify the CSV file, ignoring system files like __MACOSX
        csv_file_name = [f for f in z.namelist() if f.endswith('.csv') and not f.startswith('__MACOSX/')][0]
        # Read the specific CSV file
        with z.open(csv_file_name) as f:
            df = pd.read_csv(f)

    op_unique_carrier_lookup = pd.read_csv(file_path_airline_codes)
    
    df = (
        df
        .merge(op_unique_carrier_lookup, left_on = ["OP_UNIQUE_CARRIER"], right_on=["Code"])
        .rename(columns = {"Description":"AIRLINE", })

    )

    cols_to_keep = [
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN",
        "DEST",
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
        "ARR_DELAY",
        "ARR_DEL15"
    ]

    df = df[cols_to_keep].dropna(subset=["ARR_DELAY", "ARR_DEL15"])

    return df


def time_to_minutes(int_time):
    # Convert the integer to string to handle different lengths
    str_time = str(int_time).zfill(4)  # Ensure it's at least 4 digits long
    # Extract hours and minutes
    hours = int(str_time[:-2])  # All but last two characters
    minutes = int(str_time[-2:])  # Last two characters
    
    # Convert hour and minutes to total minutes of the day
    total_minutes = hours * 60 + minutes
    
    return total_minutes


# Define function to create cyclical features
def create_cyclical_features(df: pd.DataFrame, column: str, max_val: int):
    
    df[f'{column}_SIN'] = np.sin(2 * np.pi * df[column] / max_val)
    df[f'{column}_COS'] = np.cos(2 * np.pi * df[column] / max_val)
    
    return df


@task
def feature_engineering(df: pd.DataFrame):

    X = df.drop(["ARR_DEL15"], axis=1).copy()
    y = df['ARR_DEL15'].values

    X["CRS_DEP_TIME_MINUTES"] = X["CRS_DEP_TIME"].apply(time_to_minutes)
    X["CRS_ARR_TIME_MINUTES"] = X["CRS_ARR_TIME"].apply(time_to_minutes)

    X = create_cyclical_features(X, "DAY_OF_MONTH", X["DAY_OF_MONTH"].max())
    X = create_cyclical_features(X, "DAY_OF_WEEK", X["DAY_OF_WEEK"].max())
    X = create_cyclical_features(X, "CRS_DEP_TIME_MINUTES", 1440)
    X = create_cyclical_features(X, "CRS_ARR_TIME_MINUTES", 1440)

    # Drop distance because that would be hard for someone to find
    # Drop ARR_DELAY because we will use ARR_DEL15 to create a categorical ML problem
    X.drop(["DISTANCE", "ARR_DELAY", "DAY_OF_MONTH", "DAY_OF_WEEK", "CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_DEP_TIME_MINUTES", "CRS_ARR_TIME_MINUTES"], axis=1, inplace=True)   

    return X, y 


@task
def transform_data(X_train_raw, X_val_raw, X_test_raw):

    categorical = ["AIRLINE", "ORIGIN", "DEST"]
    numerical = ['CRS_ELAPSED_TIME', 'DAY_OF_MONTH_SIN', 'DAY_OF_MONTH_COS', 'DAY_OF_WEEK_SIN', 'DAY_OF_WEEK_COS',
                'CRS_DEP_TIME_MINUTES_SIN', 'CRS_DEP_TIME_MINUTES_COS', 'CRS_ARR_TIME_MINUTES_SIN',
                'CRS_ARR_TIME_MINUTES_COS'
                ]

    dv = DictVectorizer()

    train_dicts = X_train_raw[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = X_val_raw[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    test_dicts = X_test_raw[categorical + numerical].to_dict(orient='records')
    X_test = dv.transform(test_dicts)

    return X_train, X_val, X_test, dv


@task
def fit_base_models(model_name, X_train, X_val, y_train, y_val, dv):

    with mlflow.start_run():
        
        if not os.path.exists("./models"):
            os.makedirs("models")
        
        mlflow.set_tag("model", type(model_name()).__name__)
        mlflow.set_tag("purpose", "train models")
        
        model = model_name(random_state=42).fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        mlflow.log_params(model.get_params())
        
        mlflow.log_metric("accuracy", accuracy_score(y_val, y_pred))
        mlflow.log_metric("precision", precision_score(y_val, y_pred))
        mlflow.log_metric("recall", recall_score(y_val, y_pred))
        mlflow.log_metric("f1", f1_score(y_val, y_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_val, y_pred_proba))
        mlflow.log_metric("auprc", average_precision_score(y_val, y_pred_proba))
        
        mlflow.sklearn.log_model(model, artifact_path="models")
        
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        
        print(f"Training for {type(model_name()).__name__} complete!")


@task
def tune_xgboost(train, valid, y_val, eval_metric, dv):
    
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "XGBClassifier")
            mlflow.set_tag("purpose", "Tune XGBClassifier")
            mlflow.set_tag("eval_metric", eval_metric)

            mlflow.log_params(params)

            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=5,
                evals=[(valid, 'validation')],
                early_stopping_rounds=5
            )

            y_pred = booster.predict(valid)
            y_pred_class = (y_pred >= 0.5).astype(int)

            if eval_metric == "logloss":
                metric = log_loss(y_val, y_pred)
            # since value is positive and we want to minimize, make it negative
            elif eval_metric == "auc":
                metric = -roc_auc_score(y_val, y_pred)
            # since value is positive and we want to minimize, make it negative
            elif eval_metric == "aucpr":
                metric = -average_precision_score(y_val, y_pred) 


            mlflow.log_metric("accuracy", accuracy_score(y_val, y_pred_class))
            mlflow.log_metric("precision", precision_score(y_val, y_pred_class))
            mlflow.log_metric("recall", recall_score(y_val, y_pred_class))
            mlflow.log_metric("f1", f1_score(y_val, y_pred_class))
            mlflow.log_metric("roc_auc", roc_auc_score(y_val, y_pred))
            mlflow.log_metric("auprc", average_precision_score(y_val, y_pred))

            mlflow.xgboost.log_model(booster, artifact_path="models")

            with open("models/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        return {'loss': metric, 'status': STATUS_OK}
    
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 10, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'binary:logistic',
        'eval_metric': eval_metric,
        'seed': 42
    }
    
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials()
    )
    
    return


@task
def train_and_log_model(train, valid, test, y_val, y_test, dv, params):
    with mlflow.start_run():
        
        mlflow.set_tag("model", "XGBClassifier")
        mlflow.set_tag("purpose", "Compare best models")
        mlflow.log_params(params)
        
        booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=5,
                evals=[(valid, 'validation')],
                early_stopping_rounds=5
            )


        # Evaluate and log model on the validation set
        y_pred_val = booster.predict(valid)
        y_pred_val_class = (y_pred_val >= 0.5).astype(int)
        
        mlflow.log_metric("accuracy", accuracy_score(y_val, y_pred_val_class))
        mlflow.log_metric("precision", precision_score(y_val, y_pred_val_class))
        mlflow.log_metric("recall", recall_score(y_val, y_pred_val_class))
        mlflow.log_metric("f1", f1_score(y_val, y_pred_val_class))
        mlflow.log_metric("roc_auc", roc_auc_score(y_val, y_pred_val))
        mlflow.log_metric("auprc", average_precision_score(y_val, y_pred_val))
        
        # Evaluate and log model on test set
        y_pred_test = booster.predict(test)
        y_pred_test_class = (y_pred_test >= 0.5).astype(int)
        
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test_class))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test_class))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test_class))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test_class))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_test))
        mlflow.log_metric("test_auprc", average_precision_score(y_test, y_pred_test))
        
        mlflow.xgboost.log_model(booster, artifact_path="models")

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")


@task
def run_register_model(train, valid, test, y_val, y_test, dv, client, MLFLOW_S3_BUCKET, top_n: 5):
    
    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name("flight_delay")
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        filter_string="tags.purpose='Tune XGBClassifier'",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.auprc DESC"]
    )

    for run in runs:
        train_and_log_model(train, valid, test, y_val, y_test, dv, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name("xgboost_best_models")
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_auprc DESC"]
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="xgboost_best_models")

    # Upload the best run_id to S3 file
    s3_client = boto3.client('s3')
    s3_client.put_object(Body=run_id, Bucket=MLFLOW_S3_BUCKET, Key="best_run_id.txt")


@task
def get_monitoring_data(xgb_data, X, client, MLFLOW_S3_BUCKET):

    X = X.drop(["DISTANCE", "ARR_DELAY"], axis=1).copy()

    X["DAY_OF_WEEK"] = (
        X["DAY_OF_WEEK"]
        .replace(
            {1:"Monday", 2:"Tuesday", 3:"Wednesday", 4:"Thursday", 
            5:"Friday", 6:"Saturday", 7:"Sunday"
            }
        )
    )

    s3_bucket_path = f"s3://{MLFLOW_S3_BUCKET}/2"
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=MLFLOW_S3_BUCKET, Key='best_run_id.txt')
    best_run_id = response['Body'].read().decode('utf-8')

    logged_model = f'{s3_bucket_path}/{best_run_id}/artifacts/models'

    # Load model as a XGBoost
    model_to_deploy = mlflow.xgboost.load_model(logged_model)

    X["Arrival Delay"] = np.where(X["ARR_DEL15"] == 0, "On Time", "Delay")
    X.drop("ARR_DEL15", axis=1, inplace=True)

    # Load model as a XGBoost
    model_to_deploy = mlflow.xgboost.load_model(logged_model)
    
    preds_delay = model_to_deploy.predict(xgb_data)
    preds_on_time = 1 - preds_delay

    X["Delay"] = preds_delay
    X["On Time"] = preds_on_time

    return X


@task
def create_monitoring_report(reference_data, current_data, MLFLOW_S3_BUCKET):

    categorical = ["AIRLINE", "ORIGIN", "DEST", "DAY_OF_MONTH", "DAY_OF_WEEK"]
    numerical = ["CRS_DEP_TIME", "CRS_ARR_TIME", "CRS_ELAPSED_TIME"]

    column_mapping = ColumnMapping()

    column_mapping.target = 'Arrival Delay'
    column_mapping.prediction = ['Delay', 'On Time']
    column_mapping.pos_label = 'Delay'
    column_mapping.numerical_features = numerical
    column_mapping.categorical_features = categorical

    classification_performance_report = Report(metrics=[
        ClassificationPreset(),
        DataDriftPreset(),
    ])

    classification_performance_report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    classification_performance_report.save_html("reports/Monitoring_Report.html")

    # Upload the file to S3
    s3_client = boto3.client('s3')
    with open("reports/Monitoring_Report.html", 'rb') as file:
        s3_client.upload_fileobj(file, MLFLOW_S3_BUCKET, "reports/Monitoring_Report.html")




@flow(task_runner=SequentialTaskRunner())
def main_flow():

    load_dotenv(".env")

    MLFLOW_S3_BUCKET = os.getenv("MLFLOW_S3_BUCKET")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("flight_delay")
    client = MlflowClient(MLFLOW_TRACKING_URI)

    train_data_raw = read_data("data/july_2023.csv.zip")
    valid_data_raw = read_data("data/august_2023.csv.zip")
    test_data_raw = read_data("data/september_2023.csv.zip")

    X_train_raw, y_train = feature_engineering(train_data_raw)
    X_val_raw, y_val = feature_engineering(valid_data_raw)
    X_test_raw, y_test = feature_engineering(test_data_raw)

    X_train, X_val, X_test, dv = transform_data(X_train_raw, X_val_raw, X_test_raw)

    fit_base_models(LogisticRegression, X_train, X_val, y_train, y_val, dv)
    fit_base_models(xgb.XGBClassifier, X_train, X_val, y_train, y_val, dv)

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    test = xgb.DMatrix(X_test, label=y_test)
    
    tune_xgboost(train, valid, y_val, "logloss", dv)
    tune_xgboost(train, valid, y_val, "auc", dv)
    tune_xgboost(train, valid, y_val, "aucpr", dv)

    mlflow.set_experiment("xgboost_best_models")

    run_register_model(train, valid, test, y_val, y_test, dv, client, MLFLOW_S3_BUCKET, top_n=5)

    reference_data = get_monitoring_data(train, train_data_raw, client, MLFLOW_S3_BUCKET)
    current_data = get_monitoring_data(valid, valid_data_raw, client, MLFLOW_S3_BUCKET)

    create_monitoring_report(reference_data, current_data, MLFLOW_S3_BUCKET)


if __name__ =='__main__':
    main_flow.serve(
       name="train_flight_delay_model",
       # Run every week
       interval=timedelta(days=7),
       description="ML pipeline to process, train, and find best fligh delay model.",
       tags=["flight", "ml-pipeline"],
    )
    
