
from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
import xgboost as xgb
import pickle
import boto3

# MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# client = MlflowClient(MLFLOW_TRACKING_URI)

# registered_model = client.search_registered_models(filter_string="name='xgboost_best_model_test'")
# registered_model_run_id = registered_model[0].latest_versions[0].run_id


# logged_model = f'runs:/{registered_model_run_id}/models'

s3_bucket_path = "s3://flight-delay-mlflow/2"
s3_client = boto3.client('s3')
response = s3_client.get_object(Bucket="flight-delay-mlflow", Key='best_run_id.txt')
best_run_id = response['Body'].read().decode('utf-8')

logged_model = f'{s3_bucket_path}/{best_run_id}/artifacts/models'

# Load model as a XGBoost
model_to_deploy = mlflow.xgboost.load_model(logged_model)

# Load dv from s3
s3 = boto3.resource("s3")
dv = pickle.loads(s3.Bucket("flight-delay-mlflow").Object(f"2/{best_run_id}/artifacts/preprocessor/preprocessor.b").get()['Body'].read())

# path_to_dv = client.download_artifacts(run_id=registered_model_run_id, path='preprocessor/preprocessor.b')

# with open(path_to_dv, 'rb') as f_out:
#     dv = pickle.load(f_out)


def predict(features):
    X_transformed = dv.transform(features)
    X = xgb.DMatrix(X_transformed)
    pred = model_to_deploy.predict(X)[0]
    pred_class = (pred >= 0.5).astype(int)
    
    if pred_class:
        return_string = f"Delayed! There is a {pred:.2%} chance your flight will be delayed."
    else:
        return_string = f"On time! There is a {pred:.2%} chance your flight will be delayed."
    return return_string


app = Flask('flight-delay-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    flight_features = request.get_json()
    pred = predict(flight_features)

    result = {
        'delay': pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)