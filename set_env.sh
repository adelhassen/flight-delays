#!/usr/bin/env bash

AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY=""
AWS_DEFAULT_REGION=""
MLFLOW_S3_BUCKET=""
MLFLOW_TRACKING_URI=""  #http://127.0.0.1:5000 (local environment)
PREFECT_API_URL=""  #http://127.0.0.1:4200/api (local environment)

truncate -s 0 .env
touch .env

echo "AWS_ACCESS_KEY_ID = $AWS_ACCESS_KEY_ID" >> .env
echo "AWS_SECRET_ACCESS_KEY = $AWS_SECRET_ACCESS_KEY" >> .env
echo "AWS_DEFAULT_REGION = $AWS_DEFAULT_REGION" >> .env
echo "MLFLOW_S3_BUCKET = $MLFLOW_S3_BUCKET" >> .env
echo "MLFLOW_TRACKING_URI = $MLFLOW_TRACKING_URI" >> .env
echo "PREFECT_API_URL = $PREFECT_API_URL" >> .env
