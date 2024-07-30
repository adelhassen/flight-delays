# Flight Delay Predictor

## Project Description

The Flight Delay Predictor project is designed to build a robust machine learning pipeline for predicting flight delays. A flight is considered delayed if it is 15 minutes or more late as per the United States Federal Aviation Administration (FAA) standards. The prediction model utilizes historical flight data to make accurate predictions about future flight delays.

Flight delays are significant for several reasons. They affect customer satisfaction, with passengers experiencing frustration and inconvenience due to unexpected wait times. Economically, delays can lead to increased operational costs for airlines and airports, including costs related to fuel, crew, and ground services. Delayed flights can also cause cascading effects across the entire flight network, leading to broader disruptions. Accurately predicting flight delays can help in better resource allocation, improved customer satisfaction, and overall operational efficiency.

## Objective

The primary objective of this project is to develop a robust machine learning pipeline to predict flight delays using historical data. The focus is on creating an industry-grade machine learning workflow that includes essential MLOps components such as:

- **Experiment Tracking**: Utilizing MLFlow to track and manage experiments by logging parameters, metrics, and models.
- **Workflow Orchestration**: Using Prefect to manage and orchestrate the different tasks in the pipeline, ensuring smooth execution and monitoring.
- **Model Deployment and Monitoring**: Deploying the best model and setting up monitoring using Evidently to track data drift and model performance over time.
- **Environment Management**: Using virtual environments and Docker to maintain complex working environments.

A key consideration in this project is to avoid any data leakage by ensuring that only features available at the time of prediction are used for training the model. This ensures that the model remains reliable and valid for real-world applications.

## Data

The dataset used for this project is the Bureau of Transportation Statistics’ “Reporting Carrier-On Time Performance” dataset. This dataset is available at [this link](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) and contains monthly data dating back to 1987. It includes various details such as:

- **Arrival Information**: Arrival times, delay times, etc.
- **Departure Information**: Departure times, delay times, etc.
- **Origin Information**: Information about the origin airport.
- **Destination Information**: Information about the destination airport.
- **Flight Summaries**: Summarized details about flights, including carrier codes and flight numbers.

The dataset is preprocessed to retain only the relevant columns and remove any rows with missing data in crucial columns. The key features selected for the model include:

- **Day of Month**
- **Day of Week**
- **Airline**
- **Origin Airport**
- **Destination Airport**
- **Scheduled Departure Time**
- **Scheduled Arrival Time**
- **Scheduled Elapsed Time**
- **Distance** (considered initially but dropped as it is highly correlated to Scheduled Elapsed Time and difficult for one to retrieve)

## Pipeline Components

### Environment Setup
- **Docker**: Used to containerize the application for consistent and reproducible environments.
- **Virtual Environment**: Some scripts are run in a virtual environment to ensure compatibility and isolation of dependencies.

### Experiment Tracking
- **MLFlow**: Used to track experiments, logging parameters, metrics, and models. It helps in comparing different models and selecting the best-performing one.

### Workflow Orchestration
- **Prefect**: Manages and orchestrates the pipeline tasks, ensuring smooth execution and monitoring of the entire process.

### Model Deployment and Monitoring
- **Model Deployment**: The best-performing model is deployed using Docker. The deployment includes an API endpoint that takes flight details as input and returns delay predictions.
- **Model Monitoring**: Evidently is used to monitor the model's performance over time, tracking metrics like data drift and classification performance.


## Reproducibility Steps

### Machine Setup

1. **Set up Machine**: Create an EC2 instance or run it locally on your computer. If you run on an EC2 instance:
   ![images/ec2_setup.png](https://github.com/adelhassen/flight-delays/blob/7327c4bdc83599f4ff278b9e6392af31ba0599e3/images/ec2_setup.png)

   - Ubuntu
   - 64-bit (x86)
   - t2.xlarge (t2.micro on free tier should be enough)
   - Configure storage: 30GiB
   - Set up a key pair
3. **Configure Security Group**: Ports `4200` (Prefect), `5000` (MLFlow), and `9696` (Flask) needed to be opened in addition to port `22` (SSH) for source 0.0.0.0/0
   ![](https://github.com/adelhassen/flight-delays/blob/7327c4bdc83599f4ff278b9e6392af31ba0599e3/images/ec2_security_group.png)
5. **SSH into EC2 Instance**: Access your EC2 through SSH.
    ```bash
    ssh -i "your-key-pair.pem" ubuntu@ec2-xx-xxx-xxx-x.compute-1.amazonaws.com
    ```
6. **Set up EC2 Instance**:
   - Download Python 3.11.5 from Anaconda to avoid any Python version issues using:
     ```bash
     wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
     ```
  - Follow instructions at [this link](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/01-intro) to set up Docker properly.
5. **Create S3 Bucket**: Create an S3 bucket using AWS CLI.
6. **Clone the Repository**: Use SSH protocol to clone the repo.
    - Follow instructions [here](https://phoenixnap.com/kb/git-clone-ssh)
    - Then, run this command:
      ```bash
      git clone git@github.com:adelhassen/flight-delays.git
      ```

### Environment Setup

1. **Move into Project Directory**:
   ```bash
   cd flight-delays/
   ```
2. **Setting Environment Variables**: Configure environment variables in set_env.sh:
   - AWS_ACCESS_KEY_ID: Generate one using AWS.
   - AWS_SECRET_ACCESS_KEY: Generate one using AWS.
   - AWS_DEFAULT_REGION: Set this to your default AWS region.
   - MLFLOW_S3_BUCKET: Set this as the name of the S3 bucket created in Step 4 of Machine Setup.
   - MLFLOW_TRACKING_URI: Use `http://127.0.0.1:5000` for local development or replace `127.0.0.1` with EC2 Public IPv4 address if on EC2 instance.
   - PREFECT_API_URL: Use `http://127.0.0.1:4200/api` for local development or replace `127.0.0.1` with EC2 Public IPv4 address if on EC2 instance.
   
3. **Run set_env.sh**: This will save environment variables in a .env file.
   ```bash
   source set_env.sh
   ```
4. **Building and Running Docker Image**: Build and run the Docker image for a consistent and isolated environment. This will run `src/predict.py`, an API endpoint that accepts flight details as an input and returns a delay prediction.
   ```bash
   docker build -t flight-delay-prediction:v1 .
   ```
   ```bash
   docker run -it --rm -p 9696:9696 flight-delay-prediction:v1
   ```
5. **Virtual Environment**: Create and activate a virtual environment for running specific scripts.

### Connecting to Servers

1. **Open a New Terminal**
2. **SSH into EC2 Instance**: Access your EC2 through SSH.
    ```bash
    ssh -i "your-key-pair.pem" ubuntu@ec2-xx-xxx-xxx-x.compute-1.amazonaws.com
    ```
3. **Move to Project Folder and Set Environment Variables**
   ```bash
   cd flight-delays/ && source set_env.sh
   ```
4. **Virtual Environment**: Create and activate a virtual environment for running specific scripts. After you run this once, you can use `pipenv shell` to activate the virtual environment going forward.
   ```bash
   pip install --upgrade pip
   ```
   ```bash
   pip install pipenv
   ```
   ```bash
   pipenv install --system --deploy
   ```
   ```bash
   pipenv shell
   ```
5. **Tracking with MLFlow**: Open MLFlow UI to keep track of experiments and model performance. If you are running this locally, remove `-h 0.0.0.0`.
   ```bash
   mlflow server --backend-store-uri sqlite:///backend.db --default-artifact-root=s3://$MLFLOW_S3_BUCKET/ -h 0.0.0.0
   ```
   ![](https://github.com/adelhassen/flight-delays/blob/7327c4bdc83599f4ff278b9e6392af31ba0599e3/images/mlflow_experiments.png)
6. **Open a New Terminal and SSH into EC2 Instance**:
    ```bash
    ssh -i "your-key-pair.pem" ubuntu@ec2-xx-xxx-xxx-x.compute-1.amazonaws.com
    ```
7. **Setup Steps**: Run this command for every new terminal you SSH into. It will move you to the project directory, set environment variables, and activate the virtual environment.
   ```bash
   cd flight-delays/ && source set_env.sh && pipenv shell
   ```
8. **Orchestration with Prefect**: Let's connect to the Prefect Server. If you are running locally, please run only the second command.
    ```bash
    prefect config set PREFECT_SERVER_API_HOST=0.0.0.0
    ```
    ```bash
    prefect server start
    ```
    ![](https://github.com/adelhassen/flight-delays/blob/7327c4bdc83599f4ff278b9e6392af31ba0599e3/images/prefect_runs.png)


You should have three terminals running: Docker-deployed model API endpoint, MLFLow server, and Prefect server.
    

### Running the Pipeline

1. **Setup Steps**: Open a new terminal and SSH into EC2 then run the command to move to the project directory, set environment variables, and activate the virtual environment.
   ```bash
   cd flight-delays/ && source set_env.sh && pipenv shell
   ```
2. **Main Script**: Run `src/main.py` which handles data reading, feature engineering, model training, hyperparameter tuning, evaluation, selecting and registering the best model, and creating monitoring reports. Running this command will serve the model but not trigger a run in Prefect. The workflow is fully deployed and runs on a schedule, but we will trigger a run to test it now.
   ```bash
   python src/main/py
   ```
3. **Setup Steps**: Open a new terminal and SSH into EC2 then run the command to move to the project directory, set environment variables, and activate the virtual environment.
   ```bash
   cd flight-delays/ && source set_env.sh && pipenv shell
   ```
4. **Deploy Prefect Flow**: Trigger a Prefect run which will execute `src/main.py` by running the following command:
   ```bash
   prefect deployment run 'main-flow/train_flight_delay_model'
   ```
   ![](https://github.com/adelhassen/flight-delays/blob/7327c4bdc83599f4ff278b9e6392af31ba0599e3/images/prefect_flow.png)
5. **API Endpoint**: Use `src/example.py` to test the Docker API endpoint that provides delay predictions.
   ```bash
   python src/example.py
   ```
  ![](https://github.com/adelhassen/flight-delays/blob/7327c4bdc83599f4ff278b9e6392af31ba0599e3/images/evidently_classification.png) 

### Testing

- **Unit Tests**: Test data reading and feature engineering steps conducted in `src/main.py`.
  ```bash
  python tests/unit_tests.py
  ```
- **Integration Test**: Test the connection and predictions from the Docker-deployed model.
  ```bash
  python tests/integration_test.py
  ```
### Improvements and Next Steps

- **Improve Model**: Use more data, create historical features, and create more temporal features.
- **Infrastructure Management**: To provision and manage infrastructure.
- **CI/CD**: Automate build, test, and deploy processes.
- **Document and Comment Code**: Add information to make code and repo easier to understand.
- **Multi-container Docker**: Create containers to host all services.

### Acknowledgments
- Bureau of Transportation Statistics for providing the dataset.
- Open source libraries and tools used in the project: MLflow, Prefect, Evidently, Docker, Hyperopt, XGBoost, and more.

