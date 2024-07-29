# Flight Delay Predictor

## Project Description

The Flight Delay Predictor project is designed to build a robust, industry-grade machine learning pipeline for predicting flight delays. A flight is considered delayed if it is 15 minutes or more late, as per the United States Federal Aviation Administration (FAA) standards. The prediction model utilizes historical flight data to make accurate predictions about future flight delays.

Flight delays are significant for several reasons. They affect customer satisfaction, with passengers experiencing frustration and inconvenience due to unexpected wait times. Economically, delays can lead to increased operational costs for airlines and airports, including costs related to fuel, crew, and ground services. Moreover, delayed flights can cause cascading effects across the entire flight network, leading to broader disruptions. Therefore, accurately predicting flight delays can help in better resource allocation, improved customer satisfaction, and overall operational efficiency.

## Objective

The primary objective of this project is to develop a robust machine learning pipeline to predict flight delays using historical data. The focus is on creating an industry-grade machine learning workflow that includes essential MLOps components such as:

- **Experiment Tracking**: Utilizing MLFlow to track and manage experiments, logging parameters, metrics, and models.
- **Workflow Orchestration**: Using Prefect to manage and orchestrate the different tasks in the pipeline, ensuring smooth execution and monitoring.
- **Model Deployment and Monitoring**: Deploying the best model and setting up monitoring using Evidently to track data drift and model performance over time.
- **Environment Management**: Using virtual environments and Docker to maintain complex working environments.

A key consideration in this project is to avoid any data leakage by ensuring that only features available at the time of prediction are used for training the model. This ensures that the model remains reliable and valid for real-world applications.

## Data

The dataset used for this project is the Bureau of Transportation Statistics’ “Reporting Carrier-On Time Performance” dataset. This dataset is available at [this link](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) and contains monthly data from 1987. It includes various details such as:

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
- **Distance** (considered initially but dropped as it is difficult to predict)

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

   - Ubuntu
   - 64-bit (x86)
   - t2.xlarge (t2.micro on free tier should be enough too)
   - Configure storage: 30GiB
2. **Configure Security Group**: Ports 4200 (Prefect), 5000 (MLFlow), and 9696 (Flask) needed to be opened in addition to port 22 (SSH) for source 0.0.0.0/0
3. **Set up EC2 Instance**:
   - Download python 3.11.5 from anaconda to avoid any python version issues using:
     ```bash
     wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
     ```
  - Follow instructions at [this link](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/01-intro) to set up Docker properly.
4. **Create S3 Bucket**: Create an S3 bucket using AWS CLI.
5. **Clone the Repository**: Use SSH protocol to clone the repo.
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
3. **Setting Environment Variables**: Configure environment variables as needed.
   
5. **Building and Running Docker Image**: Build and run the Docker image for a consistent and isolated environment.
6. **Virtual Environment**: Create and activate a virtual environment for running specific scripts.

### Running the Pipeline

1. **Main Script**: Run `main.py` which handles data reading, feature engineering, model training, and evaluation.
2. **Tracking with MLFlow**: Track experiments and model performance using MLFlow.
3. **Monitoring with Evidently**: Generate monitoring reports to check data drift and classification performance.
4. **API Endpoint**: Use `example.py` to test the API endpoint that provides delay predictions.

### Testing

- **Unit Tests**: Test data reading and feature engineering steps.
- **Integration Test**: Test the connection and predictions from the Docker-deployed model.


