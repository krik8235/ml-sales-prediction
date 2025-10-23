# ML System for Price Prediction

![MIT license](https://img.shields.io/badge/License-MIT-green)
![python ver](https://img.shields.io/badge/Python-3.12-purple)
![pyenv ver](https://img.shields.io/badge/pyenv-2.5.0-orange)



**Visit**

- [User Interface](https://kuriko-iwai.vercel.app/online-commerce-intelligence-hub)
- [Related Article]()


## Table of Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [The Project Overview](#the-project-overview)
- [The System Architecture](#the-system-architecture)
  - [Core AWS Resources](#core-aws-resources)
  - [ML Lineage Integration](#ml-lineage-integration)
  - [CI/CD Pipeline Integration](#cicd-pipeline-integration)
- [The Inference](#the-inference)
  - [Models Trained](#models-trained)
  - [Performance Validation Metrics](#performance-validation-metrics)
  - [ML Techniques Implemented](#ml-techniques-implemented)
- [Quick Start](#quick-start)
  - [Installing the package manager](#installing-the-package-manager)
  - [Installing dependencies](#installing-dependencies)
  - [Adding env secrets to .env file](#adding-env-secrets-to-env-file)
  - [Running API endpoints](#running-api-endpoints)
- [Tuning](#tuning)
  - [Feature engineering](#feature-engineering)
  - [Model retraining](#model-retraining)
  - [Tuning from scratch (with caution)](#tuning-from-scratch-with-caution)
  - [Tuning for stockcode (with caution)](#tuning-for-stockcode-with-caution)
- [Deployment](#deployment)
  - [Publishing Docker image](#publishing-docker-image)
  - [Connecting cache storage](#connecting-cache-storage)
- [Package Management](#package-management)
- [Data CI/CD Automation](#data-cicd-automation)
  - [Managing **DVC** Pipeline](#managing-dvc-pipeline)
  - [Schedule run with **Prefect**](#schedule-run-with-prefect)
  - [EventlyAI Reports](#eventlyai-reports)
- [Contributing](#contributing)
  - [Pre-commit hooks](#pre-commit-hooks)
- [Trouble Shooting](#trouble-shooting)
- [Ref. Repository Structure](#ref-repository-structure)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<hr />


## The Project Overview

This project describes the development and deployment of a serverless machine learning system designed to recommend optimal retail pricing which maximizes product sales.

The system aims to allow mid-sized retailers to compete effectively with larger players.

<hr />

## The System Architecture

The architecture establishes a scalable, serverless microservice using **AWS Lambda**, triggered by an API Gateway.

The prediction logic is fully containerized via **Docker**, which stored in **AWS ECR**.

Trained models and features are centrally managed in **S3**, while **ElastiCache (Redis)** provides a low-latency caching layer for historical data and predictions.

This event-driven setup ensures automatic scaling and pay-per-use efficiency.

<div  align="center">
<img
  src='https://cdn.hashnode.com/res/hashnode/image/upload/v1760860401076/ae657214-fe63-4de0-a9ca-a1033bff2907.png'
  alt='[Figure A. The system architecture (Created by Kuriko IWAI)]'
/>
</div>

**Figure A.** The system architecture (Created by Kuriko IWAI)

### Core AWS Resources

The infrastructure leverages AWS ecosystem:

* **Docker / AWS ECR as Microservice container**: Packages the prediction logic and dependencies. AWS Lambda pulls the image from ECR for consistent, universal deployment.

* **AWS API Gateway as REST API endpoint**: Routes external client-side UI requests (via a Flask application) to trigger the Lambda function.

* **AWS Lambda as inference**: Executes the inference function, loading the container, models, and features to calculate price recommendations.

* **AWS S3 as storage & feature store**: Stores raw features, trained model artifacts, processors, and DVC metadata for ML Lineage.

* **AWS ElastiCache and Redis client as caching layer**: Stores cached analytical data and past price predictions to improve latency and resource efficiency.


### ML Lineage Integration

A dedicated ML Lineage process is integrated using **DVC (Data Version Control)** and scheduled by **Prefect**, an open-source workflow scheduler, running weekly.

* **Lineage Scope (DVC):** DVC tracks the entire lifecycle, including **Data** (ETL/preprocessing), **Experiments** (hyperparameter tuning/validation), and **Models/Prediction** (artifacts, metrics).

* **Data Quality Gate:** Models must pass stringent quality checks before being authorized to serve predictions:

    * **Data Drift Tests:** Handled by **Evently AI** to identify shifts in data distribution.

    * **Fairness Tests:** Measures SHAP scores and other custom metrics to ensure the model operates without bias.

* **Automation:** **Prefect** triggers DVC *weekly* to check for updates in data or scripts and executes the full lineage process if changes are detected, ensuring continuous model freshness and quality.


### CI/CD Pipeline Integration

The infrastructure and model lifecycle are managed through a robust MLOps practice using a CI/CD pipeline integrated with GitHub.

* **Code Lineage:** Handled by **GitHub**, protected by **branch rules** and enforced **pull request reviews**.

* **Source:** Code commit to GitHub triggers a **GitHub Actions workflow**.

* **Testing & Building:** Automated GitHub Actions run:

    * **Test Phase:** Runs PyTest (unit/integration tests), SAST (Static Application Security Testing), and SCA (Software Composition Analysis) for dependencies using **Synk**.

    * **Build Phase:** If tests pass, **AWS CodeBuild** is triggered to build the Docker image and push it to ECR.

* **Deployment:** A **human review phase** is mandatory between the build and deployment. After approval, another GitHub Actions workflow is *manually* triggered to deploy the updated Lambda function to staging or production.


<div align="center">
<img
  src='https://cdn.hashnode.com/res/hashnode/image/upload/v1760860467508/896750af-22d3-45ca-9fc1-25b96f77ab0b.png'
  alt='[Figure B. The CI/CD pipeline (Created by Kuriko IWAI)]'
/>
</div>

**Figure B.** The CI/CD pipeline (Created by Kuriko IWAI)

<hr />

## The Inference

The process is designed for consistent, automated data and model management through MLOps tools:

1. The client UI sends a price recommendation request via the **Flask** application.

2. The request hits the **API Gateway** endpoint.

3. API Gateway triggers the **AWS Lambda function**.

4. Lambda loads the Docker container from **ECR**.

5. The function retrieves the latest features and model artifacts from **S3** and checks **ElastiCache/Redis** for cached data.

6. The primary model performs inference on the logarithmically transformed quantity data and returns the optimal price recommendation.


### Models Trained

The system utilizes multiple machine learning models to ensure prediction redundancy and reliability. The primary mechanism involves predicting the **quantity of product sold** at a given price point.

* **Primary Model:** Multi-layered feedforward network (PyTorch).

    * **Role:** Serves first-line predictions.

    * **Tuning:** Tuned via **Optuna's Bayesian Optimization** (with grid search fallback).

* **Backup Models:** LightGBM, SVR, and Elastic Net (Scikit-Learn).

    * **Role:** Prioritized backups used if the primary model fails, ensuring redundancy.

    * **Tuning:** Tuned via the **Scikit-Optimize framework**.


### Performance Validation Metrics

Models are evaluated using metrics corresponding to both transformed and original data, where a lower value indicates better performance.

* **For Logged Values:** **Mean Squared Error (MSE).**

* **For Actual (Original) Values:** **Root Mean Squared Log Error (RMSLE)** and **Mean Absolute Error (MAE)**.


### ML Techniques Implemented

1. **Logarithmic Transformation (Data Preprocessing):**

    * Quantity data is logged before training and prediction to achieve a denser data distribution. This is crucial for normalizing skewed data and reducing the influence of extreme values (outliers), enabling all models to learn underlying patterns more effectively.

2. **Model Diversity and Redundancy:**

    * The system employs a hybrid approach combining a **Multi-layered Feedforward Network (Deep Learning)** as the primary predictor with diverse **Traditional Machine Learning Models** (LightGBM, SVR, Elastic Net) as backups.

    * This multi-model inference strategy provides a failover mechanism, ensuring high availability by loading a prioritized backup model if the primary fails.

3. **Advanced Hyperparameter Optimization:**

    * **Bayesian Optimization (Optuna)** is utilized for the deep learning primary model, efficiently searching the hyperparameter space to find optimal settings (with a grid search fallback available).

    * The backup Scikit-Learn models are tuned using the **Scikit-Optimize framework**.

4. **Production Quality Gates:**

    * To ensure the model remains reliable in a dynamic retail environment, the ML Lineage process incorporates necessary quality checks as techniques:

        * **Data Drift Testing (Evently AI):** Continuously identifies shifts in data distributions in production that could compromise the model's generalization capabilities.

        * **Fairness Testing:** Validates that the model operates without unwanted bias across different features or segments before being authorized to serve predictions.


<hr />

## Quick Start

### Installing the package manager

For MacOS:

```bash
brew install uv
```

For Ubuntu/Debian:

```bash
sudo apt-get install uv
```


### Installing dependencies

```bash
uv venv
source .venv/bin/activate
uv lock --upgrade
uv sync
```

or

```bash
pip env
pip install -r requirements.txt
```

- AssertionError/module mismatch errors: Set up the default Python version using `.pyenv`

```bash
pyenv install 3.12.8
pyenv global 3.12.8  (optional: `pyenv global system` to get back to the system default ver.)
uv python pin 3.12.8
echo 3.12.8 >> .python-version
```


### Adding env secrets to .env file

Create `.env` file in the project root and add secret vars following `.env.sample` file.


### Running API endpoints

```bash
uv run app.py --cache-clear
```

The API endpoint is available at `http://localhost:5002`.


<hr />


## Tuning

### Feature engineering

- The `data_handling` folder contains data relerated scripts.

- After updating scripts, run:

```bash
uv run src/data_handling/main.py
```


### Model retraining

- The retrain script will load the serialized model in the model store, then retrain with new data, and upload the retrained model to the model store.

```bash
uv run src/retrain.py
```


### Tuning from scratch (with caution)

- The main script will run feature engineering and model tuning from scratch, and update instances saved in model store and feature store in S3.

```bash
uv run src/main.py
```

- Before running the script, make sure testing the new script in notebook.

### Tuning for stockcode (with caution)

- Run the main script for stockcode to tune the model based on training data of specific stockcode.

```bash
uv run src/main_stockcode.py {STOCKCODE} --cache-clear
```

## Deployment

### Publishing Docker image

- Build and run Docker image:

```bash
docker build -t <APP NAME> .
docker run -p 5002:5002 -e ENV=local <APP NAME> app.py
```

Replace `<APP NAME>` with an app name of your choice.


- Push the Dokcer image to AWS Elastic Container Registory (ECR)

```bash
# tagging
docker tag <YOUR ECR NAME>:<YOUR ECR VERSION> <URI>.dkr.ecr.<REGION>.amazonaws.com/<ECR NAME>:<VERSION>

# push to the ECR
docker push <URI>.dkr.ecr.<REGION>.amazonaws.com/<ECR NAME>:<VERSION>
```


### Connecting cache storage

- Cache storage (ElastiCache) run on Redis engine.

- To test the connection locally:

```bash
redis-cli --tls -h clustercfg.{REDIS_CLUSTER}.cache.amazonaws.com -p 6379 -c
```

- To flush all caches (WITH CAUTION):

```bash
redis-cli -h clustercfg.{REDIS_CLUSTER}.cache.amazonaws.com -p 6379 --tls

# once connected, flush all data
FLUSHALL

# or flush specific database (if using multiple databases)
FLUSHDB
```

<hr />


## Package Management

- Add a package: `uv add <package>`
- Remove a package: `uv remove <package>`
- Run a command in the virtual environment: `uv run <command>`
- To completely refresh the environement:

```bash
rm -rf .venv
rm -rf uv.lock
uv cache clean
uv venv
source .venv/bin/activate
uv sync
```

<hr />

## Data CI/CD Automation

### Managing **DVC** Pipeline

- Run the DVC pipeline and push the updated data to cache:

```bash
dvc repro

# add updated lock file
git add dvc.lock
git commit -m'updated'
git push

# dvc push
dvc push
```

- Force run all stages in the DVC pipeline including stages without any updates:

```bash
dvc repro -f
```

- Run the DVC pipeline for a specific stockcode:

```bash
dvc repro etl_pipeline_stockcode -p stockcode={STOCKCODE}
dvc repro preprocess_stockcode -p stockcode={STOCKCODE}
```


- Train the model using data from the DVC pipeline:

```bash
uv run src/main_stockcode.py {STOCKCODE}
dvc add models/production/dfn_best_{STOCKCODE}.pth
dvc push

rm models/production/dfn_best_{STOCKCODE}.pth
```

- To check the cache status explicitly:

```bash
dvc data status --not-in-remote
```

- To edit the DVC pipeline, update `dvc.yaml` and `params.yaml` for parameter updates.


### Schedule run with **Prefect**

- Run Prefect server in local

```bash
uv run prefect server start
```


```bash
export PREFECT_API_URL="http://127.0.0.1:4200/api"
```

- Deploy the weekly DVC pipeline run (from the Docker container)

```bash
uv run src/prefect_flows.py
```


- Test run the Prefect worker

```bash
# add a user group USER to the docker
sudo dscl . -append /Groups/docker GroupMembership $USER

prefect worker start --pool <YOUR-WORKER-POOL-NAME>
```


- Create a flow run for deployment.

```bash
prefect deployment run 'etl-pipeline/deploy-etl-pipeline'
```


Ref.

- Prefect dashboard: http://127.0.0.1:4200/dashboard.
- [Prefect official documentation - deploy via Python](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python)


### EventlyAI Reports

- [Report](https://app.evidently.cloud/v2/projects/01998e47-e3d7-7ee7-ae5e-b6fdf3f80fff/reports)


<hr />


## Contributing


1. Create your feature branch (`git checkout -b feature/your-amazing-feature`)

2. Create a feature.

3. Pull the latest version of source code from the main branch (`git pull origin main`) *Address conflicts if any.

4. Commit your changes (`git add .` / `git commit -m 'Add your-amazing-feature'`)

5. Push to the branch (`git push origin feature/your-amazing-feature`)

6. Open a pull request

* Flag `#REFINEME` for any improvement needed and `#FIXME` for any errors.


### Pre-commit hooks

Pre-commit hooks runs hooks defined in the `pre-commit-config.yaml` file before every commit.

To activate the hooks:

1. Install pre-commit hooks:

```bash
uv run pre-commit install
```

2. Run pre-commit checks manually:

```bash
uv run pre-commit run --all-files
```

Pre-commit hooks help maintain code quality by running checks for formatting, linting, and other issues before each commit.

* To skip pre-commit hooks

```bash
git commit --no-verify -m "your-commit-message"
```


<hr />

## Trouble Shooting

Common issues and solutions:

* API key errors: Ensure all API keys in the `.env` file are correct and up to date. Make sure to add `load_dotenv()` on the top of the python file to apply the latest environment values.

* Data warehouse connection issues: Check logs on AWS consoles, CloudWatch. Check if `.env` and Lambda's environment configuration are correct.

* Memory errors: If processing large contracts, you may need to increase the available memory for the Python process.


* Issues related to `Python quit unexpectedly`: Check [this stackoverflow article](https://stackoverflow.com/questions/59888499/macos-catalina-python-quit-unexpectedly-error).

* `reportMissingImports` error from pyright after installing the package: This might occur when installing new libraries while VSCode is running. Open the command pallete (ctrl + shift + p) and run the Python: Restart language server task.

<hr >

## Ref. Repository Structure

```
.
└── .venv/              [.gitignore]    # stores uv venv
│
└── .github/                            # infrastructure ci/cd
│
└── .dvc/                               # dvc folder - cache, tmp, config
│
└── data/               [dvc track]     # version tracked by dvc
└── preprocessors/      [dvc track]     # version tracked by dvc
└── models/                             # stores serialized model after training and tuning
│     └──dfn/                           # deep feedforward network
│     └──gbm/                           # light gbm
│     └──en/                            # elastic net
│     └──production/    [dvc track]     # models to be stored in S3 for production use
└── reports/            [dvc track]     # reports on data drift, shap values
└── metrics/            [dvc track]     # model evaluation metrics (mae, mse, rmsle)
|
└── notebooks/                          # stores experimentation notebooks
│
└── src/                                # core functions
│     └──_utils/                        # utility functions
│     └──data_handling/                 # functions to engineer features
│     └──model/                         # functions to train, tune, validate models
│     │     └── sklearn_model
│     │     └── torch_model
│     │     └── ...
│     └──main.py                        # main script to preform inference locally (without dvc repro)
│
└── app.py                              # flask application (API endpoints)
│
└── tests/                              # pytest scripts and config
└── pytest.ini
│
└── pyproject.toml                      # project config
│
└── .env                [.gitignore]    # environment variables
│
└── uv.lock                             # dependency locking
│
└── .python-version                     # python version locking (3.12)
│
└── Dockerfile.lambda.local             # docker config
└── Dockerfile.lambda.production
└── .dockerignore
└── requirements.txt
│
└── dvc.yaml                            # dvc pipeline config
└── params.yaml
└── .dvcignore
└── dvc.lock
│
└── .pre-commit-config.yaml             # pre-commit check config
└── .synk                               # synk (dependency and code scanning) config
```

*All images and contents, unless otherwise noted, are by the author.*
