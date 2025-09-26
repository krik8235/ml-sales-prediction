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

- [Key Features](#key-features)
- [The System Architecture](#the-system-architecture)
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
- [Data CI/CD Automation with Prefect](#data-cicd-automation-with-prefect)
- [Contributing](#contributing)
  - [Pre-commit hooks](#pre-commit-hooks)
- [Trouble Shooting](#trouble-shooting)
- [Ref. Repository Structure](#ref-repository-structure)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<hr />

## Key Features

A dynamic pricing system for an online retailer using predictions served by ML models:

   - A multi-layered Feedforward Neural Network,

   - A Light GBM regressor and

   - An Elastic Net,

hosted on the containerized serverless architecture.


## The System Architecture

<img src='https://res.cloudinary.com/dfeirxlea/image/upload/v1754580450/portfolio/zafielthrjkxrdgsd5ig.png'/>

<br>

The system design focuses on the following points:

- The application is fully containerized on **Docker** for universal accessibility.
- The container image is stored in **Elastic Container Registry (ECR)**.
- **API Gateway's REST API endpoints** trigger an event to invoke the Lambda function.
- **Lambda function** loads the container image from ECR and perform inference.
- Trained models, processors, and input features are stored in the **S3** buckets.
- A **Redis client** caches analytical data and past prediction results stored in ElastiCache.

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

The API is available at `http://localhost:5002`.


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
- To completely refresh the environement, run the following commands:

```bash
rm -rf .venv
rm -rf uv.lock
uv cache clean
uv venv
source .venv/bin/activate
uv sync
```

<hr />

## Data CI/CD Automation with Prefect

1. Run Prefect server in local

```bash
uv run prefect server start
export PREFECT_API_URL="http://127.0.0.1:4200/api"
```

2. Create Docker image

```bash
uv run src/data_handling/prefect_deploy.py
```


3. Run the Prefect worker

```bash
# add a user group USER to the docker
sudo dscl . -append /Groups/docker GroupMembership $USER

prefect worker start --pool <YOUR-WORKER-POOL-NAME>
```


4. Create a flow run for deployment.

```bash
prefect deployment run 'etl-pipeline/deploy-etl-pipeline'
```

You can find the dashboard at http://127.0.0.1:4200/dashboard.

Ref. [Prefect official documentation - deploy via Python](https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python)

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
└── .dvc/                               # dvc version control
│
└── data/                               # version tracked by dvc
│     └──raw/                           # stores raw data
│     └──preprocessed/                  # stores processed data after imputation and engineering
│
└── preprocessors/                      # version tracked by dvc
│
└── models/             [.gitignore]    # stores serialized model after training and tuning
│     └──dfn/                           # deep feedforward network
│     └──gbm/                           # light gbm
│     └──en/                            # elastic net
│     └──production/                    # models to be stored in S3 for production use
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
│     └──main.py                        # main script to run the inference locally
│
└──app.py                               # Flask application (API endpoints)
│
└──pyproject.toml                       # project configuration
│
└──.env                [.gitignore]     # environment variables
│
└──uv.lock                              # dependency locking
│
└──Dockerfile                           # for Docker container image
└──.dockerignore
│
└──requirements.txt
│
└──.python-version                      # python version locking (3.12)
│
└──dvc.yaml                             # config for dvc commands
└──dvc.lock
```
