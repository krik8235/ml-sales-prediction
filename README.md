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
- [Dependencies](#dependencies)
- [Starting the Project](#starting-the-project)
  - [Installing the package manager](#installing-the-package-manager)
  - [Installing dependencies](#installing-dependencies)
  - [Adding env secrets to .env file](#adding-env-secrets-to-env-file)
  - [Tuning the models from scratch (Optional)](#tuning-the-models-from-scratch-optional)
  - [Running API endpoints](#running-api-endpoints)
  - [Updating and Running Docker image](#updating-and-running-docker-image)
- [AWS Ecosystem](#aws-ecosystem)
  - [Serverless Function: Lambda](#serverless-function-lambda)
  - [API Gateway](#api-gateway)
  - [Cache Storage: ElastiCache](#cache-storage-elasticache)
  - [Model Store, Feature Store: S3](#model-store-feature-store-s3)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
  - [Steps](#steps)
  - [Package Management with uv](#package-management-with-uv)
  - [Pre-Commit Hooks](#pre-commit-hooks)
- [Trouble Shooting](#trouble-shooting)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<hr />

## Key Features

A dynamic pricing system for an online retailer using predictions served by machine learning models:

   - A multi-layered Feedforward Neural Network,

   - A Light GBM regressor and

   - An Elastic Net,

hosted on the containerized serverless architecture.



<hr />


## Dependencies

**Inference**

* numpy
* pandas

* category-encoders

* torch
* tensorflow
* scikit-learn
* lightgbm

* optuna: Hyperparameter optimization framework for PyTorch models
* scikit-optimize:  Hyperparameter optimization framework for Scikit-learn models


**API**

* Flask
* flask-cors


**Deployment**

* python-dotenv: Environment secret control
* redis: For cache storage
* waitress: WSGI server for the Flask application.


**AWS Ecosystem**

* boto3: AWS SDK for Python, managing AWS services - S3 (for storage), Lambda
* aws-wsgi: Connecter between JSON (AWS API Gateway) and WSGI (Flask) response.


**CI/CD**

* uv: Python package installer and resolver
* pre-commit: Manage and maintain pre-commit hooks


<hr />

## Starting the Project

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
uv sync --all-extras
```

- AssertionError/module mismatch errors: Set up default Python version using `.pyenv`

```bash
pyenv install 3.12.8
pyenv global 3.12.8  (optional: `pyenv global system` to get back to the system default ver.)
uv python pin 3.12.8
echo 3.12.8 >> .python-version
```


### Adding env secrets to .env file

Create `.env` file in the project root and add secret vars following `.env.sample` file.


### Tuning the models from scratch (Optional)

```bash
uv venv
source .venv/bin/activate
uv run src/main.py --cache-clear
```


### Running API endpoints

```bash
uv run app.py --cache-clear
```

The API is available at `http://localhost:5002`.


### Updating and Running Docker image

```bash
docker build -t <APP NAME> .
docker run -p 5002:5002 -e ENV=local <APP NAME> app.py
```

Replace `<APP NAME>` with an app name of your choice.


## AWS Ecosystem

### Serverless Function: Lambda

- Push the containe image to the AWS Elastic Container Registory (ECR)

```bash
# add a tag
docker tag <YOUR ECR NAME>:<YOUR ECR VERSION> <URI>.dkr.ecr.<REGION>.amazonaws.com/<ECR NAME>:<VERSION>

# push to the ECR
docker push <URI>.dkr.ecr.<REGION>.amazonaws.com/<ECR NAME>:<VERSION>
```

- On the Lambda console, create a new Lambda function (select container image type and pushed container image URI).


### API Gateway

- Go to the API gateway console, add resources aligned with the API endpoints on the `app.py`.
- Deploy the API endpoints to the stage.
- Go back to the Lambda console, add a trigger to the API Gateway.


### Cache Storage: ElastiCache

- Open up `AWS ElastiCache` to create Redis OSS cache.
- Connect the cache storage with the `VPC`.
- Add a `security group` for the ElastiCache with outbound permissions to the Lambda URI.
- Make sure the `IAM` role for the Lambda function is authorized the ElastiCache access.

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



### Model Store, Feature Store: S3

- Uses `AWS S3`.
- Similar to the cache storage, connect it with the `VPC`.
- Make sure the `IAM` role for the Lambda function is authorized the AWS S3 access (full access).



<hr />


## Repository Structure

```
.
.venv/                  [.gitignore]    # stores uv venv
│
└── data/               [.gitignore]
│     └──raw/                           # stores raw data before engineering
│     └──preprocessed/                  # stores preprocessed data
│
└── models/             [.gitignore]    # stores serialized model after training and tuning
│     └──dfn/                           # deep feedforward network
│     └──gbm/                           # light gbm
│     └──en/                            # elastic net
│     └──production/                    # models to be stored in S3 for production use
|
└── notebooks/                          # stores experiment notebooks
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
└──Dockerfile
└──.dockerignore
│
└──requirements.txt
│
└──.python-version                      # python version locking (3.12)
```


<hr />

## Contributing

### Steps

1. Create your feature branch (`git checkout -b feature/your-amazing-feature`)

2. Create amazing features.

3. Pull the latest version of source code from the main branch (`git pull origin main`) *Address conflicts if any.

4. Commit your changes (`git add .` / `git commit -m 'Add your-amazing-feature'`)

5. Push to the branch (`git push origin feature/your-amazing-feature`)

6. Open a pull request


**Optional**

* Flag with `#! REFINEME` for any improvements needed and `#! FIXME` for any errors.


### Package Management with uv

- Add a package: `uv add <package>`
- Remove a package: `uv remove <package>`
- Run a command in the virtual environment: `uv run <command>`
- To completely refresh the environement, run the following commands:

```bash
$ rm -rf .venv
$ rm -rf uv.lock
$ uv cache clean
$ uv venv
$ source .venv/bin/activate
$ uv sync
```


### Pre-Commit Hooks

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

* Issues related to dependencies: `rm -rf uv.lock`, `uv cache clean`, `uv venv`, and run `uv pip install -r requirements.txt -v`.

* Issues related to `Python quit unexpectedly`: Check [this stackoverflow article](https://stackoverflow.com/questions/59888499/macos-catalina-python-quit-unexpectedly-error).

* `reportMissingImports` error from pyright after installing the package: This might occur when installing new libraries while VSCode is running. Open the command pallete (ctrl + shift + p) and run the Python: Restart language server task.
