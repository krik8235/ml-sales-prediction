# Overview

![MIT license](https://img.shields.io/badge/License-MIT-green)
![python ver](https://img.shields.io/badge/Python-3.12-purple)
![pyenv ver](https://img.shields.io/badge/pyenv-2.5.0-orange)

Tunes and assesses deep learning models.

**Visit:**

- [UI](https)
- [Portfolio](https://kuriko.vercel.app)

<hr />

## Table of Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Starting the Project](#starting-the-project)
  - [Installing package manager](#installing-package-manager)
  - [Installing dependencies](#installing-dependencies)
  - [Adding env secrets to .env file](#adding-env-secrets-to-env-file)
  - [Run!](#run)
- [Contributing](#contributing)
  - [Steps](#steps)
  - [Package Management with uv](#package-management-with-uv)
  - [Pre-Commit Hooks](#pre-commit-hooks)
- [Trouble Shooting](#trouble-shooting)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

<hr />

## Key Features

- Creates datasets for supervised training
- Tunes hyperparameters
- Evaluates created models


<hr />


## Dependencies

**Data Handling**


**Model Tuning**

- imblearn>=0.0
- keras-tuner>=1.4.7

- pandas>=2.2.3
- scikeras>=0.13.0
- scikit-learn==1.5.2
- tensorflow>=2.19.0
- torch
- optuna

**Visualization**

- matplotlib>=3.10.3
- seaborn>=0.13.2

**Storage**

* S3: Production storage for preprocessor
* joblib: local storage

* Redis: Production storage for models
* pickle


**Deployment**

* awsgi: connecter between JSON (AWS API Gateway) and WSGI response (Flask)
* waitress: local server
* [uv](https://docs.astral.sh/uv/): Python package installer and resolver
* [pre-commit](https://pre-commit.com/): Manage and maintain pre-commit hooks


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
|
└── notebooks/                          # stores experiment notebook files
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
app.py                                  # flask app
│
pyproject.toml                          # project configuration
│
.env                    [.gitignore]    # environment variables
│
uv.lock                                 # dependency locking
│
Dockerfile                          
.dockerignore
│
requirements.txt
│
.python-version                         # python version locking (3.12)
```


<hr />

## Starting the Project

### Installing package manager

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


### Run

   ```bash
   uv venv
   source .venv/bin/activate
   uv run main.py --cache-clear
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

* Database connection issues: Check if the Chroma DB is properly initialized and accessible.

* Memory errors: If processing large contracts, you may need to increase the available memory for the Python process.

* Issues related to dependencies: `rm -rf uv.lock`, `uv cache clean`, `uv venv`, and run `uv pip install -r requirements.txt -v`.

* Issues related to `torch` installation: Add optional dependencies by `uv add versionhq[torch]`.

* Issues related to agents and other systems: Check `.logs` directory located at the root of the project directory for error messages and stack traces.

* Issues related to `Python quit unexpectedly`: Check [this stackoverflow article](https://stackoverflow.com/questions/59888499/macos-catalina-python-quit-unexpectedly-error).

* `reportMissingImports` error from pyright after installing the package: This might occur when installing new libraries while VSCode is running. Open the command pallete (ctrl + shift + p) and run the Python: Restart language server task.

