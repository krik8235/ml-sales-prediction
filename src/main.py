import os
import torch
import warnings
import pickle
import joblib
import numpy as np
import lightgbm as lgb # type: ignore
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from skopt.space import Real, Integer, Categorical # type: ignore
from dotenv import load_dotenv # type: ignore

import src.data_handling as data_handling
import src.model.torch_model as t
import src.model.sklearn_model as sk
from src._utils import s3_upload

warnings.filterwarnings("ignore", category=RuntimeWarning)

# paths
PRODUCTION_MODEL_FOLDER_PATH = 'models/production'
DFN_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'dfn_best.pth')
GBM_FILE_PATH =  os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'gbm_best.pth')
SVR_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'svr_best.pth')
EN_FILE_PATH = os.path.join(PRODUCTION_MODEL_FOLDER_PATH, 'en_best.pth')

PREPROCESSOR_PATH = 'preprocessors/column_transformer.pkl'

# models
sklearn_models = [
    {
        'model_name': 'en',
        'base_model': ElasticNet,
        'search_space_grid': dict(
            alpha=np.logspace(np.log10(1e-3), np.log10(10.0), num=5).tolist(),
            l1_ratio=[0.1, 0.5, 1.0],
            max_iter=[10000, 50000],
            tol=[1e-5, 1e-3],
            selection=['cyclic', 'random'],
            fit_intercept=[True],
            random_state=[42]
        ),
        'search_space_bayesian': [
            Real(1e-4, 10.0, 'log-uniform', name='alpha'),
            Real(0.01, 1.0, 'uniform', name='l1_ratio'),
            Integer(10000, 50000, name='max_iter'),
            Real(1e-6, 1e-2, 'log-uniform', name='tol'),
            Categorical(['cyclic', 'random'], name='selection'),
            Categorical([True,], name='fit_intercept'),
            Integer(12, 100, name='random_state'),
        ]
    },
    {
        'model_name': 'svr',
        'base_model': SVR,
        'search_space_grid': dict(
            kernel=['linear', 'poly', 'rbf', 'sigmoid'],
            degree=[1, 3],
            gamma=["scale"],
            tol=[1e-3],
            C=[1.0],
            epsilon=[0.1],
            max_iter=[1000],
        ),
        'search_space_bayesian': [
            Categorical(['linear', 'poly', 'rbf', 'sigmoid'], name='kernel'),
            Integer(1, 100, name='degree'),
            Categorical(['scale', 'auto'], name='gamma'),
            Real(1e-4, 10.0, 'log-uniform', name='coef0'),
            Real(1e-6, 1e-1, 'log-uniform', name='tol'),
            Real(0.01, 100, 'uniform', name='C'),
            Real(1e-6, 1e-1, 'log-uniform', name='epsilon'),
            Integer(1000, 50000, name='max_iter'),
        ]
    },
    {
        'model_name': 'gbm',
        'base_model': lgb.LGBMRegressor,
        'search_space_grid': dict(
            boosting_type=['gbdt'],
            num_leaves=[31, 63],
            max_depth=[-1],
            learning_rate=[0.01, 0.05],
            n_estimators=[2000],
            subsample_freq=[1],
            subsample=[0.8, 1.0],
            colsample_bytree=[0.8, 1.0],
            min_child_samples=[20],
            min_child_weight=[0.001],
            reg_alpha=[0.1],
            reg_lambda=[0.1],
            random_state=[42],
            n_jobs=[-1],
        ),
        'search_space_bayesian': [
            Categorical(['gbdt',], name='boosting_type'),
            Integer(10, 500, name='num_leaves'),
            Integer(3, 15, name='max_depth'),
            Integer(5, 500, name='min_child_samples'),
            Real(1e-4, 10.0, 'log-uniform', name='min_child_weight'),
            Real(1e-3, 0.3, 'log-uniform', name='learning_rate'),
            Integer(100, 2000, name='n_estimators'),
            Real(0.6, 1.0, 'uniform', name='subsample'),
            Integer(0, 1, name='subsample_freq'),
            Real(0.6, 1.0, 'uniform', name='colsample_bytree'),
            Real(1e-5, 10.0, 'log-uniform', name='reg_alpha'),
            Real(1e-5, 10.0, 'log-uniform', name='reg_lambda'),
            Integer(1, 100, name='random_state'),
            Integer(-1, 1, name='n_jobs'),
        ]
    }
]


if __name__ == '__main__':
    load_dotenv(override=True)

    # explicitly disable multithreaded operation - forcing PyTorch to use a single thread for CPU operations
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    os.makedirs(PRODUCTION_MODEL_FOLDER_PATH, exist_ok=True)

    # create train, val, test datasets
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = data_handling.main_script()

    # processor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    s3_upload(PREPROCESSOR_PATH)

    ## models
    # torch dfn
    best_dfn_full_trained, checkpoint = t.main_script(X_train, X_val, y_train, y_val)
    torch.save(checkpoint, DFN_FILE_PATH)
    s3_upload(file_path=DFN_FILE_PATH)

    # svr
    best_svr_trained, best_hparams_svr = sk.main_script(X_train, X_val, y_train, y_val, **sklearn_models[1])
    if best_svr_trained is not None:
        with open(SVR_FILE_PATH, 'wb') as f:
            pickle.dump({ 'best_model': best_svr_trained, 'best_hparams': best_hparams_svr }, f)
        s3_upload(file_path=SVR_FILE_PATH)


    # elastic net
    best_en_trained, best_hparams_en = sk.main_script(X_train, X_val, y_train, y_val, **sklearn_models[0])
    if best_en_trained is not None:
        with open(EN_FILE_PATH, 'wb') as f:
            pickle.dump({ 'best_model': best_en_trained, 'best_hparams': best_hparams_en }, f)
        s3_upload(file_path=EN_FILE_PATH)

    # light gbm
    # X_train, X_val, X_test, y_train, y_val, y_test, _ = data_handling.main_script(is_scale=False)
    best_gbm_trained, best_hparams_gbm = sk.main_script(X_train, X_val, y_train, y_val, **sklearn_models[2])

    if best_gbm_trained is not None:
        with open(GBM_FILE_PATH, 'wb') as f:
            pickle.dump({'best_model': best_gbm_trained, 'best_hparams': best_hparams_gbm }, f)

        s3_upload(file_path=GBM_FILE_PATH)
