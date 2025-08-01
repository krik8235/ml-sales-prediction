import os
import warnings
import numpy as np
import lightgbm as lgb # type: ignore
from sklearn.linear_model import ElasticNet
from skopt.space import Real, Integer, Categorical # type: ignore
from dotenv import load_dotenv # type: ignore

import src.data_handling as data_handling
import src.model.torch_model as t
import src.model.sklearn_model as sk


warnings.filterwarnings("ignore", category=RuntimeWarning)
load_dotenv(override=True)

os.environ["OMP_NUM_THREADS"] = "1" # explicitly disable multithreaded operation - forcing PyTorch to use a single thread for CPU operations
os.environ["MKL_NUM_THREADS"] = "1"


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
    # create train, val, test datasets
    X_train, X_val, X_test, y_train, y_val, y_test = data_handling.main_script()

    # torch dfn
    best_dfn = t.main_script(X_train, X_val, X_test, y_train, y_val, y_test)

    # elastic net
    best_en = sk.main_script(X_train, X_val, y_train, y_val, **sklearn_models[0])

    # light gbm
    X_train, X_val, X_test, y_train, y_val, y_test = data_handling.main_script(is_scale=False)
    best_gbm = sk.main_script(X_train, X_val, y_train, y_val, **sklearn_models[1])
