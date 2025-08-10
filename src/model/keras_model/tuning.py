
from itertools import product
from keras import models # type: ignore

from src.model.keras_model.evaluate_model import evaluate_model
from src._utils import main_logger


# search_space = {
#         'num_conv_layers': [1],
#         'filters_0': [16, 32],
#         'batch_norm_0': [False],
#         'dense_units': [16, 32, 64],
#         'dropout': [0.0, 0.1, 0.2],
#         'learning_rate': [0.001, 0.0001],
#         'epochs': [200]
#     }


def grid_search(X_train, y_train, X_val, y_val, search_space: dict) -> tuple[float, dict | None, models.Model]:
    """
    Runs grid search in the search space (basically test all the combinations of hyperparameter options in the search space)
    """

    all_combinations_list = []
    best_mae =  float('inf')
    best_hparams = None
    best_model = None

    for num_conv in search_space['num_conv_layers']:
        # creating base combination of the hyperparameters
        dynamic_filters_bn_combinations = list(
            product(
                *(search_space[f'filters_{i}'] for i in range(num_conv)),
                *(search_space[f'batch_norm_{i}'] for i in range(num_conv))
            )
        )
        base_combos = product(
            search_space['learning_rate'],
            search_space['epochs'],
            search_space['dense_units'],
            search_space['dropout']
        )

        for base_combo in base_combos:
            for dynamic_combo in dynamic_filters_bn_combinations:
                hparam_set = {
                    'learning_rate': base_combo[0],
                    'epochs': base_combo[1],
                    'num_conv_layers': num_conv,
                    'dense_units': base_combo[2],
                    'dropout': base_combo[3]
                }
                filter_values = dynamic_combo[:num_conv]
                for i in range(num_conv): hparam_set[f'filters_{i}'] = filter_values[i]
                all_combinations_list.append(hparam_set)

    for i, hparams in enumerate(all_combinations_list):
        current_mae, model = evaluate_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, hparams=hparams)

        if current_mae < best_mae:
            best_mae = current_mae
            best_hparams = hparams
            best_model = model

    main_logger.info(f'Best hparams:\n{best_hparams}\nBest acuracy {best_mae:.4f}')

    return best_mae, best_hparams, best_model
