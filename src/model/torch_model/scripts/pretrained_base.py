import torch.nn as nn

class DFN(nn.Module):
    def __init__(self, input_dim = None, **kwargs):
        super(DFN, self).__init__()

        layers = []
        num_layers = kwargs.get('num_layers', 1)
        hidden_units_per_layer = kwargs.get('hidden_units_per_layer', [64] * num_layers)
        batch_norm = kwargs.get('batch_norm', False)
        dropout_rates = kwargs.get('dropout_rates', [0.1 for _ in range(0, num_layers)])

        # creates hidden layers
        current_input_dim = input_dim if input_dim is not None else 32

        for i in range(num_layers):
            output_dim = hidden_units_per_layer[i]

            layers.append(nn.Linear(current_input_dim, output_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(output_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rates[i]))

            # align with the ouput dimensions
            current_input_dim = output_dim

        layers.append(nn.Linear(current_input_dim, 1))
        self.model_layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.model_layers(x)


# class DFN_Optuna(nn.Module):
#     """
#     A class to reconstruct DFN from the optimal hyperparameters from Optuna.

#     Example of state_dict keys:
#         ['model_layers.0.weight', 'model_layers.0.bias', 'model_layers.1.weight', 'model_layers.1.bias', 'model_layers.1.running_mean', 'model_layers.1.running_var', 'model_layers.1.num_batches_tracked', 'model_layers.4.weight', 'model_layers.4.bias', 'model_layers.5.weight', 'model_layers.5.bias', 'model_layers.5.running_mean', 'model_layers.5.running_var', 'model_layers.5.num_batches_tracked', 'model_layers.8.weight', 'model_layers.8.bias', 'model_layers.9.weight', 'model_layers.9.bias', 'model_layers.9.running_mean', 'model_layers.9.running_var', 'model_layers.9.num_batches_tracked', 'model_layers.12.weight', 'model_layers.12.bias']
#     """

#     def __init__(self, input_dim, state_dict):
#         super(DFN_Optuna, self).__init__()

#         layers = []
#         model_keys = [key for key in state_dict.keys() if key.startswith("model_layers.")]
#         sorted_keys = sorted(model_keys, key=lambda x: int(x.split('.')[1]))

#         layer_groups = {}
#         for key in sorted_keys:
#             index = int(key.split('.')[1])
#             if index not in layer_groups:
#                 layer_groups[index] = []
#             layer_groups[index].append(key)

#         for k, keys in layer_groups.items():
#             k = int(k)

#             if k % 4 == 0:
#                 weight_key = [k for k in keys if 'weight' in k][0]
#                 input_dim = input_dim if k == 0 else state_dict[weight_key].shape[1]
#                 output_dim = state_dict[weight_key].shape[0]
#                 layers.append(nn.Linear(input_dim, output_dim))

#             else:
#                 key_running_mean = [k for k in keys if 'running_mean' in k][0]
#                 if key_running_mean:
#                     num_features = state_dict[key_running_mean].shape[0]
#                     layers.append(nn.BatchNorm1d(num_features))

#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(p=0.1))

#         self.model_layers = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model_layers(x)
