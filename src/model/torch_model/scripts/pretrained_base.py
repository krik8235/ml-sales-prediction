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
