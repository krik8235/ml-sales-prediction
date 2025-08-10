import tensorflow as tf # type: ignore
from tensorflow import keras  # type: ignore
from keras import models, layers, optimizers, metrics, losses, backend, callbacks # type: ignore


def _build_cnn_model(hparams: dict, input_shape: tuple) -> models.Model:
    """
    Internal function to build a model for each evaluation.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(hparams['filters_0'], (3, 3), activation='relu', padding='same')(inputs)
    if hparams.get('batch_norm_0', False): x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)

    for i in range(1, hparams.get('num_conv_layers', 1)):
        if f'filters_{i}' in hparams:
            x = layers.Conv2D(hparams.get(f'filters_{i}', 16), (3, 3), activation='relu', padding='same')(x)
            if hparams.get(f'batch_norm_{i}', False): x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(hparams['dense_units'], activation='relu')(x)
    if hparams['dropout'] > 0: x = layers.Dropout(hparams['dropout'])(x)

    outputs = layers.Dense(1)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=hparams['learning_rate']), # type: ignore
        loss=losses.MeanSquaredError(),
        metrics=[metrics.MeanAbsoluteError(name='mae')]
    )
    return model


def evaluate_model(X_train, y_train, X_val, y_val, hparams: dict, metric: str = 'val_mae') -> tuple[float, models.Model]:
    """
    Trains and evaluates Keras models based on validation dataset and given metrics (default: MAE).
    """

    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
    model = _build_cnn_model(hparams=hparams, input_shape=X_train_reshaped.shape[1:])
    early_stopping = callbacks.EarlyStopping(
        monitor=metric,
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    model.fit(
        X_train_reshaped, y_train, # type: ignore
        epochs=hparams.get('epochs', 1000),
        validation_data=(X_val, y_val), # type: ignore
        batch_size=16,
        callbacks=[early_stopping],
        verbose=0 # type: ignore
    )
    _, mae = model.evaluate(X_val, y_val, verbose=0) # type: ignore

    # clear keras backend
    backend.clear_session()

    return mae, model
