from tensorflow.keras import layers, models, optimizers


## todo remove
def create_cnn1(input_shape, num_classes):
    norm_layer = layers.Normalization()
    # # Fit the state of the layer to the spectrograms
    # # with `Normalization.adapt`.

    print("adapting specs")
    # norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda item, label: item))
    print("creating model")
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # Downsample the input.
            layers.Resizing(32, 32),
            # Normalize.
            norm_layer,
            layers.Conv2D(32, 3, activation="relu"),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes),
        ]
    )
    optimizers.Adam(learning_rate=0.01)

    return model


def create_speech_cnn(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Conv2D(32, 3, activation="relu", input_shape=input_shape, padding="same"),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(padding="same"),
            layers.Dropout(0.25),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(padding="same"),
            layers.Dropout(0.25),
            layers.GlobalAveragePooling2D(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model
