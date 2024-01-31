from tensorflow import keras


def create_mnist_cnn(num_classes=10, optimizer="adam"):
    model = keras.models.Sequential(
        [
            keras.layers.InputLayer((28, 28)),
            keras.layers.Reshape((28, 28, 1)),
            keras.layers.Conv2D(
                32,
                kernel_size=(5, 5),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(
                64,
                kernel_size=(5, 5),
                activation="relu",
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
