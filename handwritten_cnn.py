
"""
Handwritten Digit Recognition using CNN and MNIST.
Downloads MNIST using Keras, trains a compact CNN, evaluates it, and saves model/plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Fixes the Tkinter error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    tf.random.set_seed(42)

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize images
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    model = build_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # Save trained model
    model.save("mnist_cnn.keras")

    # Save training accuracy graph
    plt.figure()
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST CNN Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mnist_training_accuracy.png", dpi=150)
    plt.close()

    print("Saved model: mnist_cnn.keras")
    print("Saved plot: mnist_training_accuracy.png")

    # Generate sample predictions
    probabilities = model.predict(x_test[:10], verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        ax.set_title(f"Pred: {predictions[i]} | True: {y_test[i]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("mnist_sample_predictions.png", dpi=150)
    plt.close()

    print("Saved sample predictions: mnist_sample_predictions.png")
    print("\nTask 3 completed successfully!")


if __name__ == "__main__":
    main()
