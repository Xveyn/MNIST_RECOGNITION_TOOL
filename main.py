#main.py version 1.0
import sys
import os
import tkinter as tk
from tkinter import *
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Error: TensorFlow could not be imported.")
    print("Make sure TensorFlow is installed, e.g., with 'pip install tensorflow'")


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class CNN:
    def __init__(self):
        self.model = self._build_model() if TF_AVAILABLE else None

    def _build_model(self):
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Binarisieren der Daten mit Schwellenwert 128
        x_train = (x_train > 128).astype(np.float32)
        x_test = (x_test > 128).astype(np.float32)

        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # Data Augmentation konfigurieren
        datagen = ImageDataGenerator(
            rotation_range=10,  # Zufällige Rotation zwischen 0 und 10 Grad
            width_shift_range=0.1,  # Zufällige horizontale Verschiebung
            height_shift_range=0.1,  # Zufällige vertikale Verschiebung
            shear_range=0.1,  # Zufälliges Shearing
            zoom_range=0.1,  # Zufälliger Zoom
        )

        datagen.fit(x_train)

        self.model.fit(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_test, y_test),
                       steps_per_epoch=len(x_train) // 32, epochs=15)
        self.model.save('mnist_cnn.h5')

    def load_model(self, path):
        path = resource_path(path)
        self.model = tf.keras.models.load_model(path)

    def predict(self, image):
        image = image.reshape(1, 28, 28, 1).astype(np.float32)
        prediction = self.model.predict(image)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        return digit, confidence


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")

        # Fenster nicht skalierbar machen
        self.root.resizable(False, False)

        self.canvas_size = 280  # 28x28 grid
        self.pixel_size = 10  # Each cell is 10x10 pixels
        self.drawing = False

        # Rahmen um den Zeichenbereich hinzufügen
        self.frame = Frame(self.root, bd=2, relief=SUNKEN)
        self.frame.grid(row=0, column=0, pady=2, columnspan=2)

        self.canvas = Canvas(self.frame, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0)

        self.button_predict = Button(self.root, text="Predict", command=self.predict)
        self.button_predict.grid(row=1, column=0, pady=2, padx=2)
        self.button_clear = Button(self.root, text="Clear", command=self.clear)
        self.button_clear.grid(row=1, column=1, pady=2, padx=2)

        self.label_result = Label(self.root, text="Draw a digit and click Predict")
        self.label_result.grid(row=2, column=0, pady=2, columnspan=2)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.image = np.zeros((28, 28), dtype=np.float32)

        self.cnn = CNN()
        if self.cnn:
            try:
                self.cnn.load_model('mnist_cnn.h5')
            except:
                self.cnn.train()
                self.cnn.load_model('mnist_cnn.h5')

        # Pixelraster erstellen
        self.create_grid()

    def create_grid(self):
        for i in range(0, self.canvas_size, self.pixel_size):
            self.canvas.create_line([(i, 0), (i, self.canvas_size)], tag='grid_line', fill='gray')
            self.canvas.create_line([(0, i), (self.canvas_size, i)], tag='grid_line', fill='gray')

    def start_drawing(self, event):
        self.drawing = True

    def stop_drawing(self, event):
        self.drawing = False

    def draw(self, event):
        if not self.drawing:
            return

        x, y = event.x, event.y
        if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
            x_pixel, y_pixel = x // self.pixel_size, y // self.pixel_size
            self.canvas.create_rectangle(x_pixel * self.pixel_size, y_pixel * self.pixel_size,
                                         (x_pixel + 1) * self.pixel_size, (y_pixel + 1) * self.pixel_size,
                                         fill='black')
            self.image[y_pixel, x_pixel] = 1.0

    def clear(self):
        self.canvas.delete("all")
        self.create_grid()  # Pixelraster nach dem Löschen wiederherstellen
        self.image = np.zeros((28, 28), dtype=np.float32)
        self.label_result.config(text="Draw a digit and click Predict")

    def predict(self):
        digit, confidence = self.cnn.predict(self.image)
        self.label_result.config(text=f"Prediction: {digit} with confidence {confidence:.2f}")


def main():
    try:
        tf.keras.models.load_model('mnist_cnn.h5')
    except:
        cnn_model = CNN()
        cnn_model.train()

    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
