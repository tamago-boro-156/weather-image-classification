# AI学習
import json
from src.preprocessing import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomBrightness, RandomContrast, Input, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

data_augmentation = Sequential([
    RandomBrightness(0.1, value_range = (0, 1)),
    RandomContrast(0.1)
])

model = Sequential()

model.add(Input(shape = (224, 224, 3)))
model.add(data_augmentation)

model.add(Conv2D(
    filters = 32,
    kernel_size = (3, 3),
    activation = "relu"
))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(
    filters = 64,
    kernel_size = (3, 3),
    activation = "relu"
))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation = "relu"))
model.add(Dense(3, activation = "softmax"))

#model.summary()

model.compile(
    optimizer = "Adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

X_train = X_train.astype("float32")
X_val = X_val.astype("float32")
X_test = X_test.astype("float32")

history = model.fit(
    X_train,
    y_train,
    validation_data = (X_val, y_val),
    epochs = 10,
    batch_size = 32
)

with open("models/weather_cnn_history.json", "w") as f:
    json.dump(history.history, f)

model.save("models/weather_cnn_augmented.keras")