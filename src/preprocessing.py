import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_data():
    data_dir = "./data"

    classes = ["sunny", "cloudy", "rainy"]
    images = []
    labels = []

    for cls in classes:
        folder = os.path.join(data_dir, cls)
        files = os.listdir(folder)

        for file in files:
            img_path = os.path.join(folder, file)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img) / 255

            images.append(img_array)
            labels.append(classes.index(cls))

    images = np.array(images)
    labels = np.array(labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        images,
        labels,
        test_size = 0.3,
        random_state = 42,
        stratify = labels
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size = 0.5,
        random_state = 42,
        stratify = y_temp
    )

    """
    print(f"学習用: {X_train.shape[0]}枚")
    print(f"検証用: {X_val.shape[0]}枚")
    print(f"テスト用: {X_test.shape[0]}枚")

    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    print(np.unique(y_test, return_counts=True))
    """

    return X_train, X_val, X_test, y_train, y_val, y_test