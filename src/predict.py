# 予測
import numpy as np
import sys
import os
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

classes = ["sunny", "cloudy", "rainy"]

def predict_weather(img_path, model):
    try:
        img = Image.open(img_path).convert("RGB")
    except UnidentifiedImageError:
        print("画像ファイルとして読み込めません．")
        sys.exit(1)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255

    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    max_probability = np.max(prediction) * 100
    
    return predicted_class, prediction

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("画像ファイルを指定してください．")
        sys.exit(1)

    img_paths = sys.argv[1:]

    model = load_model("models/weather_cnn_augmented.keras")
    for img_path in img_paths:
        if not os.path.isfile(img_path):
            print("指定された画像が見つかりません．")
            sys.exit(1)

        result, probabilities = predict_weather(img_path, model)

        if result is None:
            continue
        
        print(f"\n画像: {img_path}")
        print(f"予測された天気: {result}")

        for i, cls in enumerate(classes):
            probability = probabilities[0][i] * 100
            print(f"{cls}: {probability:.2f}%")