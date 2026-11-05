# 確認
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from src.preprocessing import load_data

model = load_model("models/weather_cnn_baseline.keras")

with open("models/weather_cnn_history.json", "r") as f:
    history = json.load(f)

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

X_test = X_test.astype("float32")

classes = ["sunny", "cloudy", "rainy"]

# Accuracyのグラフ
epochs = range(1, len(history["accuracy"]) + 1)

plt.plot(epochs, history["accuracy"], label="Train")
plt.plot(epochs, history["val_accuracy"], label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Lossのグラフ
epochs = range(1, len(history["loss"]) + 1)

plt.plot(epochs, history["loss"], label="Train")
plt.plot(epochs, history["val_loss"], label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
#print(f"テストデータでの損失: {loss}")
#print(f"テストデータでの精度: {accuracy}")

predictions = model.predict(X_test)
#print(predictions.shape)
#print(predictions[0])
y_pred = np.argmax(predictions, axis=1)
#print(y_pred)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred, target_names=classes))

report = classification_report(
    y_test,
    y_pred,
    target_names = classes,
    output_dict = True
)
result = {
    "model": "baseline",
    "accuracy": accuracy,
    "precision": report["macro avg"]["precision"],
    "recall": report["macro avg"]["recall"],
    "f1-score": report["macro avg"]["f1-score"]
}
csv_path = "models/results.csv"
file_exists = os.path.exists(csv_path)
with open(csv_path, "a", newline = "") as f:
    writer = csv.DictWriter(f, fieldnames = result.keys())
    if not file_exists or os.path.getsize(csv_path) == 0:
        writer.writeheader()
    writer.writerow(result)
print(f"評価結果保存済: {csv_path}")

print("Accuracy:", np.mean(y_test == y_pred))

wrong_indices = np.where((y_test ==2) & (y_pred ==1))[0]
for i in wrong_indices[:5]:
    plt.imshow(X_test[i])
    plt.title(f"True: {classes[y_test[i]]}, Pred: {classes[y_pred[i]]}")
    plt.axis('off')
    plt.show()