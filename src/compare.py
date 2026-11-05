import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("models/results.csv")
print(df)

df.plot(
    x = "model",
    y = ["accuracy", "precision", "recall", "f1-score"],
    kind = "bar"
)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Model Comparison")
plt.savefig("results/model_comparison.png")
plt.show()