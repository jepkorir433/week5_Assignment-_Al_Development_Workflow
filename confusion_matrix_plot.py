import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Hypothetical student dropout data
y_true = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # Actual outcomes
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 1]  # Model predictions

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
            xticklabels=["Predicted: Not Dropped Out", "Predicted: Dropped Out"],
            yticklabels=["Actual: Not Dropped Out", "Actual: Dropped Out"])
plt.title("Confusion Matrix - Student Dropout Prediction")
plt.xlabel("Predicted Outcome")
plt.ylabel("Actual Outcome")
plt.tight_layout()

# Save and show the graph
plt.savefig("dropout_confusion_matrix.png", dpi=300)  # Optional: saves image
plt.show()
