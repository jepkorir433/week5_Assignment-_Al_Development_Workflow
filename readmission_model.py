# Step 1: Import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import joblib

# Step 2: Create sample data
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# Step 3: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Export the model
joblib.dump(model, 'readmission_model.pkl')

print("✅ Model trained and saved as 'readmission_model.pkl'")
