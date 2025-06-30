# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score

# 2. Load Dataset (This is sample data; replace with your own CSV later)
data = {
    'attendance': [85, 60, 78, 90, 55],
    'gpa': [3.5, 2.1, 2.8, 3.9, 1.8],
    'gender': ['F', 'M', 'F', 'M', 'F'],
    'lms_activity': [25, 10, 15, 30, 5],
    'dropout': [0, 1, 0, 0, 1]  # 1 = dropout, 0 = not dropout
}
df = pd.DataFrame(data)

# 3. Preprocessing
# Encode the 'gender' column (F = 0, M = 1)
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

# Normalize numerical features to range (0, 1)
scaler = MinMaxScaler()
df[['attendance', 'gpa', 'lms_activity']] = scaler.fit_transform(df[['attendance', 'gpa', 'lms_activity']])

# 4. Split Features and Target
X = df.drop('dropout', axis=1)
y = df['dropout']

# Split the data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 6. Make Predictions
y_pred = model.predict(X_test)

# 7. Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# 8. Print Evaluation Results
print("Model Accuracy:", round(accuracy, 2))
print("Model Recall:", round(recall, 2))
