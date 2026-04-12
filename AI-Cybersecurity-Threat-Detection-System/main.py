import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Column names
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment",
    "urgent","hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted",
    "num_root","num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

data = pd.read_csv("data/dataset.csv", header=None)
data.columns = columns
data = data.drop("difficulty", axis=1)
print(data.head())

# Check labels BEFORE conversion
print("\nOriginal Labels:")
print(data['label'].value_counts().head())

# Convert labels properly
data['label'] = data['label'].apply(lambda x: 0 if x == 'normal' else 1)

print("\nConverted Labels:")
print(data['label'].value_counts())

# Encode categorical columns
le = LabelEncoder()
for col in ['protocol_type', 'service', 'flag']:
    data[col] = le.fit_transform(data[col])

print("\nFinal Data Shape:", data.shape)
print(data.head())
from sklearn.model_selection import train_test_split

# Separate features and target
X = data.drop("label", axis=1)
y = data["label"]

print("\nFeature shape:", X.shape)
print("Label shape:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

print("\nModel Training Completed ✅")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
import joblib

joblib.dump(model, "models/model.pkl")
print("Model saved successfully ✅")
from sklearn.metrics import classification_report

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - Threat Detection")
plt.savefig("images/confusion_matrix.png")
plt.show()
import pandas as pd

importance = model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:\n", feat_df.head(10))
print("\n--- THREAT DETECTION SYSTEM ---")

# Take some test samples
sample_data = X_test.iloc[:10]

# Predict on new data
sample_predictions = model.predict(sample_data)

# Show alerts
for i, pred in enumerate(sample_predictions):
    if pred == 1:
        print(f"🚨 ALERT: Threat detected in sample {i}")
    else:
        print(f"✅ Normal traffic in sample {i}")
        import pandas as pd

results = sample_data.copy()
results["Prediction"] = sample_predictions

results.to_csv("outputs/predictions.csv", index=False)

print("\nPredictions saved to outputs/predictions.csv ✅")
import matplotlib.pyplot as plt

# Accuracy bar chart
metrics = ["Accuracy"]
values = [accuracy]

plt.figure()
plt.bar(metrics, values)
plt.title("Model Performance")
plt.ylabel("Score")
plt.savefig("images/accuracy.png")
plt.show()

print("Accuracy graph saved ✅")
plt.figure(figsize=(10,5))
feat_df.head(10).plot(x="Feature", y="Importance", kind="bar")

plt.title("Top 10 Important Features")
plt.savefig("images/feature_importance.png")
plt.show()

print("Feature importance graph saved ✅")
# Select important columns only
results = sample_data.copy()
results["Prediction"] = sample_predictions

# Keep only few columns for clean output
results_small = results[[
    "duration", "protocol_type", "service", "src_bytes", "dst_bytes", "Prediction"
]]

results_small.to_csv("outputs/predictions.csv", index=False)

print("Clean predictions saved ✅")