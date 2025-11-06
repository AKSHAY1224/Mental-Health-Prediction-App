# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# 1️⃣ Load dataset
df = pd.read_csv("depression_anxiety_data.csv")

# 2️⃣ Drop ID column if present
if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# 3️⃣ Handle missing values
df.fillna({
    "depressiveness": False,
    "depression_treatment": False,
    "anxiousness": False,
    "anxiety_treatment": False,
    "sleepiness": False
}, inplace=True)

# 4️⃣ Encode categorical columns (convert text to numbers)
df_encoded = pd.get_dummies(df, drop_first=True)

# 5️⃣ Separate features & target
X = df_encoded.drop(["phq_score", "gad_score"], axis=1)
y_phq = df_encoded["phq_score"]
y_gad = df_encoded["gad_score"]

# 6️⃣ Split data
X_train, X_test, y_train_phq, y_test_phq = train_test_split(X, y_phq, test_size=0.2, random_state=42)
_, _, y_train_gad, y_test_gad = train_test_split(X, y_gad, test_size=0.2, random_state=42)

# 7️⃣ Train models
phq_model = RandomForestRegressor(n_estimators=100, random_state=42)
phq_model.fit(X_train, y_train_phq)

gad_model = RandomForestRegressor(n_estimators=100, random_state=42)
gad_model.fit(X_train, y_train_gad)

# 8️⃣ Evaluate
phq_pred = phq_model.predict(X_test)
gad_pred = gad_model.predict(X_test)

print("PHQ Model:")
print("MAE:", mean_absolute_error(y_test_phq, phq_pred))
print("R2:", r2_score(y_test_phq, phq_pred))

print("\nGAD Model:")
print("MAE:", mean_absolute_error(y_test_gad, gad_pred))
print("R2:", r2_score(y_test_gad, gad_pred))

# 9️⃣ Save models
pickle.dump(phq_model, open("phq_model.pkl", "wb"))
pickle.dump(gad_model, open("gad_model.pkl", "wb"))

print("\n✅ Models saved successfully as phq_model.pkl and gad_model.pkl!")
