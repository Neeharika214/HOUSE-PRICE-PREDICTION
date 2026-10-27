
# ================================================================
# HOUSE PRICE PREDICTION - COMPLETE GOOGLE COLAB CODE
# Dataset: Ames Housing
# Model: Random Forest Regression
# Price output: Indian Rupees (Lakhs)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# ================================================================
# 1. LOAD DATASET
# ================================================================

url = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"

df = pd.read_csv(url, sep="\t")
df.columns = df.columns.str.strip()

print("=" * 65)
print("AMES HOUSING DATASET")
print("=" * 65)
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
display(df.head())

# ================================================================
# 2. SELECT FEATURES
# ================================================================

features = [
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Garage Area",
    "Total Bsmt SF",
    "Year Built",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "TotRms AbvGrd",
    "1st Flr SF",
    "2nd Flr SF",
    "Lot Area"
]

target = "SalePrice"

# Check that all columns exist
required_columns = features + [target]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df = df[required_columns].copy()

# Convert columns to numeric values
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing target prices
df = df.dropna(subset=[target])

# ================================================================
# 3. REMOVE EXTREME PRICE OUTLIERS
# ================================================================

Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1

lower_limit = max(0, Q1 - 1.5 * IQR)
upper_limit = Q3 + 1.5 * IQR

df = df[df[target].between(lower_limit, upper_limit)].copy()

print("\nRows after price outlier removal:", len(df))

# ================================================================
# 4. PREPARE FEATURES AND TARGET
# ================================================================

X = df[features].copy()

# Convert US house prices into INR lakhs.
# Assumed exchange rate: 1 USD = ₹95
USD_TO_INR = 95

y = df[target] * USD_TO_INR / 100000

# Split data before filling missing feature values
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Fill missing feature values using training medians only
training_medians = X_train.median()

X_train = X_train.fillna(training_medians)
X_test = X_test.fillna(training_medians)

print("\nDATA SPLIT DETAILS")
print("-" * 50)
print("Total houses:", len(X))
print("Training houses (80%):", len(X_train))
print("Testing houses (20%):", len(X_test))

# ================================================================
# 5. TRAIN RANDOM FOREST MODEL
# ================================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions for both datasets
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# ================================================================
# 6. MODEL EVALUATION
# ================================================================

r2_train = r2_score(y_train, train_pred)
r2_test = r2_score(y_test, test_pred)

mae_train = mean_absolute_error(y_train, train_pred)
mae_test = mean_absolute_error(y_test, test_pred)

rmse_train = np.sqrt(mean_squared_error(y_train, train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, test_pred))

mape_train = mean_absolute_percentage_error(
    y_train, train_pred
) * 100

mape_test = mean_absolute_percentage_error(
    y_test, test_pred
) * 100

# This is an MAPE-based score, not standard accuracy
score_train = 100 - mape_train
score_test = 100 - mape_test

print("\n" + "=" * 75)
print("MODEL PERFORMANCE")
print("=" * 75)

print(f"{'Metric':<25}{'Training':>20}{'Testing':>20}")
print("-" * 75)

print(f"{'R² Score':<25}{r2_train:>20.4f}{r2_test:>20.4f}")
print(f"{'MAPE (%)':<25}{mape_train:>19.2f}%{mape_test:>19.2f}%")
print(f"{'100 - MAPE (%)':<25}{score_train:>19.2f}%{score_test:>19.2f}%")
print(f"{'MAE (₹ Lakhs)':<25}{mae_train:>20.2f}{mae_test:>20.2f}")
print(f"{'RMSE (₹ Lakhs)':<25}{rmse_train:>20.2f}{rmse_test:>20.2f}")

print("=" * 75)
print("R² measures explained variation; higher is generally better.")
print("MAE, RMSE and MAPE: lower is generally better.")
print("100 - MAPE is only an approximate error-based score.")

# ================================================================
# 7. GRAPH: ACTUAL VS PREDICTED PRICE
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    test_pred,
    alpha=0.55,
    s=25,
    label="Test predictions"
)

min_val = min(y_test.min(), test_pred.min())
max_val = max(y_test.max(), test_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="Perfect prediction"
)

plt.xlabel("Actual Price (₹ Lakhs)")
plt.ylabel("Predicted Price (₹ Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ================================================================
# 8. GRAPH: TRAINING VS TESTING R² SCORE
# ================================================================

plt.figure(figsize=(7, 5))

values = [r2_train, r2_test]
bars = plt.bar(
    ["Training R²", "Testing R²"],
    values
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.ylabel("R² Score")
plt.title("Training vs Testing R²")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ================================================================
# 9. GRAPH: MAPE-BASED SCORE
# ================================================================

plt.figure(figsize=(7, 5))

values = [score_train, score_test]
bars = plt.bar(
    ["Training Score", "Testing Score"],
    values
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.ylabel("100 - MAPE (%)")
plt.title("MAPE-Based Prediction Score")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ================================================================
# 10. GRAPH: MEAN ABSOLUTE ERROR
# ================================================================

plt.figure(figsize=(7, 5))

values = [mae_train, mae_test]
bars = plt.bar(
    ["Training MAE", "Testing MAE"],
    values
)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"₹{value:.2f} L",
        ha="center",
        va="bottom"
    )

plt.ylabel("Mean Absolute Error (₹ Lakhs)")
plt.title("Average House Price Prediction Error")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ================================================================
# 11. GRAPH: FEATURE IMPORTANCE
# ================================================================

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(9, 6))
importance.plot(kind="barh")

plt.xlabel("Importance")
plt.title("Feature Importance in House Price Prediction")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# ================================================================
# 12. ACTUAL VS PREDICTED SAMPLE TABLE
# ================================================================

comparison = pd.DataFrame({
    "Actual Price (₹ Lakhs)": y_test.to_numpy(),
    "Predicted Price (₹ Lakhs)": test_pred
})

comparison["Absolute Error (₹ Lakhs)"] = np.abs(
    comparison["Actual Price (₹ Lakhs)"]
    - comparison["Predicted Price (₹ Lakhs)"]
)

print("\nSAMPLE TEST PREDICTIONS")
display(comparison.head(15).round(2))

# ================================================================
# 13. PREDICT PRICE FOR A NEW HOUSE
# ================================================================

print("\n" + "=" * 60)
print("ENTER DETAILS FOR YOUR HOUSE")
print("=" * 60)

def safe_input(prompt, default, cast_type, min_value=None, max_value=None):
    while True:
        raw = input(
            f"{prompt} [Default: {default}]: "
        ).strip()

        try:
            value = default if raw == "" else cast_type(raw)

            if not np.isfinite(value):
                print("Please enter a valid finite number.")
                continue

            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Value must not exceed {max_value}.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please try again.")

user_values = {}

user_values["Overall Qual"] = safe_input(
    "Overall Quality (1-10)",
    int(round(training_medians["Overall Qual"])),
    int, 1, 10
)

user_values["Gr Liv Area"] = safe_input(
    "Living Area (sq ft)",
    int(round(training_medians["Gr Liv Area"])),
    float, 1
)

user_values["Garage Cars"] = safe_input(
    "Garage Capacity (cars)",
    int(round(training_medians["Garage Cars"])),
    int, 0
)

user_values["Garage Area"] = safe_input(
    "Garage Area (sq ft)",
    int(round(training_medians["Garage Area"])),
    float, 0
)

user_values["Total Bsmt SF"] = safe_input(
    "Basement Area (sq ft)",
    int(round(training_medians["Total Bsmt SF"])),
    float, 0
)

user_values["Year Built"] = safe_input(
    "Year Built",
    int(round(training_medians["Year Built"])),
    int, 1800, 2026
)

user_values["Full Bath"] = safe_input(
    "Full Bathrooms",
    int(round(training_medians["Full Bath"])),
    int, 0
)

user_values["Half Bath"] = safe_input(
    "Half Bathrooms",
    int(round(training_medians["Half Bath"])),
    int, 0
)

user_values["Bedroom AbvGr"] = safe_input(
    "Bedrooms Above Ground",
    int(round(training_medians["Bedroom AbvGr"])),
    int, 0
)

user_values["TotRms AbvGrd"] = safe_input(
    "Total Rooms Above Ground",
    int(round(training_medians["TotRms AbvGrd"])),
    int, 1
)

user_values["1st Flr SF"] = safe_input(
    "First Floor Area (sq ft)",
    int(round(training_medians["1st Flr SF"])),
    float, 0
)

user_values["2nd Flr SF"] = safe_input(
    "Second Floor Area (sq ft)",
    int(round(training_medians["2nd Flr SF"])),
    float, 0
)

user_values["Lot Area"] = safe_input(
    "Lot Area (sq ft)",
    int(round(training_medians["Lot Area"])),
    float, 1
)

user_house = pd.DataFrame(
    [[user_values[col] for col in features]],
    columns=features
)

# Predict in INR lakhs
predicted_price = model.predict(user_house)[0]

print("\n" + "=" * 60)
print("PREDICTED HOUSE PRICE")
print("=" * 60)
print(f"Estimated Price: ₹{max(0, predicted_price):.2f} Lakhs")
print(f"Estimated Price: ₹{max(0, predicted_price) * 100000:,.0f}")
print("=" * 60)

print("\nNote: This model is trained on US housing data.")
print("The INR value uses an assumed exchange rate of ₹95/USD.")
print("It is not a direct estimate of an Indian property's market value.")
