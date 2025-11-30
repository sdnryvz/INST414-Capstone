# To push to git use these three steps:
# git add .
# git commit -m "Describe what you changed"
# git push origin main 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/Users/nuryavuz/Desktop/inst414-capstone/spotify_songs1.csv")
df.describe()
print(df.shape) #(32833, 23)

# key variables (the numeric metrics: playlist_genre & subgenre, danceability, energy, key, loudness, mode, speechiness, acousticness
# instrumentalness, liveness, valence, tempo, duration_ms)

# data cleaning for sprint 2 plotting
categorical_cols = ['playlist_genre', 'playlist_subgenre']
numeric_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'duration_ms']
target_col = 'track popularity'

# summary statistics
summary_numeric = df[numeric_cols].describe().T
print(summary_numeric)

summary_by_genre = df.groupby('playlist_genre')[numeric_cols].agg(['mean', 'std', 'median', 'count'])
print(summary_by_genre)

# distribution of playlist genre
plt.figure(figsize=(10, 5))
genre_counts = df['playlist_genre'].value_counts()
sns.barplot(x=genre_counts.index, y=genre_counts.values)
plt.xticks(rotation=45, ha='right')
plt.title('Playlist Genre Distribution')
plt.xlabel('Playlist Genre')
plt.ylabel('Count of Tracks')
plt.tight_layout()
plt.show()

# playlist subgenres
plt.figure(figsize=(10, 5))
top_n = 15  # change if you want more/less
subgenre_counts = df['playlist_subgenre'].value_counts().head(top_n)
sns.barplot(x=subgenre_counts.index, y=subgenre_counts.values)
plt.xticks(rotation=60, ha='right')
plt.title(f'Top {top_n} Playlist Subgenres')
plt.xlabel('Playlist Subgenre')
plt.ylabel('Count of Tracks')
plt.tight_layout()
plt.show()

# distribution of numeric audio features
n_features = len(numeric_cols)
n_cols = 4 
n_rows = (n_features + n_cols - 1) // n_cols 
plt.figure(figsize=(4 * n_cols, 3 * n_rows))

for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(df[col].dropna(), bins=30)
    plt.title(col)
    plt.xlabel('')
    plt.ylabel('Count')

plt.suptitle('Distributions of Audio Features', y=1.02)
plt.tight_layout()
plt.show()

# heatmap of features
heatmap_cols = numeric_cols.copy()
if target_col in df.columns:
    heatmap_cols = heatmap_cols + [target_col]

corr = df[heatmap_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap (Audio Features & Popularity)')
plt.tight_layout()
plt.show()



##### MODELING

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

X = df[numeric_cols + categorical_cols].copy()
y = df['track_popularity']

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# splitting the data into training and validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# baseline model
baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

print("\n--- Baseline Model Metrics ---")
print(f"MAE:  {baseline_mae:.2f}")
print(f"RMSE: {baseline_rmse:.2f}")
print(f"R²:   {baseline_r2:.3f}")

### CANDIDATE MODELS 
import time
results = []

# random forest model
start = time.time()

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=1,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

train_time_rf = time.time() - start

y_pred_rf = rf_model.predict(X_test)

results.append([
    "Random Forest",
    mean_absolute_error(y_test, y_pred_rf),
    np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    r2_score(y_test, y_pred_rf),
    train_time_rf,
    "Medium"
])

# catboost model (candidate 1)
from catboost import CatBoostRegressor
X_cb = df[numeric_cols + categorical_cols].copy()
y_cb = df['track_popularity']

X_train_cb, X_test_cb, y_train_cb, y_test_cb = train_test_split(
    X_cb, y_cb, test_size=0.2, random_state=1
)

start = time.time()

cat_model = CatBoostRegressor(
    depth=8,
    iterations=500,
    learning_rate=0.05,
    loss_function='RMSE',
    verbose=False,
    random_state=1
)

cat_model.fit(
    X_train_cb, 
    y_train_cb,
    cat_features=categorical_cols
)

train_time_cat = time.time() - start

y_pred_cat = cat_model.predict(X_test_cb)

results.append([
    "CatBoost",
    mean_absolute_error(y_test_cb, y_pred_cat),
    np.sqrt(mean_squared_error(y_test_cb, y_pred_cat)),
    r2_score(y_test_cb, y_pred_cat),
    train_time_cat,
    "Low"
])

# linear regression model (candidate 2)
from sklearn.linear_model import LinearRegression

start = time.time()

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

train_time_lin = time.time() - start

y_pred_lin = lin_model.predict(X_test)

results.append([
    "Linear Regression",
    mean_absolute_error(y_test, y_pred_lin),
    np.sqrt(mean_squared_error(y_test, y_pred_lin)),
    r2_score(y_test, y_pred_lin),
    train_time_lin,
    "High"
])

# comparison table 
comparison_df = pd.DataFrame(
    results,
    columns=["Model", "MAE ↓", "RMSE ↓", "R² ↑", "Training Time (s)", "Interpretability"]
)

print("\n=== MODEL COMPARISON ===")
print(comparison_df)


### REGRESSION DIAGNOSTICS

import scipy.stats as stats

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)


mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print("=== Train Performance (Random Forest) ===")
print(f"MAE:  {mae_train:.2f}")
print(f"RMSE: {rmse_train:.2f}")
print(f"R²:   {r2_train:.3f}")

print("\n=== Test Performance (Random Forest) ===")
print(f"MAE:  {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R²:   {r2_test:.3f}")

# residuals 
residuals = y_test - y_test_pred

# residuals vs predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted popularity")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residuals vs Predicted Popularity (Random Forest)")
plt.tight_layout()
plt.show()

# histogram of residuals (prediction error distribution)
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30)
plt.xlabel("Residual (actual - predicted)")
plt.ylabel("Count")
plt.title("Distribution of Prediction Errors (Random Forest)")
plt.tight_layout()
plt.show()

# qq plot of residuals
plt.figure(figsize=(5, 5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q–Q Plot of Residuals (Random Forest)")
plt.tight_layout()
plt.show()

### CROSS VALIDATION AND GENERALIZATION
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

rf_cv_model = RandomForestRegressor(
    n_estimators=300,
    random_state=1,
    n_jobs=-1
)

cv_scores = cross_val_score(
    rf_cv_model,
    X,
    y,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

print("=== 5-fold Cross-Validation (Random Forest, R²) ===")
print("Fold R² scores:", cv_scores)
print("Mean R²:", cv_scores.mean())
print("Std R²:", cv_scores.std())


### FEATURE IMPORTANCE AND INTERPRETABILITY
importances = rf_model.feature_importances_
fi = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values("importance", ascending=False)

print("=== Top 15 Features by Importance ===")
print(fi.head(15))


top_k = 10
plt.figure(figsize=(8, 5))
plt.barh(fi["feature"].head(top_k)[::-1], fi["importance"].head(top_k)[::-1])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances – Random Forest")
plt.tight_layout()
plt.show()

### ERROR ANALYSIS

test_results = df.loc[y_test.index].copy()

test_results["predicted_popularity"] = y_test_pred
test_results["residual"] = y_test - y_test_pred
test_results["abs_error"] = test_results["residual"].abs()

worst_10 = test_results.sort_values("abs_error", ascending=False).head(10)

cols_to_show = [
    c for c in [
        "track_name", "artist_name", "track_popularity",
        "predicted_popularity", "abs_error", "playlist_genre", "playlist_subgenre"
    ] if c in worst_10.columns
]

print("=== 10 Worst Predictions (by absolute error) ===")
print(worst_10[cols_to_show])

# LEARNING CURVE 
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

rf = RandomForestRegressor(
    n_estimators=300, 
    random_state=42,
    n_jobs=-1
)

# learning curve
train_sizes, train_scores, val_scores = learning_curve(
    rf,
    X,
    y,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 8),
    shuffle=True,
    random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_mean, 'o-', label="Training R²")
plt.plot(train_sizes, val_mean, 'o-', label="Validation R²")
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)

plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.title("Learning Curve – Random Forest")
plt.legend()
plt.grid(True)
plt.show()


###  UPDATED ** EXTRACT AND PLOT FEATURE IMPORTANCE 
importances = rf.feature_importances_
feature_names = X.columns

indices = np.argsort(importances)[::-1]
top_n = 10
top_features = feature_names[indices][:top_n]
top_importances = importances[indices][:top_n]

plt.figure(figsize=(8, 6))
plt.barh(top_features[::-1], top_importances[::-1])
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature Name")
plt.title("Feature Importance: Top 10 Predictors of Song Popularity")
plt.tight_layout()
plt.savefig("feature_importance_top10.png", dpi=300, bbox_inches="tight")
plt.show()

feature_importance_caption = """
Figure X: Feature Importance: Top 10 Predictors of Song Popularity.
This figure shows the ten most important features used by the Random Forest model to predict song popularity.
Loudness, duration, energy, and tempo are the strongest predictors. However, the spread between importance scores indicates that no single 
feature is a dominant driver of popularity, consistent with the model’s limited generalization performance.
"""
print(feature_importance_caption)

### UPDATED LEARNING CURVE PLOT

from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    rf,
    X,
    y,
    cv=5,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 8),
    shuffle=True,
    random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, marker="o", label="Training R²")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)

plt.plot(train_sizes, val_mean, marker="o", label="Validation R²")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

plt.xlabel("Training Set Size (Number of Songs)")
plt.ylabel("R² Score")
plt.title("Learning Curve: Training vs. Validation R² for Random Forest")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve_rf.png", dpi=300, bbox_inches="tight")
plt.show()

learning_curve_caption = """
Figure X: Learning Curve: Training vs. Validation R² for Random Forest.
This plot shows how training and validation R² change as the training set size increases. Training R² is very high,
while validation R² stays low and plateaus early, indicating that the model memorizes the training data but does not gain
better generalization with more data. This pattern suggests that the limiting factor is missing or weak features rather than
insufficient data or model capacity.
"""
print(learning_curve_caption)

### UPDATED RESIDUALS VS PREDICTIONS 

y_pred_test = rf.predict(X_test)
residuals = y_test - y_pred_test

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Popularity (0–100)")
plt.ylabel("Residual (Actual – Predicted Popularity)")
plt.title("Residuals vs. Predicted Popularity: Random Forest Test Set")
plt.tight_layout()
plt.savefig("residuals_vs_predicted_rf.png", dpi=300, bbox_inches="tight")
plt.show()

residuals_caption = """
Figure X: Residuals vs. Predicted Popularity: Random Forest Test Set.
This plot shows prediction errors (residuals) as a function of predicted popularity. The wide, triangular spread of residuals 
for low predicted values indicates much higher error variance for low popularity songs, while errors are smaller and more 
consistent for higher predicted popularity. This heteroscedastic pattern confirms that the model struggles most with low popularity 
or niche tracks and performs relatively better on mainstream songs.
"""
print(residuals_caption)

### MODEL COMPARISON PLOT

model_names = ["Baseline", "Linear Regression", "Random Forest"]
r2_scores = [0, 0.138, 0.273]  

plt.figure(figsize=(8, 6))
plt.bar(model_names, r2_scores)
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Model Performance Comparison: Baseline vs. Linear Regression vs. Random Forest")
plt.ylim(0, max(r2_scores) + 0.1)
plt.tight_layout()
plt.savefig("model_comparison_r2.png", dpi=300, bbox_inches="tight")
plt.show()

model_comp_caption = """
Figure X: Model Performance Comparison: Baseline vs. Linear Regression vs. Random Forest.
This figure compares the R² scores of the baseline model, linear regression, and Random Forest on the test set.
Although the Random Forest achieves the highest R², overall performance remains modest, indicating that audio features alone 
do not fully explain variation in song popularity. The results support the conclusion that external factors not present in the 
dataset—such as marketing, playlist placement, and artist fame—play a major role in determining popularity.
"""
print(model_comp_caption)
