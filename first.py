import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.regressor import PredictionError
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import KFold, cross_validate
from yellowbrick.model_selection import FeatureImportances
from sklearn.model_selection import GridSearchCV
import pickle

df = pd.read_csv('flights.csv')

print(f'First 5 rows: {df.head()}')
print(f'Last 5 rows: {df.tail()}')
print(f'Shape: {df.shape}')
print(f'Info: {df.info()}')
print(f'Describe: {df.describe()}')
print(f'Columns: {df.columns}')
print(f'Dtypes: {df.dtypes}')
print(f'Isnull: {df.isnull().sum()}')
print(f'Isnull sum: {df.isnull().sum().sum()}')
print(f'Isnull sum percentage: {df.isnull().sum().sum() / df.size * 100}')
print(f'Isnull sum percentage: {df.isnull().sum().sum() / df.size * 100}')

print(f'Unique values: {df.nunique()}')
print(f'Unique values percentage: {df.nunique() / df.size * 100}')
print(f'Unique values percentage: {df.nunique() / df.size * 100}')

print(f'Describe object: {df.describe(include='O')}')

average_delay = df.groupby('airline')['delay'].mean()
print(f'Average delay: {average_delay}')

average_delay = df.groupby('airline')['delay'].mean().reset_index()
sns.barplot(x='airline', y='delay', data=average_delay)
plt.title('Average delay by airline')
plt.xlabel('Airline')
plt.ylabel('Average delay')
# plt.show()()

sns.countplot(df, x='airline')
plt.title('Count of flights by airline')
plt.xlabel('Airline')
plt.ylabel('Count')
# plt.show()()

average_delay = df.groupby('schengen')['delay'].mean().reset_index()
sns.barplot(x='schengen', y='delay', data=average_delay)
plt.title('Average delay by schengen')
plt.xlabel('Schengen')
plt.ylabel('Average delay')
# plt.show()()

average_delay = df.groupby('is_holiday')['delay'].mean().reset_index()
sns.barplot(x='is_holiday', y='delay', data=average_delay)
plt.title('Average delay by is_holiday')
plt.xlabel('Is holiday')
plt.ylabel('Average delay')
# plt.show()()

order = df['aircraft_type'].value_counts().index
sns.countplot(df, x='aircraft_type', order=order)
plt.title('Count of flights by aircraft type')
plt.xticks(rotation=70)
plt.xlabel('Aircraft type')
plt.ylabel('Count')
# plt.show()()

# Bin calculate
def bin_calculate(df, column):
    Q75, Q25 = np.percentile(df[column], [75, 25])
    IQR = Q75 - Q25
    lower_bound = Q25 - 1.5 * IQR
    upper_bound = Q75 + 1.5 * IQR
    larg_bin = 2 * IQR * np.power(len(df[column]), -1/3)
    return lower_bound, upper_bound, larg_bin

lower_bound, upper_bound, larg_bin = bin_calculate(df, 'arrival_time')
print(f'Lower bound: {lower_bound}')
print(f'Upper bound: {upper_bound}')
print(f'Larg bin: {larg_bin}')

# Arrival time distribution
sns.histplot(df, x='arrival_time', binwidth=larg_bin, kde=True)
plt.title('Arrival time distribution')
plt.xlabel('Arrival time')
plt.ylabel('Count')
# plt.show()()

# Departure time distribution
sns.histplot(df, x='departure_time', binwidth=larg_bin, kde=True)
plt.title('Departure time distribution')
plt.xlabel('Departure time')
plt.ylabel('Count')
# plt.show()()

mean_delay = df['delay'].mean()
median_delay = df['delay'].median()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(df, y='delay', ax=axes[0])
axes[0].set_title('Delay distribution Box Plot')

axes[0].axhline(y=mean_delay, color='red', linestyle='--', label='Mean')
axes[0].legend()


# Columns day and year
# Convert day of year to date using year and day columns
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + (df['day'] + 1).astype(str), format='%Y-%j')

print(df.head())

df['is_weekend'] = df['date'].dt.weekday.isin([5, 6])
print(df.head())

df['day_name'] = df['date'].dt.day_name()
print(df.head())

print(df.nunique())
print(df['is_weekend'].unique())

print(df['schengen'].replace({'non-schengen': 0, 'schengen': 1}))
print(df['is_holiday'].replace({'False': 0, 'True': 1}))
print(df['is_weekend'].replace({'False': 0, 'True': 1}))

# Replace categorical columns to numeric
df['schengen'] = df['schengen'].replace({'non-schengen': 0, 'schengen': 1})
df['is_holiday'] = df['is_holiday'].replace({False: 0, True: 1})
df['is_weekend'] = df['is_weekend'].replace({False: 0, True: 1})

print(df.head())

categorical_variables = ['airline', 'aircraft_type', 'origin', 'day_name']

# Dummy variables
df_encoded = pd.get_dummies(df, columns=categorical_variables, dtype=int)
print(df_encoded.head())


print(df_encoded[['arrival_time', 'departure_time']].corr())


df_clean = df_encoded.drop(columns=['year', 'day', 'date', 'flight_id', 'departure_time'], axis=1)
print(df_clean.head())

X = df_clean.drop(columns=['delay'], axis=1) # Take out the target variable
y = df_clean['delay'] # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model_dummy = DummyRegressor()

# Train the model
model_dummy.fit(X_train, y_train)

# Predict the target variable
y_pred_dummy = model_dummy.predict(X_test)

def calculate_metrcis_regressor(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    r2 = r2_score(y_test, y_pred)  # R-squared

    return {
        'Mean Squared Error': round(mse, 4),
        'Root Mean Squared Error': round(rmse, 4),
        'Mean Absolute Error': round(mae, 4),
        'R-squared Score': round(r2, 4)
    }

print(calculate_metrcis_regressor(y_test, y_pred_dummy)) # Base model

model_regressor = RandomForestRegressor(random_state=42, max_depth=5, n_estimators=100)

# Train the model
model_regressor.fit(X_train, y_train)

# Predict the target variable
y_pred_regressor = model_regressor.predict(X_test)

print(calculate_metrcis_regressor(y_test, y_pred_regressor)) # Random Forest model

# Convert to numpy arrays for visualization
visualizer = PredictionError(model_regressor)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
# visualizer.show()

# Residuals plot
viz = ResidualsPlot(model_regressor)
viz.fit(X_train.values, y_train.values)
viz.score(X_test.values, y_test.values)
# viz.show()

# K-fold cross-validation
scoring = {
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error',
    'r2': 'r2'
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

kfold_results = cross_validate(model_regressor, X_train, y_train, scoring=scoring, cv=kfold, return_train_score=True)
print(kfold_results)

for metric in scoring.keys():
    print(f"{metric.upper()} Scores: {[f'{val:.3f}' for val in kfold_results[f'test_{metric}']]}")
    print(f"{metric.upper()} Mean: {kfold_results[f'test_{metric}'].mean():.3f}")
    print(f"{metric.upper()} Std: {kfold_results[f'test_{metric}'].std():.3f}")
    print('-' * 50)

# Feature importance
feature_importance = model_regressor.feature_importances_
print(feature_importance)

# Feature importance plot
viz = FeatureImportances(model_regressor, relative=False, topn=10)
viz.fit(X_train, y_train)
# viz.show() # Show the feature importance plot

importance = model_regressor.feature_importances_
feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance}).sort_values(by='Importance', ascending=False)
print(feature_importance)

# Create DataFrame with only the metrics we want to track
results_df = pd.DataFrame(index=['MSE', 'RMSE', 'MAE', 'R2'])
model_selected_features = RandomForestRegressor(random_state=42, max_depth=5)

for count in [1, 5, 10, 15, 20, 25, 30]:
    selected_features = feature_importance['Feature'].values[:count]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model_selected_features.fit(X_train_selected, y_train)
    y_pred = model_selected_features.predict(X_test_selected)

    metrics = calculate_metrcis_regressor(y_test, y_pred)
    results_df[f'{count} Features'] = list(metrics.values())

print("\nModel performance with different numbers of features:")
print(results_df)

selected_features = feature_importance['Feature'].values[:13] # Select the top 13 features
X_selected = X[selected_features]  # Use X instead of X_train
print("\nSelected features data:")
print(X_selected.head())

# Split the data with selected features
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4, 6],
    'n_estimators': [100, 150, 200]
}

# Grid search
cv = KFold(n_splits=5, shuffle=True, random_state=42)

model_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid=param_grid, cv=cv, scoring='r2')
model_grid_search.fit(X_train, y_train)

print("\nBest parameters:")
print(model_grid_search.best_params_)
print("\nBest score:")
print(model_grid_search.best_score_)

# Save the model
try:
    with open('model_grid_search.pkl', 'wb') as f:
        pickle.dump(model_grid_search, f)
except Exception as e:
    print(f"Error saving model: ", str(e))

# Load the model
try:
    with open('model_grid_search.pkl', 'rb') as f:
        model_grid_search = pickle.load(f)
except Exception as e:
    print(f"Error loading model: ", str(e))






