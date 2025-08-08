# In[1]: Importing necessary libraries

import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import warnings
import os


warnings.filterwarnings("ignore")


# In[2]: Reading the dataset

script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(script_dir, '..', '..', 'data','base_data', 'Water Database.xlsx')
data_path = os.path.abspath(data_path)

data = pd.read_excel(data_path)
data.head()



#In[3]: PreProcessing data
print(100*data.isnull().sum()/len(data))

data = data.drop(columns=['Feed_intake_kg', 'num_animals', 'NH3', 'Humidity']) #50% missing values and use kiko nb animal variable

# Replace missing by average (water_L is 20% missing)
# Impute water_L by month median
data['water_L'] = data.groupby(data['Date'].dt.month)['water_L'].transform(lambda x: x.fillna(x.median()))


data['CO2'].fillna(data['CO2'].mean(), inplace=True)
data['Outside_temperature'].fillna(data['Outside_temperature'].mean(), inplace=True)

# Drop row with missing values
data.dropna(inplace=True)

print(100*data.isnull().sum()/len(data))

# In[4]: Merging data at farm level

def aggregate_building_data(df):
    df['Date'] = pd.to_datetime(df['Date'])

    aggregation_rules = {
        'water_L': 'sum',
        'Inside_temperature':"mean",
        'CO2': 'sum',
        'Outside_temperature': 'mean',
        'Indiv_Water_mL': 'sum'
    }

    # Group by lotcode and date
    aggregated_df = df.groupby(['lotcode', 'Date'], as_index=False).agg(aggregation_rules)

    return aggregated_df

data = aggregate_building_data(data)
data.head()
daily_consumption = data.groupby('Date')['water_L'].sum()

plt.figure(figsize=(10, 5))
plt.plot(daily_consumption.index, daily_consumption.values)
plt.xlabel('Date')
plt.ylabel('Total Water Consumption (L)')
plt.title('Evolution of Water Consumption Over Time')
plt.tight_layout()
plt.show()

# In[4]: Correlation matrix

correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Basic")
plt.show()
print(data.describe())

# In[5]: Merge kiko dataset

def merge_on_date(env_data):

    census_data_path = os.path.join(script_dir, '..', '..', 'data','temp_results', 'filtered_census_t2m.csv')
    census_data_path = os.path.abspath(census_data_path)
    
    weight_data = pd.read_csv(census_data_path)
    weight_data.head()

    # S'assurer que les colonnes Date soient bien en datetime
    env_data["Date"] = pd.to_datetime(env_data["Date"], errors="coerce")
    weight_data["Date"] = pd.to_datetime(weight_data["Date"], errors="coerce")

    merged_data = pd.merge(env_data, weight_data, on="Date", how="inner")

    return merged_data

data = merge_on_date(data)
data.head()

# In[6]: Merged correlation matrix
for col in ['water_L']:
    lower, upper = data[col].quantile([0.01,0.99])
    data[col] = data[col].clip(lower, upper)

#In[7] : Water consumption evolution check

daily_consumption = data.groupby('Date')['water_L'].sum()

plt.figure(figsize=(10, 5))
plt.plot(daily_consumption.index, daily_consumption.values)
plt.xlabel('Date')
plt.ylabel('Total Water Consumption (L)')
plt.title('Evolution of Water Consumption Over Time')
plt.tight_layout()
plt.show()

data["avg_weight"] = data["avg_weight"].str.replace(",", ".", regex=False).astype(float)

data['Date'] = pd.to_datetime(data['Date']).dt.month  # Only month number remains
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Merged + month Correlation Matrix")
plt.show()


# In[8]: Droping useless variable

data = data.drop(columns=['lotcode', 'Indiv_Water_mL', 'CO2', 'Outside_temperature', 'avg_weight', 'Inside_temperature'])
print(data.describe())
# In[9]: Outliers detection

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x=data['water_L'])
plt.title('Water_L Outliers')

plt.tight_layout()
plt.show()


#In[10]: Splitting data into features and target variable

X = data.drop('water_L', axis = 1)
y = data['water_L']

# In[11]: Normalizing and Standardizing data

to_scale = [
    'num_anim_total',
    'total_weight',
    't2m_C',
]

# Build a pipeline: standardize → normalize
pipe = Pipeline([
    ('std', StandardScaler()),
    ('norm', MinMaxScaler())
])

X[to_scale] = pipe.fit_transform(X[to_scale])

print(X[to_scale].describe().loc[['mean','std','min','max']])

X['month_sin'] = np.sin(2 * np.pi * X['Date'] / 12)
X['month_cos'] = np.cos(2 * np.pi * X['Date'] / 12)

X.drop(columns=['Date'], inplace=True)


X.describe()


# In[12]: Creating training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 1)

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred).round(2))
print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)).round(2))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred).round(2)) 

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
plt.figure(figsize=(10, 8))
sns.heatmap(results.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Actual vs Predicted Correlation Heatmap")
plt.show()

# 1) Scatter actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)  # 45° line
plt.xlabel('Actual Total Water Consumption')
plt.ylabel('Predicted Total Water Consumption')
plt.title('Actual vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.hlines(0, y_pred.min(), y_pred.max(), linestyles='dashed', colors='r')
plt.xlabel('Predicted Total Water Consumption')
plt.ylabel('Residuals (Actual – Predicted)')
plt.title('Residuals vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[13]: Training the model with hyperparameter tuning

parameters = {
    'n_neighbors': np.arange(1, 30, 1),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': np.arange(30, 50, 1),
    'p': [1, 2]  # Manhattan (1) and Euclidean (2) distances
}

knn_grid = RandomizedSearchCV(
    estimator=knn,
    param_distributions=parameters, 
    n_iter=500, 
    cv=10,
    verbose=1,
    random_state=1,
    scoring='neg_mean_squared_error'
)

knn_grid.fit(X_train, y_train)

print("\nBest Parameters: ", knn_grid.best_params_)
print("Best Score (Negative MSE): ", knn_grid.best_score_)

y_pred = knn_grid.predict(X_test)


#In[14] : Checking error margin
y_test_1 = np.array(y_test)
y_pred_1 = np.array(y_pred)

pct_error = np.abs(y_pred_1 - y_test_1) / np.abs(y_test_1) * 100
bins = [0, 2, 5, 10,15,20,30, 50, np.inf]
labels = ['<2%', '2-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '>50%']

categories = np.digitize(pct_error, bins, right=False)

counts = {label: 0 for label in labels}
total = len(pct_error)
for idx in categories:
    counts[labels[idx-1]] += 1

percentages = {label: (counts[label] / total) * 100 for label in labels}
for label in labels:
    print(f"{label} error: {percentages[label]:.2f}% ({counts[label]} of {total})")

#In[15] : Checking other metrics

print("\nMean Squared Error: ", mean_squared_error(y_test, y_pred).round(2))
print("Root Mean Squared Error: ", np.sqrt(mean_squared_error(y_test, y_pred)).round(2))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred).round(2))

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
plt.figure(figsize=(10, 8))
sns.heatmap(results.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Actual vs Predicted Correlation Heatmap")
plt.show()

df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.round(2)})
print(df1.head())
# In[16]: Saving the model

best_knn = knn_grid.best_estimator_

# Scatter actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)  # 45° line
plt.xlabel('Actual Total Water Consumption')
plt.ylabel('Predicted Total Water Consumption')
plt.title('Actual vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2) Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.hlines(0, y_pred.min(), y_pred.max(), linestyles='dashed', colors='r')
plt.xlabel('Predicted Total Water Consumption')
plt.ylabel('Residuals (Actual – Predicted)')
plt.title('Residuals vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()

# Export the model
model_path = os.path.join(script_dir, '..', '..', 'data','models', 'knn_water_model.joblib')
model_path = os.path.abspath(model_path)
joblib.dump(best_knn, model_path)
print("Saved KNN model to knn_water_model.joblib")
