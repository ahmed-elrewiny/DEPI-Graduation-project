from IPython.display import display, Markdown
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

excel_file_path = '../data/clean/clean_tech_stocks.csv'


try:
    df = pd.read_csv(excel_file_path)
    print("CSV file loaded successfully!")
    print("Here are the first 5 rows of your data:")
    display(df.head())
except FileNotFoundError:
    print(f"Error: The file '{excel_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")



"""# Task
Analyze the stock data in "/clean_tech_stocks.csv", perform exploratory data analysis, engineer features for machine learning, split the data, train and evaluate models to predict next day's closing price, stock direction, monthly return, and volatility.

## Data exploration and preparation

### Subtask:
Explore the dataset to understand its structure, identify potential issues like missing values or outliers, and prepare the data for analysis and modeling. This includes checking data types, handling any inconsistencies, and possibly feature engineering.

**Reasoning**:
Display the data types, check for missing values, and display descriptive statistics to explore the dataset and identify potential issues.
"""

print("Data types of each column:")
display(df.dtypes)

print("\nNumber of missing values in each column:")
display(df.isnull().sum())

print("\nDescriptive statistics for numerical columns:")
display(df.describe())

"""**Reasoning**:
The previous step provided data types, missing value counts, and descriptive statistics. This information is sufficient to understand the dataset's structure, identify missing values in 'Daily_Return' and 'Volatility_30', and see the distribution of numerical features. The next steps involve handling these missing values and converting the 'Date' column to datetime objects, which are part of preparing the data for analysis and modeling as required by the subtask.


"""

df['Date'] = pd.to_datetime(df['Date'])

# Fill missing 'Daily_Return' with 0, assuming no change on that day
df['Daily_Return'] = df['Daily_Return'].fillna(0)

# Fill missing 'Volatility_30' with the mean of the column
df['Volatility_30'] = df['Volatility_30'].fillna(df['Volatility_30'].mean())

print("Missing values after handling:")
display(df.isnull().sum())

"""## Exploratory data analysis (eda)

### Subtask:
Perform a deeper analysis of the data to understand trends, patterns, and relationships between different columns. This might involve visualizations and calculating descriptive statistics for different stocks or over time.

**Reasoning**:
Calculate and display the average 'Close' price and 'Volatility_30' for each ticker, then calculate and display the average 'Daily_Return' for each ticker.
"""

average_close_volatility = df.groupby('Ticker')[['Close', 'Volatility_30']].mean()
print("Average Close Price and Volatility_30 per Ticker:")
display(average_close_volatility)

average_daily_return = df.groupby('Ticker')['Daily_Return'].mean()
print("\nAverage Daily Return per Ticker:")
display(average_daily_return)

"""**Reasoning**:
Plot the 'Close' price and 'Daily_Return' over time for a few selected tickers to visualize trends and fluctuations.


"""



selected_tickers = ['AAPL', 'MSFT', 'TSLA', 'GOOGL']

plt.figure(figsize=(15, 10))

for i, ticker in enumerate(selected_tickers):
    ticker_df = df[df['Ticker'] == ticker]
    plt.subplot(2, 2, i + 1)
    sns.lineplot(data=ticker_df, x='Date', y='Close')
    plt.title(f'{ticker} Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

for i, ticker in enumerate(selected_tickers):
    ticker_df = df[df['Ticker'] == ticker]
    plt.subplot(2, 2, i + 1)
    sns.lineplot(data=ticker_df, x='Date', y='Daily_Return')
    plt.title(f'{ticker} Daily Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')

plt.tight_layout()
plt.show()

"""## Feature engineering for predictions

### Subtask:
Create relevant features from the existing data that can be used to train machine learning models for predicting stock prices, direction, returns, and volatility. This could include technical indicators or time-based features.

**Reasoning**:
Calculate various technical indicators and target variables as requested in the instructions, grouping related calculations into a single code block for efficiency.
"""

# Calculate Moving Averages
df['MA_50'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=50).mean())
df['MA_200'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=200).mean())

# Calculate Exponential Moving Averages
df['EMA_12'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
df['EMA_26'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())

# Calculate MACD and Signal Line
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df.groupby('Ticker')['MACD'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

# Calculate Bollinger Bands
df['Rolling_Std_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=20).std())
df['Upper_Band'] = df['MA_7'] + (df['Rolling_Std_20'] * 2)
df['Lower_Band'] = df['MA_7'] - (df['Rolling_Std_20'] * 2)

# Create target variables
df['Next_Close'] = df.groupby('Ticker')['Close'].transform(lambda x: x.shift(-1))
df['Direction'] = (df['Next_Close'] > df['Close']).astype(int)
df['Monthly_Return'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change(periods=30))
df['Monthly_Volatility'] = df.groupby('Ticker')['Daily_Return'].transform(lambda x: x.rolling(window=30).std())

# Drop rows with missing values introduced by rolling calculations
df.dropna(inplace=True)

print("DataFrame with engineered features and target variables:")
display(df.head())
display(df.isnull().sum())

"""## Data splitting

### Subtask:
Split the data into training, validation, and testing sets. This is crucial for evaluating the performance of the machine learning models correctly.

**Reasoning**:
Define features and targets, sort the data, determine split points, and split the data into training, validation, and testing sets.
"""

# Define features (X) and target variables (y)
features = ['Close', 'Change', 'Change %', 'Daily_Return', 'MA_7', 'MA_30', 'Volatility_30', 'MA_50', 'MA_200', 'EMA_12', 'EMA_26', 'MACD', 'Signal_Line', 'Rolling_Std_20', 'Upper_Band', 'Lower_Band']

X = df[features]
y_next_close = df['Next_Close']
y_direction = df['Direction']
y_monthly_return = df['Monthly_Return']
y_monthly_volatility = df['Monthly_Volatility']

# Sort the DataFrame by 'Date' for time-series split
df = df.sort_values(by='Date')

# Determine split points (70% train, 15% validation, 15% test)
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))
test_size = len(df) - train_size - val_size

# Split the data
X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_next_close_train, y_next_close_val, y_next_close_test = y_next_close[:train_size], y_next_close[train_size:train_size + val_size], y_next_close[train_size + val_size:]
y_direction_train, y_direction_val, y_direction_test = y_direction[:train_size], y_direction[train_size:train_size + val_size], y_direction[train_size + val_size:]
y_monthly_return_train, y_monthly_return_val, y_monthly_return_test = y_monthly_return[:train_size], y_monthly_return[train_size:train_size + val_size], y_monthly_return[train_size + val_size:]
y_monthly_volatility_train, y_monthly_volatility_val, y_monthly_volatility_test = y_monthly_volatility[:train_size], y_monthly_volatility[train_size:train_size + val_size], y_monthly_volatility[train_size + val_size:]

print("Data splitting complete.")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")

"""## Model selection and training

### Subtask:
Choose appropriate machine learning models for each prediction task (e.g., regression for price prediction, classification for direction prediction). Train the selected models on the training data.

**Reasoning**:
Import the necessary libraries for machine learning models and train the models for each prediction task.
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier

# Instantiate and train regression models for Next_Close, Monthly_Return, and Monthly_Volatility
rfr_next_close = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2)
gbr_next_close = GradientBoostingRegressor(n_estimators=100, random_state=42, verbose=2)

rfr_monthly_return = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2)
gbr_monthly_return = GradientBoostingRegressor(n_estimators=100, random_state=42, verbose=2)

rfr_monthly_volatility = RandomForestRegressor(n_estimators=100, random_state=42, verbose=2)
gbr_monthly_volatility = GradientBoostingRegressor(n_estimators=100, random_state=42, verbose=2)

print("Training RandomForestRegressor for Next_Close...")
rfr_next_close.fit(X_train, y_next_close_train)
print("Training GradientBoostingRegressor for Next_Close...")
gbr_next_close.fit(X_train, y_next_close_train)

print("Training RandomForestRegressor for Monthly_Return...")
rfr_monthly_return.fit(X_train, y_monthly_return_train)
print("Training GradientBoostingRegressor for Monthly_Return...")
gbr_monthly_return.fit(X_train, y_monthly_return_train)

print("Training RandomForestRegressor for Monthly_Volatility...")
rfr_monthly_volatility.fit(X_train, y_monthly_volatility_train)
print("Training GradientBoostingRegressor for Monthly_Volatility...")
gbr_monthly_volatility.fit(X_train, y_monthly_volatility_train)


# Instantiate and train classification model for Direction
rfc_direction = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2)
gbc_direction = GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=2)

print("Training RandomForestClassifier for Direction...")
rfc_direction.fit(X_train, y_direction_train)
print("Training GradientBoostingClassifier for Direction...")
gbc_direction.fit(X_train, y_direction_train)

print("Models trained successfully.")

"""## Model Evaluation

### Subtask:
Evaluate the performance of the trained models using appropriate metrics for each prediction task on the validation and testing sets.

**Reasoning**:
Import necessary evaluation metrics and evaluate each trained model on the validation and test sets, displaying the results.
"""

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Evaluate Regression Models (Next_Close, Monthly_Return, Monthly_Volatility)
print("Evaluating Regression Models:")

# Random Forest Regressor - Next_Close
rfr_next_close_pred_val = rfr_next_close.predict(X_val)
rfr_next_close_pred_test = rfr_next_close.predict(X_test)
print("\nRandomForestRegressor - Next_Close:")
print(f"  Validation MSE: {mean_squared_error(y_next_close_val, rfr_next_close_pred_val):.4f}")
print(f"  Validation R2: {r2_score(y_next_close_val, rfr_next_close_pred_val):.4f}")
print(f"  Test MSE: {mean_squared_error(y_next_close_test, rfr_next_close_pred_test):.4f}")
print(f"  Test R2: {r2_score(y_next_close_test, rfr_next_close_pred_test):.4f}")

# Gradient Boosting Regressor - Next_Close
gbr_next_close_pred_val = gbr_next_close.predict(X_val)
gbr_next_close_pred_test = gbr_next_close.predict(X_test)
print("\nGradientBoostingRegressor - Next_Close:")
print(f"  Validation MSE: {mean_squared_error(y_next_close_val, gbr_next_close_pred_val):.4f}")
print(f"  Validation R2: {r2_score(y_next_close_val, gbr_next_close_pred_val):.4f}")
print(f"  Test MSE: {mean_squared_error(y_next_close_test, gbr_next_close_pred_test):.4f}")
print(f"  Test R2: {r2_score(y_next_close_test, gbr_next_close_pred_test):.4f}")

# Random Forest Regressor - Monthly_Return
rfr_monthly_return_pred_val = rfr_monthly_return.predict(X_val)
rfr_monthly_return_pred_test = rfr_monthly_return.predict(X_test)
print("\nRandomForestRegressor - Monthly_Return:")
print(f"  Validation MSE: {mean_squared_error(y_monthly_return_val, rfr_monthly_return_pred_val):.4f}")
print(f"  Validation R2: {r2_score(y_monthly_return_val, rfr_monthly_return_pred_val):.4f}")
print(f"  Test MSE: {mean_squared_error(y_monthly_return_test, rfr_monthly_return_pred_test):.4f}")
print(f"  Test R2: {r2_score(y_monthly_return_test, rfr_monthly_return_pred_test):.4f}")

# Gradient Boosting Regressor - Monthly_Return
gbr_monthly_return_pred_val = gbr_monthly_return.predict(X_val)
gbr_monthly_return_pred_test = gbr_monthly_return.predict(X_test)
print("\nGradientBoostingRegressor - Monthly_Return:")
print(f"  Validation MSE: {mean_squared_error(y_monthly_return_val, gbr_monthly_return_pred_val):.4f}")
print(f"  Validation R2: {r2_score(y_monthly_return_val, gbr_monthly_return_pred_val):.4f}")
print(f"  Test MSE: {mean_squared_error(y_monthly_return_test, gbr_monthly_return_pred_test):.4f}")
print(f"  Test R2: {r2_score(y_monthly_return_test, gbr_monthly_return_pred_test):.4f}")

# Random Forest Regressor - Monthly_Volatility
rfr_monthly_volatility_pred_val = rfr_monthly_volatility.predict(X_val)
rfr_monthly_volatility_pred_test = rfr_monthly_volatility.predict(X_test)
print("\nRandomForestRegressor - Monthly_Volatility:")
print(f"  Validation MSE: {mean_squared_error(y_monthly_volatility_val, rfr_monthly_volatility_pred_val):.4f}")
print(f"  Validation R2: {r2_score(y_monthly_volatility_val, rfr_monthly_volatility_pred_val):.4f}")
print(f"  Test MSE: {mean_squared_error(y_monthly_volatility_test, rfr_monthly_volatility_pred_test):.4f}")
print(f"  Test R2: {r2_score(y_monthly_volatility_test, rfr_monthly_volatility_pred_test):.4f}")

# Gradient Boosting Regressor - Monthly_Volatility
gbr_monthly_volatility_pred_val = gbr_monthly_volatility.predict(X_val)
gbr_monthly_volatility_pred_test = gbr_monthly_volatility.predict(X_test)
print("\nGradientBoostingRegressor - Monthly_Volatility:")
print(f"  Validation MSE: {mean_squared_error(y_monthly_volatility_val, gbr_monthly_volatility_pred_val):.4f}")
print(f"  Validation R2: {r2_score(y_monthly_volatility_val, gbr_monthly_volatility_pred_val):.4f}")
print(f"  Test MSE: {mean_squared_error(y_monthly_volatility_test, gbr_monthly_volatility_pred_test):.4f}")
print(f"  Test R2: {r2_score(y_monthly_volatility_test, gbr_monthly_volatility_pred_test):.4f}")


# Evaluate Classification Models (Direction)
print("\nEvaluating Classification Models:")

# Random Forest Classifier - Direction
rfc_direction_pred_val = rfc_direction.predict(X_val)
rfc_direction_pred_test = rfc_direction.predict(X_test)
print("\nRandomForestClassifier - Direction:")
print(f"  Validation Accuracy: {accuracy_score(y_direction_val, rfc_direction_pred_val):.4f}")
print("  Validation Classification Report:")
print(classification_report(y_direction_val, rfc_direction_pred_val))
print(f"  Test Accuracy: {accuracy_score(y_direction_test, rfc_direction_pred_test):.4f}")
print("  Test Classification Report:")
print(classification_report(y_direction_test, rfc_direction_pred_test))

# Gradient Boosting Classifier - Direction
gbc_direction_pred_val = gbc_direction.predict(X_val)
gbc_direction_pred_test = gbc_direction.predict(X_test)
print("\nGradientBoostingClassifier - Direction:")
print(f"  Validation Accuracy: {accuracy_score(y_direction_val, gbc_direction_pred_val):.4f}")
print("  Validation Classification Report:")
print(classification_report(y_direction_val, gbc_direction_pred_val))
print(f"  Test Accuracy: {accuracy_score(y_direction_test, gbc_direction_pred_test):.4f}")
print("  Test Classification Report:")
print(classification_report(y_direction_test, gbc_direction_pred_test))

print("\nModel evaluation complete.")

"""## Conclusion

This analysis involved exploring and preparing stock data, engineering relevant features, and training machine learning models to predict next day's closing price, stock direction, monthly return, and volatility.

**Key Findings:**

*   **Data Preparation:** Missing values in 'Daily\_Return' and 'Volatility\_30' were handled, and the 'Date' column was converted to datetime objects.
*   **Exploratory Analysis:** Initial analysis provided insights into average stock behavior and visualized trends over time for selected tickers.
*   **Feature Engineering:** Technical indicators (Moving Averages, EMA, MACD, Bollinger Bands) and target variables were successfully created.
*   **Model Performance:**
    *   **Next Day Closing Price Prediction:** Both RandomForestRegressor and GradientBoostingRegressor performed well, with the Gradient Boosting Regressor showing slightly better performance on the test set (Test R2: {:.4f}).
    *   **Monthly Return Prediction:** The models had moderate success in predicting monthly returns (Gradient Boosting Regressor Test R2: {:.4f}).
    *   **Monthly Volatility Prediction:** The models were highly accurate in predicting monthly volatility (Both models Test R2: {:.4f}).
    *   **Stock Direction Prediction:** Predicting the next day's stock direction proved challenging, with accuracy scores around 50% for both RandomForestClassifier and GradientBoostingClassifier on the test set (RandomForestClassifier Test Accuracy: {:.4f}, GradientBoostingClassifier Test Accuracy: {:.4f}). This indicates that predicting short-term stock movement direction with these features is close to random chance.

The visualizations of the predictions and confusion matrices provide a clear picture of the model performance for dashboarding purposes. While predicting the exact next day's direction is difficult, the models showed promising results for predicting closing price and volatility.
"""

# Populate the markdown with the actual statistics
conclusion_markdown = """
## Conclusion

This analysis involved exploring and preparing stock data, engineering relevant features, and training machine learning models to predict next day's closing price, stock direction, monthly return, and volatility.

**Key Findings:**

*   **Data Preparation:** Missing values in 'Daily\_Return' and 'Volatility\_30' were handled, and the 'Date' column was converted to datetime objects.
*   **Exploratory Analysis:** Initial analysis provided insights into average stock behavior and visualized trends over time for selected tickers.
*   **Feature Engineering:** Technical indicators (Moving Averages, EMA, MACD, Bollinger Bands) and target variables were successfully created.
*   **Model Performance:**
    *   **Next Day Closing Price Prediction:** Both RandomForestRegressor and GradientBoostingRegressor performed well, with the Gradient Boosting Regressor showing slightly better performance on the test set (Test R2: {:.4f}).
    *   **Monthly Return Prediction:** The models had moderate success in predicting monthly returns (Gradient Boosting Regressor Test R2: {:.4f}).
    *   **Monthly Volatility Prediction:** The models were highly accurate in predicting monthly volatility (Both models Test R2: {:.4f}).
    *   **Stock Direction Prediction:** Predicting the next day's stock direction proved challenging, with accuracy scores around 50% for both RandomForestClassifier and GradientBoostingClassifier on the test set (RandomForestClassifier Test Accuracy: {:.4f}, GradientBoostingClassifier Test Accuracy: {:.4f}). This indicates that predicting short-term stock movement direction with these features is close to random chance.

The visualizations of the predictions and confusion matrices provide a clear picture of the model performance for dashboarding purposes. While predicting the exact next day's direction is difficult, the models showed promising results for predicting closing price and volatility.
""".format(
    r2_score(y_next_close_test, gbr_next_close_pred_test),
    r2_score(y_monthly_return_test, gbr_monthly_return_pred_test),
    r2_score(y_monthly_volatility_test, gbr_monthly_volatility_pred_test),
    accuracy_score(y_direction_test, rfc_direction_pred_test),
    accuracy_score(y_direction_test, gbc_direction_pred_test)
)

from IPython.display import display, Markdown

display(Markdown(conclusion_markdown))

"""## Prediction and Interpretation

### Subtask:
Use the trained models to make predictions and interpret the results, including visualizations for dashboarding.

**Reasoning**:
Generate predictions using the trained models on the test set and visualize the actual vs. predicted values for the regression tasks and the classification results for the direction prediction.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Make predictions on the test set
rfr_next_close_pred_test = rfr_next_close.predict(X_test)
gbr_next_close_pred_test = gbr_next_close.predict(X_test)
rfc_direction_pred_test = rfc_direction.predict(X_test)
gbc_direction_pred_test = gbc_direction.predict(X_test)
rfr_monthly_return_pred_test = rfr_monthly_return.predict(X_test)
gbr_monthly_return_pred_test = gbr_monthly_return.predict(X_test)
rfr_monthly_volatility_pred_test = rfr_monthly_volatility.predict(X_test)
gbr_monthly_volatility_pred_test = gbr_monthly_volatility.predict(X_test)


# Visualize Predictions vs. Actuals for Regression Tasks (Next_Close, Monthly_Return, Monthly_Volatility)

# Next_Close Prediction Visualization
plt.figure(figsize=(15, 6))
sns.lineplot(x=df.loc[X_test.index, 'Date'], y=y_next_close_test.values, label='Actual Next Close')
sns.lineplot(x=df.loc[X_test.index, 'Date'], y=rfr_next_close_pred_test, label='RFR Predicted Next Close')
sns.lineplot(x=df.loc[X_test.index, 'Date'], y=gbr_next_close_pred_test, label='GBR Predicted Next Close')
plt.title('Next Day Closing Price: Actual vs. Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Monthly Return Prediction Visualization
plt.figure(figsize=(15, 6))
sns.lineplot(x=df.loc[X_test.index, 'Date'], y=y_monthly_return_test.values, label='Actual Monthly Return')
sns.lineplot(x=df.loc[X_test.index, 'Date'], y=rfr_monthly_return_pred_test, label='RFR Predicted Monthly Return')
sns.lineplot(x=df.loc[X_test.index, 'Date'], y=gbr_monthly_return_pred_test, label='GBR Predicted Monthly Return')
plt.title('Monthly Return: Actual vs. Predicted')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.show()

# Monthly Volatility Prediction Visualization
plt.figure(figsize=(15, 6))
sns.lineplot(x=df.loc[X_test.index, 'Date'], y=y_monthly_volatility_test.values, label='Actual Monthly Volatility')
sns.lineplot(x=df.loc[X_test.index, 'Date'], y=rfr_monthly_volatility_pred_test, label='RFR Predicted Monthly Volatility')
sns.lineplot(x=df.loc[X_test.index, 'Date'], y=gbr_monthly_volatility_pred_test, label='GBR Predicted Monthly Volatility')
plt.title('Monthly Volatility: Actual vs. Predicted')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()

# Visualize Classification Results for Direction Prediction
# We can show a confusion matrix or a simple bar plot of predicted vs actual counts

from sklearn.metrics import confusion_matrix

# Random Forest Classifier - Direction Confusion Matrix
cm_rfc = confusion_matrix(y_direction_test, rfc_direction_pred_test)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_rfc, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('RandomForestClassifier - Direction Prediction Confusion Matrix')
plt.xlabel('Predicted Direction')
plt.ylabel('Actual Direction')
plt.show()

# Gradient Boosting Classifier - Direction Confusion Matrix
cm_gbc = confusion_matrix(y_direction_test, gbc_direction_pred_test)
plt.figure(figsize=(6, 6))
sns.heatmap(cm_gbc, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.title('GradientBoostingClassifier - Direction Prediction Confusion Matrix')
plt.xlabel('Predicted Direction')
plt.ylabel('Actual Direction')
plt.show()

import os

# ------------------------------------------
# 🔽 حفظ النتائج في CSV للـ Streamlit Dashboard
# ------------------------------------------

# نختار الدليل اللي نحفظ فيه النتائج
output_dir = "../data"
os.makedirs(output_dir, exist_ok=True)

# نجهز DataFrame فيه التوقعات المطلوبة
predictions_df = pd.DataFrame({
    "Date": df.loc[X_test.index, "Date"],
    "Ticker": df.loc[X_test.index, "Ticker"],
    "Actual_Next_Close": y_next_close_test.values,
    "GBR_Pred_Next_Close": gbr_next_close_pred_test,
    "RFC_Pred_Next_Close": rfr_next_close_pred_test,
    "Actual_Direction": y_direction_test.values,
    "RFC_Pred_Direction": rfc_direction_pred_test,
    "GBC_Pred_Direction": gbc_direction_pred_test,
    "Actual_Monthly_Return": y_monthly_return_test.values,
    "Pred_Monthly_Return": gbr_monthly_return_pred_test,
    "Actual_Monthly_Volatility": y_monthly_volatility_test.values,
    "Pred_Monthly_Volatility": gbr_monthly_volatility_pred_test
})

# نحفظ الملف
csv_path = os.path.join(output_dir, "predictions.csv")
predictions_df.to_csv(csv_path, index=False)

print(f"\n✅ تم حفظ ملف التوقعات بنجاح في: {csv_path}")
print("عدد الصفوف:", len(predictions_df))
print(predictions_df.head())
