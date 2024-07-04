
### Description
# This Python script is designed to predict the movement of the S&P 500 index for the following day.
# It utilizes historical data from Yahoo Finance and implements a Random Forest Classifier to make predictions.
# The script involves several steps including data import, data cleaning, feature engineering, model training, prediction, and backtesting to evaluate the model's performance.

############################################################
# IMPORT DATA #
############################################################

# Function to import S&P 500 data from Yahoo Finance
def import_sp500data():
    import yfinance as yf # Importing the yfinance library for financial data
    sp500 = yf.Ticker("^GSPC") # Getting S&P500 data using its ticker symbol (^GSPC)
    sp500 = sp500.history(period="max") # Fetching all available historical data
    return sp500

sp500 = import_sp500data() # Assign the imported data to the variable sp500

# OPTIONAL: Function to view the raw data and plot the closing prices
def view_sp500data():
    print(sp500) # Print the raw data
    import matplotlib.pyplot as plt # Importing matplotlib for plotting
    sp500.plot.line(y="Close", use_index=True) # Plotting the closing prices with dates as the x-axis
    plt.show() # Display the plot

# OPTIONAL: Uncomment the following line to view the data and plot
# view_sp500data() 

############################################################
# CLEAN DATA #
############################################################

# Function to clean the S&P 500 data
def clean_sp500data(sp500):
    columns_to_remove = ["Dividends", "Stock Splits"] # Columns that are not needed for our analysis
    for col in columns_to_remove:
        if (col in sp500.columns):
            sp500 = sp500.drop(columns=[col]) # Remove unnecessary columns
    
    sp500 = sp500.loc["1990-01-01":].copy() # Filtering data to only include dates from January 1st, 1990 onwards
    return sp500

sp500 = clean_sp500data(sp500) # Clean the data using the function

# OPTIONAL: Uncomment the following line to view the cleaned data and plot
# view_sp500data() 

############################################################
# ML TARGET #
############################################################

# Creating the target variable for the machine learning model
sp500["Tomorrow"] = sp500["Close"].shift(-1) # Shifting the close prices by one day to create the target variable
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int) # Binary target: 1 if the price goes up, 0 otherwise

############################################################
# ML MODEL #
############################################################

# Importing required libraries for machine learning
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Setting up the Random Forest model parameters
model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Splitting the data into training and testing sets
train = sp500.iloc[:-100] # Training data excluding the last 100 entries
test = sp500.iloc[-100:] # Testing data consisting of the last 100 entries

# Defining predictor variables to be used for training the model
predictors = ["Close", "Volume", "Open", "Low", "High"]

# Training the model using the training data
model.fit(train[predictors], train["Target"])

# Making predictions on the test data
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index) # Creating a Pandas Series for the predictions

# OPTIONAL: Uncomment the following line to view the predictions
# print(preds)

# Evaluating the model's accuracy using precision score
from sklearn.metrics import precision_score
score = precision_score(test["Target"], preds) # Calculating the precision score
# OPTIONAL: Uncomment the following line to view the prediction score
# print(f"Prediction score = {score}")

# Plotting actual vs predicted values
import matplotlib.pyplot as plt
combined = pd.concat([test["Target"], preds], axis=1) # Combining actual and predicted values into a single DataFrame
combined.plot() # Plotting the results
# OPTIONAL: Uncomment the following line to display the plot
# plt.show()

############################################################
# ML BACKTESTING #
############################################################

# Function to predict and combine the results for training and testing data
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"]) # Training the model on the training data
    preds = model.predict_proba(test[predictors])[:, 1] # Getting probabilities of the positive class (price going up)
    preds[preds >= 0.6] = 1 # Thresholding probabilities to 1 if >= 60%
    preds[preds < 0.6] = 0 # Thresholding probabilities to 0 if < 60%
    preds = pd.Series(preds, index=test.index, name="Predictions") # Creating a Pandas Series for the predictions
    combined = pd.concat([test["Target"], preds], axis=1) # Combining actual and predicted values into a single DataFrame
    return combined

# Function to backtest the model with a rolling window approach
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = [] # List to store all predictions
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy() # Training data up to the current step
        test = data.iloc[i:(i+step)].copy() # Testing data for the current step
        predictions = predict(train, test, predictors, model) # Making predictions
        all_predictions.append(predictions) # Storing the predictions
    return pd.concat(all_predictions) # Concatenating all predictions into a single DataFrame

# Function to print prediction statistics
def print_predictions_stats(sp500, model, predictors):
    predictions = backtest(sp500, model, predictors) # Backtesting the model
    print("-" * 50)
    print("Predictions Summary (0=market go down): ")
    print(predictions["Predictions"].value_counts()) # Printing the count of predictions
    print("-" * 50)
    print("Prediction Accuracy: ")
    print(precision_score(predictions["Target"], predictions["Predictions"])) # Printing the precision score
    print("-" * 50)

# Defining longer time frames to recognize market trends
horizons = [2, 5, 60, 250, 1000] # Different horizons in days
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean() # Calculating rolling averages for the specified horizon

    ratio_column = f"Close_Ratio_{horizon}" # Naming the ratio column
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"] # Calculating the ratio of the current close price to the rolling average

    trend_column = f"Trend_{horizon}" # Naming the trend column
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"] # Calculating the sum of the target variable over the rolling window

    new_predictors += [ratio_column, trend_column] # Adding the new predictors to the list

sp500 = sp500.dropna() # Dropping any rows with NaN values

# Running and printing the backtesting results on the updated data
print_predictions_stats(sp500, model, predictors)
