
<h1 align="center">Machine Learning S&P500 Stock Price Prediction</h1>
<h1 align="center">Bar Bettash </h1>
<p align="center">
<a href="https://www.linkedin.com/in/barbettash/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="https://www.linkedin.com/in/barbettash/" height="30" width="40" /></a>
</p>


## Objective

This project builds a Random Forest model to predict daily S&P 500 closing direction (higher or lower) using historical data. This can aid financial decisions like trading, risk management, and investment planning.


### Built With

Python <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a>

## Data Source

The historical data for the S&P 500 index is sourced from Yahoo Finance using the `yfinance` library. The data spans multiple decades, allowing for robust analysis and model training.

### Prerequisites


* Yfinance: Fetches financial data (e.g., stock prices, splits) from Yahoo Finance.
  ```sh
  pip install yfinance

* Matplotlib: Creates static, animated, and interactive visualizations (charts, graphs).
  ```sh
  pip install matplotlib

* Sklearn (scikit-learn): Provides machine learning algorithms and tools for data analysis.
  ```sh
  pip install scikit-learn
  
* Pandas: Offers high-performance data structures (Series, DataFrames) for data analysis and manipulation.
  ```sh
  pip install Pandas

## Key Features

### 1. Data Import and Visualization
- The data is imported using `yfinance` and includes historical records of the S&P 500 index.
- Optional visualization is provided to plot the closing prices over time, enabling a visual inspection of trends and patterns.

### 2. Data Cleaning
- Non-essential columns such as Dividends and Stock Splits are removed to streamline the dataset.
- The dataset is filtered to include data from January 1st, 1990 onwards to ensure consistency and relevance.

### 3. Feature Engineering
- A new target variable `Target` is created to indicate whether the index will close higher the next day.
- Historical data is shifted to align today's features with tomorrow's closing price, creating a supervised learning setup.

### 4. Model Training
- A Random Forest Classifier is used for prediction, chosen for its robustness and ability to handle complex datasets.
- The model is trained using features such as Close, Volume, Open, Low, and High prices.
- Training and test datasets are split to evaluate the model's performance on unseen data.

### 5. Model Evaluation
- The precision score is calculated to assess the accuracy of predictions.
- A combined plot is generated to visually compare the actual versus predicted movements.

### 6. Backtesting
- A backtesting methodology is implemented to evaluate the model's performance over historical data, ensuring the model's robustness and reliability.

### 7. Additional Feature Engineering
- Rolling averages and trend indicators are computed over various time horizons to capture longer-term market movements.
- New predictors such as `Close_Ratio` and `Trend` are added to enhance the model's predictive power.


### Example:
--------------------------------------------------
Predictions Summary (0=market go down): 

Number of Predictions

0.0    4347

1.0     844

--------------------------------------------------
Prediction Accuracy: 

0.5414691943127962

--------------------------------------------------

<!-- CONTACT -->
## Contact

<p align="left">
<a href="https://www.linkedin.com/in/barbettash/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="https://www.linkedin.com/in/barbettash/" height="30" width="40" /></a>
</p>


**bar.bettash.jobs@gmail.com** 


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This code is provided "as is" and may not function as intended in the future due to potential updates to external libraries, frameworks, websites, or APIs it interacts with. The code is no longer actively maintained and may require modifications to adapt to future changes.

**Recommendations:**

* Keep an eye on updates to libraries and dependencies to ensure compatibility.
* Be prepared to adapt the code based on future changes in the target website or API.

