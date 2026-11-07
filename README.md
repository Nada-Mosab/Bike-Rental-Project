# Bike-Rental-Project
first regression project

## dataset URL:
https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

# Comprehensive Exploratory Data Analysis (EDA) Report
**Dataset:** Seoul Bike Sharing Dataset  
**Focus:** Data Distributions, Feature Analysis, and Target Relationships (`Rented Bike Count`)

---

## 1. Categorical Variables Analysis & Statistical Breakdown
* **Granular Statistical Summary:** Computed descriptive statistics (count, mean, standard deviation, median, min, and max) for bike rentals across categorical variables, primarily **Seasons** and **Holiday**.
* **Comparative Visualizations:** Built structured bar plots comparing average bike rentals across different seasons, holiday statuses, and operational days (`Functioning Day`) to analyze behavioral demand patterns.

---

## 2. Numerical Variables Statistical Analysis & Distribution Inspection
* **Statistical Metrics:** Extracted comprehensive descriptive statistics for all continuous variables to observe central tendencies and spread.
* **Skewness Evaluation:** Calculated **Skewness** coefficients for all numerical features to systematically identify normal vs. skewed distributions (such as right-skewed target variables and weather extremes).
* **Bivariate Distribution Analysis:** Generated a **Pairplot** matrix across key continuous variables (`Rented Bike Count`, `Temperature(°C)`, `Humidity(%)`, and `Wind speed (m/s)`) to analyze pairwise distributions, correlations, and clustering trends.

---

## 3. Feature-Target Relationships & Multi-Variable Interactions
* **Target Correlation Analysis:** Evaluated continuous features directly impacting the target variable (`Rented Bike Count`).
* **Multi-Variable Cross-Plotting:** Created advanced scatter plots (e.g., Temperature vs. Rented Bike Count segmented by **Seasons**) to reveal how seasonal variations modulate the impact of weather conditions on daily rental demand.

---

## 4. Feature Engineering & Categorical Encoding
- **One-Hot Encoding:** Converted categorical and text features (**Seasons**, **Holiday**, and **Functioning Day**) into numerical binary columns using `pd.get_dummies` with `drop_first=True` to prevent multicollinearity.
- **Feature & Target Separation:** Isolated the target variable (`Rented Bike Count`) into `y` and independent predictive features into `X` by dropping non-informative columns like dates.

---

## 5. Train/Test Split & Feature Scaling
- **Data Splitting:** Divided the processed dataset into 80% training set and 20% testing set using `train_test_split` with a fixed random seed (`random_state=42`) to ensure reproducibility and prevent data leakage.
- **Feature Scaling:** Applied `StandardScaler` exclusively on the training features to normalize numerical distributions, transforming the training data via `fit_transform` and testing data via `transform`.

---

## 6. Model Training & Evaluation (Modeling Phase)
- **Baseline Model (Linear Regression):** Trained a linear regression model on the scaled features, evaluating performance using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$ Score).
- **Advanced Model (Random Forest Regressor):** Trained a non-linear ensemble tree-based model (`Random Forest Regressor`) using 100 estimators on the original unscaled features to capture complex interactions.
- **Model Comparison Table:** Consolidated performance metrics into a comparative evaluation dataframe, demonstrating the superiority of tree-based modeling in handling non-linear relationships for bike rental demand.

---

## 7. Model Persistence & Streamlit Application (`main.py`)
- **Model Serialization:** Saved trained models (`linear_regression_model.pkl`, `random_forest_model.pkl`), the fitted scaler (`scaler.pkl`), and feature columns list (`model_features.pkl`) securely inside the `Models/` directory using `joblib`.
- **Interactive Web App Deployment:** Built a fully responsive **Streamlit** application (`main.py`) allowing users to input environmental and temporal parameters dynamically, choose between models, receive real-time demand predictions, and view side-by-side model comparisons.