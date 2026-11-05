# Bike-Rental-Project
first regression project

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