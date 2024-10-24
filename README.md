# Space Y Rocket Reusability Prediction Capstone Project 🚀

This project, part of the IBM Data Science Professional Certificate Capstone, aims to predict the success of first-stage rocket landings—critical for reducing launch costs for space companies like SpaceX. The project includes data collection, cleaning, exploratory data analysis (EDA), machine learning, and interactive visualizations. By analyzing past rocket launches, this project helps Space Y, a SpaceX competitor, optimize rocket reusability and reduce space travel costs.

## Project Overview

### Background
SpaceX has revolutionized space travel by reusing rockets and cutting launch costs. This project simulates Space Y’s efforts to achieve the same. We focus on predicting whether Space Y’s first-stage rockets will successfully land, using machine learning and historical data.

### Problem Statement
Rocket launches are expensive, with the first stage making up 70% of the cost. Reusing it can greatly reduce expenses. The goal of this project is to predict successful first-stage landings, helping Space Y cut costs and remain competitive.

## Data Collection

- **SpaceX API:** Historical rocket launch data (payload, orbit, launch site, etc.).
- **Web Scraping:** Supplementary data from Wikipedia.
- **CSV Files:** Pre-existing datasets for further analysis.

## Data Wrangling

- Handling missing values.
- Encoding categorical variables (Launch Sites, Orbits).
- Feature engineering (e.g., extracting years from dates).

## Exploratory Data Analysis (EDA)

EDA was done using visualizations and SQL to explore:

- Launch success by site and orbit.
- Trends over time.
- Payload mass effects on success rates.

## Machine Learning Models

We trained and evaluated several models, including:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Trees
- Support Vector Machines (SVM)

GridSearchCV was used for hyperparameter tuning, and the models were evaluated for accuracy, precision, recall, and confusion matrices.

## Interactive Visualizations

- **Folium Map:** Visualizing launch outcomes by location.
- **Plotly Dash:** An interactive dashboard for exploring the data and predictions.

## Results

Our machine learning models accurately predicted landing outcomes, helping Space Y make cost-optimizing decisions. We discovered key insights about flight number, payload mass, and orbit type trends affecting success rates.

## Conclusion

Data science and machine learning can significantly aid cost reduction and decision-making in the aerospace industry. Predicting rocket landing success allows Space Y to optimize reusability and stay competitive with industry leaders like SpaceX.

## Technologies Used

- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- Jupyter Notebooks
- SQL for Data Analysis
- Folium for mapping
- Plotly Dash for interactive dashboards
- APIs and Web Scraping

## Repository Structure

- **Data Collection & Wrangling Notebooks:** Code for web scraping, API data retrieval, and data preprocessing.
- **Exploratory Data Analysis (EDA) Notebooks:** Visualizations and insights from the dataset.
- **Machine Learning Notebooks:** Model building, tuning, and evaluation.
- **Interactive Visualizations:** Code for creating maps and dashboards.
