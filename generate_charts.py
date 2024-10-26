import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os

# Set the specific paths for your project
base_path = r"C:\Users\Brian\Desktop\Python Programming\Coursera\Applied-Data-Science-Capstone"
data_path = os.path.join(base_path, 'Datasets')
charts_path = os.path.join(base_path, 'Charts')

# Ensure Charts directory exists
if not os.path.exists(charts_path):
  os.makedirs(charts_path)

def load_data():
  """Load and prepare the dataset"""
  try:
      df = pd.read_csv(os.path.join(data_path, 'dataset_part_2.csv'))
      print("Dataset loaded successfully!")
      return df
  except Exception as e:
      print(f"Error loading file: {e}")
      raise

def create_launch_site_success_rate(df):
  """Create launch site success rate visualization"""
  site_success = df.groupby('Launch_Site')['Mission_Outcome'].apply(
      lambda x: (x == 'Success').mean()
  ).reset_index()
  site_success.columns = ['Launch_Site', 'Success_Rate']
  
  fig = px.bar(site_success, 
               x='Launch_Site', 
               y='Success_Rate',
               title='Launch Success Rate by Site',
               color='Success_Rate',
               color_continuous_scale='viridis')
  
  fig.update_layout(
      xaxis_title="Launch Site",
      yaxis_title="Success Rate",
      yaxis_tickformat='.0%'
  )
  fig.write_html(os.path.join(charts_path, "launch_success_by_site.html"))

def create_payload_analysis(df):
  """Create payload mass vs success visualization"""
  df['Success'] = (df['Mission_Outcome'] == 'Success').astype(int)
  
  fig = px.scatter(df, 
                  x='PAYLOAD_MASS__KG_',
                  y='Success',
                  color='Orbit',
                  title='Payload Mass vs Launch Success',
                  trendline="lowess")
  
  fig.update_layout(
      xaxis_title="Payload Mass (kg)",
      yaxis_title="Success (1=Success, 0=Failure)"
  )
  fig.write_html(os.path.join(charts_path, "payload_success.html"))

def create_orbit_distribution(df):
  """Create orbit type distribution visualization"""
  orbit_counts = df['Orbit'].value_counts()
  
  fig = px.pie(values=orbit_counts.values, 
               names=orbit_counts.index,
               title='Distribution of Orbit Types')
  
  fig.write_html(os.path.join(charts_path, "orbit_distribution.html"))

def create_feature_importance(df):
  """Create feature importance visualization"""
  # Prepare features
  categorical_columns = ['Booster_Version', 'Launch_Site', 'Orbit', 'Customer']
  df_encoded = pd.get_dummies(df[categorical_columns])
  
  # Add numerical columns
  df_encoded['PAYLOAD_MASS__KG_'] = df['PAYLOAD_MASS__KG_']
  
  X = df_encoded
  y = (df['Mission_Outcome'] == 'Success').astype(int)
  
  dt = DecisionTreeClassifier(random_state=42)
  dt.fit(X, y)
  
  feature_importance = pd.DataFrame({
      'feature': X.columns,
      'importance': dt.feature_importances_
  }).sort_values('importance', ascending=True)
  
  fig = px.bar(feature_importance.tail(10),
               x='importance',
               y='feature',
               orientation='h',
               title='Top 10 Features Importance in Launch Success Prediction')
  
  fig.update_layout(
      xaxis_title="Importance Score",
      yaxis_title="Feature"
  )
  fig.write_html(os.path.join(charts_path, "feature_importance.html"))

def create_success_over_time(df):
  """Create success rate over time visualization"""
  df['Year'] = pd.to_datetime(df['Date']).dt.year
  df['Success'] = (df['Mission_Outcome'] == 'Success').astype(int)
  yearly_success = df.groupby('Year')['Success'].mean().reset_index()
  
  fig = px.line(yearly_success,
                x='Year',
                y='Success',
                title='Launch Success Rate Over Time',
                markers=True)
  
  fig.update_layout(
      xaxis_title="Year",
      yaxis_title="Success Rate",
      yaxis_tickformat='.0%'
  )
  fig.write_html(os.path.join(charts_path, "success_over_time.html"))

def create_mission_outcomes(df):
  """Create mission outcomes visualization"""
  mission_outcomes = df['Mission_Outcome'].value_counts()
  
  fig = px.pie(values=mission_outcomes.values, 
               names=mission_outcomes.index,
               title='Distribution of Mission Outcomes')
  
  fig.write_html(os.path.join(charts_path, "mission_outcomes.html"))

def create_booster_version_analysis(df):
  """Create booster version success rate visualization"""
  booster_success = df.groupby('Booster_Version')['Mission_Outcome'].apply(
      lambda x: (x == 'Success').mean()
  ).reset_index()
  booster_success.columns = ['Booster_Version', 'Success_Rate']
  
  fig = px.bar(booster_success,
               x='Booster_Version',
               y='Success_Rate',
               title='Success Rate by Booster Version',
               color='Success_Rate',  # Use Success_Rate for color
               color_continuous_scale='Viridis')  # Change color scale
  
  fig.update_layout(
      xaxis_title="Booster Version",
      yaxis_title="Success Rate",
      yaxis_tickformat='.0%',
      showlegend=False
  )
  
  # Add data labels
  fig.for_each_trace(lambda t: t.update(name=t.name, text=t.y, textposition='auto'))
  
  fig.write_html(os.path.join(charts_path, "booster_success_rate.html"))

def main():
  """Main function to run all visualizations"""
  print("Starting visualization generation...")
  
  # Load data
  df = load_data()
  
  # Print dataset information
  print("\nDataset Shape:", df.shape)
  print("\nColumns in dataset:")
  print(df.columns.tolist())
  print("\nMissing values:")
  print(df.isnull().sum())
  
  # Generate all visualizations
  create_launch_site_success_rate(df)
  create_payload_analysis(df)
  create_orbit_distribution(df)
  create_feature_importance(df)
  create_success_over_time(df)
  create_mission_outcomes(df)
  create_booster_version_analysis(df)
  
  print("\nAll visualizations have been generated in the Charts folder:")
  for html_file in os.listdir(charts_path):
      if html_file.endswith('.html'):
          print(f"- {html_file}")

if __name__ == "__main__":
  main()