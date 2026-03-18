# 6PASNMLN Machine Learning in Neuroscience
# Optimal Pipeline for Summative Assessment
# K22016090

# this pipeline trains the optimal pipeline discovered as part of the exploratory
# analysis and generates predictions on the hold out 

# === 1. Import Libraries ===
import os
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

# === 2. Load and Prepare the Data ===
print(" === Loading the data ===")

# check working directory to confirm you are in the right pathway
# assuming that  all data files will be located in the directory where the script is running
print(os.getcwd())

# Load the training data
print("Loading training data...")
data = pd.read_csv('data.csv', sep=',')

print(f"Original data shape: {data.shape}")

# Clean the data
print(" === Cleaning the data ===")
# Droping non-numeric columns such as participant ID
# this is to avoid data leakage and because this does not provide predictive information
data_numeric = data.select_dtypes(include=['number'])

# Remove zero variance columns
# these features are constant and offer no predictive information
feature_variance = data_numeric.drop(columns=['IST_intelligence_total']).var()
zero_var_features = feature_variance[feature_variance == 0].index.tolist()

if zero_var_features:
    data_numeric = data_numeric.drop(columns=zero_var_features)
    print(f"Removed {len(zero_var_features)} zero-variance features")    

# Seperate features (X) and target variables (y)
X_train = data_numeric.drop(columns=['IST_intelligence_total'])
y_train = data_numeric['IST_intelligence_total']

print(f"Final Features count: {X_train.shape[1]}")
print(f"Sample size: {X_train.shape[0]}")


# === 3. Build Optimal Pipeline ===
# Optimal model identified in exploratory pipeline: ElasticNet + PCA

# Hyperparameters were optimised during exploratory pipleine
# via Gridsearch with a 5-fold CV

optimal_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=5)),
    ('elasticnet', ElasticNet(
        alpha=10,
        l1_ratio=0.7,
        max_iter=10000,
        random_state=42
    ))
])

print("=== Optimal Pipeline Configuration ===")
print(" Model: ElasticNet + PCA")
print("Hyperparametrs:")
print("   - PCA components: 5")
print("   - Alpha: 10")
print("   - L1 ratio: 0.7")

# === 4. Train Model on all Training data ===
print("=== Training Model ===")
# we are fitting the pipeline on all available training data
# we are not doing a train/test split as validation was performed during the exploratory analysis

# lets fit the pipeline on the training data
optimal_pipeline.fit(X_train, y_train)
print("Training Complete")

# Report the training performances for verification
train_score = optimal_pipeline.score(X_train, y_train)
print(f"Training R²: {train_score:.4f}")

# === 5. Load and Prepare Holdout Dataset ===
print("=== Loading Holdout Dataset ===")
# load your seperate holdout dataset here
print("Loading holdout dataset...")
holdout_data = pd.read_csv('holdout.csv')

# we want to store participant IDS for output
participant_ids = holdout_data['participant_id'].values

# do some cleaning as in training
print("=== Cleaning Holdout Data ===")
holdout_numeric = holdout_data.select_dtypes(include=['number'])

# Remove target variable as it exists in holdout
if 'IST_intelligence_total' in holdout_numeric.columns:
   holdout_numeric = holdout_numeric.drop(columns=['IST_intelligence_total'])

# Remove columns with zero variance
if zero_var_features:
   features_to_drop = [f for f in zero_var_features if f in holdout_numeric.columns]
   if features_to_drop:
      holdout_numeric = holdout_numeric.drop(columns=features_to_drop)

X_holdout = holdout_numeric
print(f"Holdout features: {X_holdout.shape[1]}")
print(f"Holdout samples: {X_holdout.shape[0]}")

# === 6. Generate Predictions ===
print("=== Generating Predictions ===")
predictions = optimal_pipeline.predict(X_holdout)
print(f"Generated {len(predictions)} predictions")

# === 7. Save Predictions ===
print("=== Saving Predictions ===")
# create output dataframe with columns:
# Participant ID and IST Intelligence total
output_df = pd.DataFrame({
   'participant_id': participant_ids, 
   'IST_intelligence_total': predictions
})

# save to a CSV
output_filename = 'predictions.csv'
output_df.to_csv(output_filename, index = False) 

print(f"Predictions saved to : {output_filename}")

# === 8. Summary Statistics ===

print("=== Prediction Summary ===")
print(f"Mean: {predictions.mean():.2f}")
print(f"STD: {predictions.std():.2f}")
print(f"Min: {predictions.min():.2f}")
print(f"Max: {predictions.max():.2f}")
print("="*60)
print("\nPipeline executions complete")
