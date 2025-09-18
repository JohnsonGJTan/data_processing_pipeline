import pandas as pd
from dataprocessingpipeline import DataPipe, DataPipeline

# 1. Sample Data
train_data = pd.DataFrame({
    'age': [25, 30, None, 45, 25],
    'city': ['New York', 'London', 'London', 'Tokyo', 'New York'],
    'rating': ['low', 'medium', 'low', 'high', 'medium']
})
# Convert columns to appropriate categorical dtypes
train_data['city'] = train_data['city'].astype('category')
train_data['rating'] = pd.Categorical(train_data['rating'], categories=['low', 'medium', 'high'], ordered=True)

test_data = pd.DataFrame({
    'age': [60, None, 35],
    'city': ['Paris', 'London', 'New York'],
    'rating': ['high', 'low', 'medium']
})
test_data['city'] = test_data['city'].astype('category')
test_data['rating'] = pd.Categorical(test_data['rating'], categories=['low', 'medium', 'high'], ordered=True)

# 2. Define the Pipeline Steps
pipeline_steps = [
    DataPipe('num_impute', {'col_name': 'age', 'method': 'median'}),
    DataPipe('one_hot_encode', {'col_names': ['city'], 'handle_unknown': 'ignore'}),
    DataPipe('ordinal_encode', {'col_names': ['rating'], 'orders': [['low', 'medium', 'high']]})
]

# 3. Create and Fit the Pipeline
pipeline = DataPipeline(pipeline=pipeline_steps)
pipeline.fit(train_data)

# 4. Transform New Data
transformed_data = pipeline.transform(test_data)

print("--- Transformed Test Data ---")
print(transformed_data)

# 5. Save and Load the Fitted Pipeline
pipeline.save('my_fitted_pipeline.joblib')
loaded_pipeline = DataPipeline.load('my_fitted_pipeline.joblib')

# Use the loaded pipeline for inference
new_data = pd.DataFrame({'age': [22], 'city': ['London'], 'rating': ['medium']})
new_data['city'] = new_data['city'].astype('category')
new_data['rating'] = pd.Categorical(new_data['rating'], categories=['low', 'medium', 'high'], ordered=True)

prediction_ready_data = loaded_pipeline.transform(new_data)
print("\n--- Prediction Ready Data ---")
print(prediction_ready_data)