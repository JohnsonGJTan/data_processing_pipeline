# dataprocessing

A lightweight and intuitive Python library for building clear, reusable, and serializable data preprocessing pipelines for machine learning.

## Installation

To get started, clone the repository and install the package in editable mode. This is recommended for local development.

```bash
git clone https://github.com/JohnsonGJTan/dataprocessing.git
cd dataprocessing
pip install -e .
```

## Quickstart

Below is a simple example of how to define a pipeline, fit it on training data, and use it to transform new, unseen data.

```python
import pandas as pd
from dataprocessing import DataPipe, DataPipeline

# 1. Define Sample Data
train_data = pd.DataFrame({
    'age': [25, 30, None, 45, 25],
    'city': ['New York', 'London', 'London', 'Tokyo', 'New York'],
    'rating': ['low', 'medium', 'low', 'high', 'medium']
})
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

# 3. Create, Fit, and Transform
pipeline = DataPipeline(pipeline=pipeline_steps)
pipeline.fit(train_data)
transformed_data = pipeline.transform(test_data)

print("--- Transformed Test Data ---")
print(transformed_data)
#    age  rating  city_London  city_New York  city_Tokyo
# 0 60.0     2.0          0.0            0.0         0.0
# 1 30.0     0.0          1.0            0.0         0.0
# 2 35.0     1.0          0.0            1.0         0.0


# 4. Save and Load for Later Use
pipeline.save('my_fitted_pipeline.joblib')
loaded_pipeline = DataPipeline.load('my_fitted_pipeline.joblib')

print("\n--- Prediction Ready Data ---")
prediction_ready_data = loaded_pipeline.transform(
    pd.DataFrame({'age': [22], 'city': ['London'], 'rating': ['medium']})
)
print(prediction_ready_data)
#    age  rating  city_London  city_New York  city_Tokyo
# 0 22.0     1.0          1.0            0.0         0.0
```