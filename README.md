# DataProcessingPipeline: A Declarative Data Preprocessing Library

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DataProcessingPipeline is a lightweight, intuitive Python library for building clear, reusable, and serializable data preprocessing pipelines for machine learning. It's designed to bring clarity and reproducibility to your feature engineering workflow by combining a declarative API with robust data schema management.

## Key Features

* **Declarative API:** Define your preprocessing steps as a simple list of operations and parameters.
* **Schema-Aware:** Tracks changes to your data's structure (new columns, type changes) at every step.
* **Scikit-learn Inspired:** Follows the familiar `fit`/`transform` paradigm, making it intuitive to use.
* **Serializable:** Save your fitted pipeline to a file using `joblib` and load it back later for inference, ensuring consistency between training and production.
* **JSON Configurable:** Define an entire pipeline structure in a JSON file for easy configuration management.

## Installation

```bash
# Clone the repository and install locally in editable mode
git clone [https://github.com/JohnsonGJTan/data_processing_pipeline.git](https://github.com/JohnsonGJTan/data_processing_pipeline.git)
cd data_processing_pipeline
pip install -e .
```

## Core Concepts

* **`DataSchema`**: A metadata object that describes the structure of your DataFrame, tracking `continuous`, `nominal` (unordered), and `ordinal` (ordered) categorical columns.
* **`DataPipe`**: A wrapper for a single processing step (e.g., `'num_impute'`) and its parameters (e.g., `{'col_name': 'age', 'method': 'median'}`).
* **`DataPipeline`**: A sequence of `DataPipe` objects that can be fitted to training data and then used to transform new data.

## Quickstart

Here's how to build, fit, transform, and save a pipeline.

```python
import pandas as pd
from DataProcessingPipeline import DataPipe, DataPipeline

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
```
## Available steps

| Process String | Description                                                 |
|----------------|-------------------------------------------------------------|
| `drop_col`       | Drops a specified column.                                   |
| `append_na_mask` | Adds a binary column indicating missing values.             |
| `num_impute`     | Imputes missing numerical data (e.g., with median).         |
| `cat_impute`     | Imputes missing categorical data (e.g., with mode).         |
| `outliers`       | Handles outliers by clipping and/or adding an outlier mask. |
| `one_hot_encode` | Applies one-hot encoding to categorical columns.            |
| `ordinal_encode` | Applies ordinal encoding to ordered categorical columns.    |
| `target_encode`  | Applies target encoding.                                    |
| `label_encode`   | Applies label encoding (typically for a target variable).   |
| `cat_group`      | Groups multiple categories into new ones using a map.       |
| `num_bin`        | Bins a numerical column into ordered categories.            |

## License

This project is licensed under MIT License