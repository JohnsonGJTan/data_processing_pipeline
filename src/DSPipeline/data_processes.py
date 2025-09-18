from dataclasses import dataclass, field
from typing import Optional, Literal, cast

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, TargetEncoder, OneHotEncoder

from .data_schema import DataSchema, data_schema_validate

@dataclass
class ProcessOutput:
    '''A dataclass '''
    process: str
    params: dict = field(default_factory = dict)
    data: pd.DataFrame = field(default_factory = pd.DataFrame)
    schema: DataSchema = field(default_factory = DataSchema.empty)

def _drop_col(
        col_name: str, 
        data: Optional[pd.DataFrame] = None, 
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput:

    '''Function to drop column from data/schema if it exists.'''

    params = {
        'col_name': col_name
    }

    output = ProcessOutput(process = 'drop_col', params=params)
    input_schema = data_schema_validate(data=data, input_schema=input_schema)
    output.schema = input_schema.copy()._del_col(col_name)
    if data is not None:
        output.data = data.drop(col_name, axis=1, errors='ignore')
    
    return output

def _append_na_mask(
        col_name: str, 
        data: Optional[pd.DataFrame] = None, 
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput:
    
    '''Function to append mask for missing values of a column'''
    
    params = {
        'col_name': col_name
    }
    
    # Check if input_schema and data are valid
    input_schema = data_schema_validate(data=data, input_schema=input_schema)
    
    # Suffices to just check arguments against schema
    if col_name not in input_schema.columns:
        raise TypeError(f"{col_name} is not in input_schema")
    
    output = ProcessOutput(process='append_na_mask', params=params)
    output.schema = input_schema.copy()._append_num(col_name + '_missing')    

    if data is not None:
        mask = pd.Series(data[col_name].isnull(), name=col_name + '_missing')
        output.data = pd.concat([data, mask], axis=1)

    return output
    
def _impute(
        col_name: str, 
        fill_val, 
        data: Optional[pd.DataFrame] = None, 
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput:
    
    '''Function to impute column with fill_val'''
    
    params = {
        'col_name': col_name,
        'fill_val': fill_val
    }
    
    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    output = ProcessOutput(process='impute', params=params)
    output.schema = input_schema.copy()
    if col_name in output.schema.nominal:
        if not isinstance(fill_val, str):
            raise TypeError("Non-null fill_val must be of type string")
        output.schema.nominal[col_name].add(fill_val)
    elif col_name in input_schema.ordinal and fill_val not in input_schema.ordinal[col_name]:
        if not isinstance(fill_val, str):
            raise TypeError("Non-null fill_val must be of type string")
        output.schema.ordinal[col_name].append(fill_val)
        
    # Modify data and append ref to output if needed 
    if data is not None:
        data = data.copy()
        # Check for nominal (ordinal and nominal) columns
        if isinstance(data[col_name].dtype, pd.CategoricalDtype) and fill_val not in data[col_name].cat.categories:
            data[col_name] = data[col_name].cat.add_categories([fill_val])
        imputed_col = data[col_name].fillna(fill_val)
        output.data = data.assign(**{col_name: imputed_col})
    
    return output

def _cat_impute(
        col_name: str, 
        method: str = 'mode', 
        fill_val: Optional[str] = None, 
        data: Optional[pd.DataFrame] = None, 
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput:
    
    '''Function to impute a nominal column'''
    
    params = {
        'col_name': col_name,
        'method': method,
        'fill_val': fill_val,
    }
    
    # For non-null fill_value we reroute to _impute
    if fill_val is not None:
        return _impute(data=data, col_name=col_name, fill_val=fill_val, input_schema=input_schema)        
    
    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    if method not in ['mode']:
        raise ValueError(f"{method} is an invalid nominal impute method")
    if col_name not in input_schema.columns:
        raise TypeError(f"{col_name} is not in input_schema")
    if col_name not in list(input_schema.nominal.keys()) + list(input_schema.ordinal.keys()):
        raise TypeError(f"{col_name} is not a nominal column in input_schema")
    output = ProcessOutput(process='cat_impute', params=params)
    output.schema = input_schema.copy() # If no fill_val given then we should not be able to create new categories

    # Modify data and append ref to output if needed 
    if data is not None:
        # Will implement more methods in future
        if method == 'mode':
            fill_val = data[col_name].mode()[0]
        imputed_col = data[col_name].fillna(fill_val)
        output.data = data.assign(**{col_name: imputed_col})
        output.params['fill_val'] = fill_val
    
    return output

def _num_impute(
        col_name: str, 
        method: str = 'median', 
        fill_val = None, 
        data: Optional[pd.DataFrame] = None,
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput:
    
    '''Function to impute continuous column'''

    params = {
        'col_name': col_name,
        'method': method,
        'fill_val': fill_val
    }

    # For non-null fill_val reroute to _impute
    if fill_val is not None:
        if not pd.api.types.is_numeric_dtype(fill_val):
            raise TypeError("Non-null fill_val must be of numeric dtype")
        return _impute(data=data, col_name=col_name, fill_val=fill_val, input_schema=input_schema)

    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    if method not in ['median']:
        raise ValueError(f"{method} is an invalid continuous impute method")

    output = ProcessOutput(process='num_impute', params=params)
    if input_schema is not None:
        if col_name not in input_schema.continuous:
            raise ValueError(f"{col_name} is not a continuous column in input_schema")
        output.schema = input_schema.copy()

    # Modify data and append ref to output if needed 
    if data is not None:
        if method == 'median':
            fill_val = data[col_name].median()
        imputed_col = data[col_name].fillna(fill_val)
        output.data = data.assign(**{col_name: imputed_col})            
        output.params['fill_val'] = fill_val

    return output

def _outliers(
    col_name: str, 
    outlier_levels: list[int], 
    impute_method: str = 'none',
    append_mask: bool = True,
    bounds: list[tuple[float, float]] = [],
    data: Optional[pd.DataFrame] = None, 
    input_schema: Optional[DataSchema] = None,
    ) -> ProcessOutput:
    
    '''Function to append outlier level and impute sufficiently large outliers'''

    params = {
        'col_name': col_name,
        'outlier_levels': outlier_levels,
        'impute_method': impute_method,
        'append_mask': append_mask,
        'bounds': bounds
    }

    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    if not impute_method in ['clip', 'none']:
        raise ValueError(f"{impute_method} is an invalid impute method")
    if not outlier_levels == sorted(outlier_levels):
        raise ValueError("outlier_levels are not sorted in increasing order")
    if not outlier_levels[0] > 0:
        raise ValueError("outlier_levels contain non-positive values")
    if bounds:
        if not isinstance(bounds, list):
            raise TypeError("bounds must be of type list[tuple[float, float]]")
        if not len(bounds) == len(outlier_levels):
                raise ValueError("bounds and outlier_levels need to be of equal length")
        for bound in bounds:
            if not isinstance(bound, tuple):
                raise TypeError("bounds must be of type list[tuple[float, float]]")
            if len(bound) != 2:
                raise TypeError("bounds must be of type list[tuple[float, float]]")
            if not isinstance(bound[0], float) or not isinstance(bound[1], float):
                raise TypeError("bounds must be of type list[tuple[float, float]]")
        if not bounds[0][0] < bounds[0][1]:
            raise ValueError("invalid bounds given")
        for i in range(1, len(bounds)):
            if not bounds[i][0] < bounds[i-1][0]:
                raise ValueError("invalid bounds given")
            if not bounds[i][1] > bounds[i-1][1]:
                raise ValueError("invalid bounds given")

    output = ProcessOutput(process='outliers', params=params)

    if col_name not in input_schema.columns:
        raise ValueError(f"{col_name} is not a column in the schema")
    if col_name not in input_schema.continuous:
        raise ValueError(f"{col_name} is not a continuous column in input_schema")
    output_schema = input_schema.copy()
    if append_mask:
        output_schema = output_schema._append_ord(col_name + '_outlier', ordinal_categories = outlier_levels)
    output.schema = output_schema
            
    # Modify data and append ref to output if needed 
    if data is not None:
        if data[col_name].isnull().any():
            raise ValueError(f"{col_name} contains null entries")
        data = data.copy()
        if bounds:
            outlier_bounds = bounds
        else:
            # Generate outlier bounds
            mean, std = data[col_name].mean(), data[col_name].std()
            outlier_bounds = [(mean - std*sigma, mean + std*sigma) for sigma in outlier_levels] 
        lower_bound, upper_bound = outlier_bounds[-1]

        if impute_method == 'clip':
            impute_col = data[col_name].clip(lower_bound, upper_bound)
        else:
            impute_col = data[col_name]
    
        output.data = data.assign(**{col_name: impute_col})
    
        if append_mask:
            conditions = [
                (data[col_name] <= bound[0]) | (data[col_name] >= bound[1])
                for bound in reversed(outlier_bounds)
            ]
            choices = list(reversed(outlier_levels))
            mask_values = np.select(conditions, choices, default=0)
            mask = pd.Series(data=mask_values, name = col_name + '_outlier', index=data.index)
            output.data = pd.concat([output.data, mask], axis=1)
            output.params['bounds'] = outlier_bounds 
    
    return output

def _ordinal_encode(
        col_names: list[str], 
        orders: Optional[list[list]] = None, 
        handle_unknown: Literal['error', 'use_encoded_value'] = 'error',
        unknown_value: Optional[int] = None,
        encoder: Optional[OrdinalEncoder] = None , 
        data: Optional[pd.DataFrame] = None, 
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput:

    '''Function to encode ordinal nominal columns as numeric'''

    params = {
        'col_names': col_names,
        'orders': orders,
        'handle_unknown': handle_unknown,
        'unknown_value': unknown_value,
        'encoder': encoder,
    }

    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    if orders is None and encoder is None:
        raise Exception("At least one of orders or encoder must be non-null")

    if orders is not None:
        if not isinstance(orders, list):
            raise TypeError("orders must be of type list[list]")
        if len(orders) != len(col_names):
            raise ValueError("orders and col_names must have equal length")
        for order in orders:
            if not isinstance(order, list):
                raise TypeError("orders must be of type list[list]")
            if len(order) != len(set(order)):
                raise ValueError("orders contain lists with duplicates")
    
    if encoder is not None:
        if not isinstance(encoder, OrdinalEncoder): raise TypeError("Non-null encoder must be of type OrdinalEncoder")
        # Check that columns of encoder matches col_names
        if set(encoder.feature_names_in_) != set(col_names):
            raise ValueError("col_names does not match required columns for encoder")
        getattr_encoder = getattr(encoder, '_sklearn_output_config', None)
        if getattr_encoder is None or not isinstance(getattr_encoder, dict) or 'transform' not in getattr_encoder.keys() or getattr_encoder['transform'] != 'pandas':
            error = "Non-null one-hot encoder output must be set to pandas."
            raise ValueError(error)
        #if getattr(encoder, 'output_transform', None) != 'pandas':
        #    raise ValueError("Non-null ordinal encoder output must be set to pandas")

    output = ProcessOutput(process='ordinal_encode', params=params)

    if not (set(col_names) <= set(input_schema.columns)):
        raise ValueError("col_names contains columns not in input_schema") 
    if not (set(col_names) <= set(input_schema.ordinal.keys())):
        raise ValueError("col_names contains columns which are not nominal (ordinal)")
    output.schema = input_schema.copy()
    # What columns are removed?
    for col_name in col_names:
        if not col_name in output.schema.ordinal:
            raise ValueError(f"{col_name} is not an ordinal column")
        output.schema = output.schema._del_col(col_name)._append_num(col_name)

    # Modify data and append ref to output if needed 
    if data is not None:

        if encoder is not None:
            ordinal_encoder = encoder
        else:
            assert isinstance(orders, list) # To make Pylance not give warning
            ordinal_encoder = OrdinalEncoder(categories=orders, handle_unknown=handle_unknown, unknown_value=unknown_value)
            ordinal_encoder.set_output(transform='pandas')
            ordinal_encoder.fit(data[col_names])

        encoded = cast(pd.DataFrame, ordinal_encoder.transform(data[col_names]))
        output.data = pd.concat([data.drop(col_names, axis=1), encoded], axis=1)
        output.params['encoder'] = ordinal_encoder

    return output
    
def _one_hot_encode(
        col_names: list[str], 
        handle_unknown: Literal['error', 'ignore'] = 'error', 
        encoder: Optional[OneHotEncoder] = None, 
        data: Optional[pd.DataFrame] = None,
        input_schema: Optional[DataSchema] = None, 
        ) -> ProcessOutput:
    
    '''Function to encode nominal encoding with dummy features'''
    
    params = {
        'col_names': col_names,
        'handle_unknown': handle_unknown,
        'encoder': encoder
    }
    
    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    if not (set(col_names) <= set(input_schema.columns)):
        raise ValueError("col_names contains columns not in input_schema") 
    if not (set(col_names) <= set(input_schema.ordinal.keys())) and not (set(col_names) <= set(input_schema.nominal.keys())):
        raise ValueError("col_names contains columns which are not nominal")

    output = ProcessOutput(process='one_hot_encode', params=params)
    # validate encoder
    if encoder is not None:
        if not isinstance(encoder, OneHotEncoder):
            raise TypeError("Non-null encoder must be of type OneHotEncoder")
        if set(encoder.feature_names_in_) != set(col_names):
            raise ValueError("col_names does not match the required columns for encoder")
        
        getattr_encoder = getattr(encoder, '_sklearn_output_config', None)
        if getattr_encoder is None or not isinstance(getattr_encoder, dict) or 'transform' not in getattr_encoder.keys() or getattr_encoder['transform'] != 'pandas':
            error = "Non-null one-hot encoder output must be set to pandas."
            raise ValueError(error)

    output.schema = input_schema.copy()

    for col_name in col_names:
        if col_name not in output.schema.ordinal and col_name not in output.schema.nominal:
            raise ValueError(f"{col_name} is not a valid nominal column")
        # append the new encoded columns
        if col_name in output.schema.ordinal:
            categories = output.schema.ordinal[col_name]
        else:
            categories = output.schema.nominal[col_name]
        for category in categories:
            output.schema = output.schema._append_num(col_name + "_" + category)
        output.schema = output.schema._del_col(col_name)

    # Modify data and append ref to output if needed 
    if data is not None:
        if encoder is not None:
            one_hot_encoder = encoder
        else:
            one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown).fit(data[col_names])
            one_hot_encoder.set_output(transform='pandas')
        
        encoded = cast(pd.DataFrame, one_hot_encoder.transform(data[col_names]))
        output.data = pd.concat([data.drop(col_names, axis=1), encoded], axis=1)
        output.params['encoder'] = one_hot_encoder

    return output

def _target_encode(
        col_names: list[str], 
        target: str, 
        encoder: Optional[TargetEncoder] = None, 
        data: Optional[pd.DataFrame] = None, 
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput:
    
    '''Function to target encode nominal columns'''
    
    params = {
        'col_names': col_names,
        'target': target,
        'encoder': encoder,
    }
    
    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    if target in col_names :
        raise ValueError(f"{target} is in col_names")
    if target not in list(input_schema.ordinal.keys()) + list(input_schema.nominal.keys()):
        raise ValueError(f"{target} is not a nominal column")
    
    if encoder is not None:
        if not isinstance(encoder, TargetEncoder):
            raise TypeError("Non-null encoder must be of type TargetEncoder")
        if set(encoder.feature_names_in_) != set(col_names):
            raise ValueError("col_names does not match the required columns for encoder")
        getattr_encoder = getattr(encoder, '_sklearn_output_config', None)
        if getattr_encoder is None or not isinstance(getattr_encoder, dict) or 'transform' not in getattr_encoder.keys() or getattr_encoder['transform'] != 'pandas':
            error = "Non-null one-hot encoder output must be set to pandas."
            raise ValueError(error)
        #if getattr(encoder, 'output_transform', None) != 'pandas':
        #    raise ValueError("Non-null target encoder output must be set to pandas")

    output = ProcessOutput(process='target_encode', params=params)
    if not (set(col_names) <= set(input_schema.columns)):
        raise ValueError("col_names contains columns not in input_schema")
    output.schema = input_schema.copy()

    for col_name in col_names:
        if col_name not in output.schema.ordinal and col_name not in output.schema.nominal:
            raise ValueError(f"{col_name} is not a nominal column in the schema")
        output.schema = output.schema._del_col(col_name)._append_num(col_name)
        
    # Modify data and append ref to output if needed 
    if data is not None:
        if encoder is not None:
            target_encoder = encoder
        else:
            target_encoder = TargetEncoder(random_state=42)
            target_encoder.set_output(transform='pandas')
            target_encoder.fit(data[col_names], data[target])
        encoded = cast(pd.DataFrame, target_encoder.transform(data[col_names]))
        output.data = pd.concat([data.drop(col_names, axis=1), encoded], axis=1)
        output.params['encoder'] = target_encoder

    return output

def _label_encode(
        col_name: str, 
        encoder: Optional[LabelEncoder] = None, 
        data: Optional[pd.DataFrame] = None, 
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput: 
    
    '''Function to label encode nominal columns'''

    params = {
        'col_name': col_name,
        'encoder': encoder
    }

    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    if col_name not in input_schema.columns:
        raise ValueError(f"{col_name} is not a column")
    if col_name in input_schema.continuous:
        raise ValueError(f"{col_name} is not a nominal column")
    # Modify data and append ref to output if needed 
    output = ProcessOutput(process='label_encode', params=params)
    output.schema = input_schema.copy()._del_col(col_name)._append_num(col_name)

    if data is not None:
        if encoder is not None:
            label_encoder = encoder
        else:
            label_encoder = LabelEncoder().fit(data[col_name])
        encoded = cast(np.ndarray, label_encoder.transform(data[col_name]))
        output.data = data.assign(**{col_name: encoded.tolist()})
        output.params['encoder'] = label_encoder

    return output

def _cat_group(
        col_name: str, 
        map: dict, 
        data: Optional[pd.DataFrame] = None, 
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput:
    
    '''Function to group and transform categories in a nominal column'''
    
    params = {
        'col_name': col_name,
        'map': map,
    }
    
    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    output = ProcessOutput(process='cat_group', params=params)
    map_categories = list(set(map.values()))

    if col_name not in input_schema.columns:
        raise ValueError(f"{col_name} is not a column in input_schema")
    if col_name in input_schema.continuous:
        raise ValueError(f"{col_name} is not a nominal column in input_schema")
    output.schema = input_schema.copy()._del_col(col_name)._append_unord(col_name, map_categories)

    # Modify data and append ref to output if needed 
    if data is not None:
        col_grouped = pd.Series(pd.Categorical(
            values= data[col_name].map(map),
            #values=[map[val] for val in data[col_name]],
            categories = map_categories,
            ordered=False
        ))
        output.data = data.assign(**{col_name: col_grouped})
    
    return output

def _num_bin(
        col_name: str, 
        bins: list[float], 
        data: Optional[pd.DataFrame] = None, 
        input_schema: Optional[DataSchema] = None,
        ) -> ProcessOutput:
  
    '''Function to discretize/bin continuous columns'''
    
    params = {
        'col_name': col_name,
        'bins': bins,
    }
  
    # Validate schema and data are correct type and compatible
    input_schema = data_schema_validate(data=data, input_schema=input_schema)

    # Validate process arguments against schema and modify it
    if not isinstance(bins, list):
        raise TypeError("bins must be of type list[float]")
    for cut in bins:
        if not isinstance(cut, float):
            raise TypeError("bins must be of type list[float]")
    if bins != sorted(bins):
        raise ValueError("bins is required to be sorted in ascending order")

    output = ProcessOutput(process='num_bin', params=params)
    
    # Generate categories for schema
    bin_categories = [f'(-inf, {bins[0]}]']
    for i in range(len(bins)-1):
        bin_categories.append(f"({bins[i]}, {bins[i+1]}]")
    bin_categories.append(f'({bins[-1]}, inf)')

    if col_name not in input_schema.columns:
        raise ValueError(f"{col_name} is not a column in input_schema")
    if col_name not in input_schema.continuous:
        raise ValueError(f"{col_name} is not a continuous column in input_schema")
    output.schema = input_schema.copy()._del_col(col_name)._append_ord(col_name, bin_categories)

    # Modify data and append ref to output if needed 
    if data is not None:
        cut_bins = [-np.inf] + bins + [np.inf]
        binned_col = pd.cut(data[col_name], bins = cut_bins, labels = bin_categories, ordered=True)
        output.data = data.assign(**{col_name: binned_col})

    return output