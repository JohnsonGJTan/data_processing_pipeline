from typing import Self, Optional
import copy

import pandas as pd

from ..utils import is_sub_with_gap

class DataSchema:

    def __init__(self, continuous: set[str], nominal: dict[str, set], ordinal: dict[str, list]):
        
        if not isinstance(continuous, set): raise TypeError("continuous must be of type set[str]")
        for col_name in continuous:
            if not isinstance(col_name, str): raise TypeError("continuous must be of type set[str]")
        
        if not isinstance(nominal, dict): raise TypeError("nominal must be of type dict[str, set]")
        for key, value in nominal.items():
            if not isinstance(key, str): raise TypeError("nominal must be of type dict[str, set]")  
            if not isinstance(value, set): raise TypeError("nominal must be of type dict[str, set]")

        if not isinstance(ordinal, dict): raise TypeError("ordinal must be of type dict[set, list]")
        for key, value in ordinal.items():
            if not isinstance(key, str): raise TypeError("ordinal must be of type dict[str, list]") 
            if not isinstance(value, list): raise TypeError("ordinal must be of type dict[str, list]")
            if len(value) != len(set(value)):
                raise ValueError(f"ordinal nominal column '{key}' contains duplicate categories")
        
        # Check that continuous, nominal, and ordinal have different columns
        continuous_cols = continuous
        nominal_cols = set(nominal.keys())
        ordinal_cols = set(ordinal.keys())
        if len(set().union(continuous_cols, nominal_cols, ordinal_cols)) != len(continuous_cols) + len(nominal_cols) + len(ordinal_cols):
            raise ValueError("continuous, nominal, and ordinal contain duplicate columns")
        
        self.continuous = continuous
        self.nominal = nominal 
        self.ordinal = ordinal

    def __eq__(self, other) -> bool:
       
        # Check if they have the same type
        if not isinstance(other, self.__class__):
            return NotImplemented #TypeError(f"'==' not supported between instances of 'DataSchema' and {type(other)}")
        
        return (self.continuous == other.continuous and
                self.nominal == other.nominal and
                self.ordinal == other.ordinal)

    def __le__(self, other: Self) -> bool:
        
        if not isinstance(other, self.__class__):
            return NotImplemented #TypeError(f"'<=' not supported between instances of 'DataSchema' and {type(other)}")
        
        # Check if they have the same columns
        if self.continuous != other.continuous or set(self.nominal.keys()) != set(other.nominal.keys()) or set(self.ordinal.keys()) != set(other.ordinal.keys()):
            return False
        
        # Check that the nominal categories of self is not a subset of other
        for col_name in self.nominal.keys():
            if not (self.nominal[col_name] <= other.nominal[col_name]):
                return False

        for col_name in self.ordinal.keys():
            if not is_sub_with_gap(self.ordinal[col_name], other.ordinal[col_name]):
                return False

        return True

    def __ge__(self, other: Self) -> bool:
        
        if not isinstance(other, self.__class__):
            return NotImplemented #TypeError(f"'>=' not supported between instances of 'DataSchema' and {type(other)}")

        return other.__le__(self)

    def __lt__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented #TypeError(f"'<' not supported between instances of 'DataSchema' and {type(other)}")
        return self <= other and self != other
    
    def __gt__(self, other: Self) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented #TypeError(f"'>' not supported between instances of 'DataSchema' and {type(other)}")
        return self >= other and self != other

    def __str__(self):
        num = f"continuous: {self.continuous}"
        unord = f"nominal: {self.nominal}"
        ord_ = f"ordinal: {self.ordinal}"
        return f"DataSchema(\n  {num},\n  {unord},\n  {ord_}\n)"
    
    def __repr__(self):
        return self.__str__()

    @property
    def columns(self) -> list[str]:
        return list(self.continuous) + list(self.nominal.keys()) + list(self.ordinal.keys())

    @classmethod
    def build(cls, data: pd.DataFrame, detect_col_type: bool = True) -> Self:
        
        if not isinstance(data, pd.DataFrame):
            return TypeError(f"build not supported for instances of {type(data)}")

        if detect_col_type:
            for col_name in data.columns:
                pass

        continuous = set()
        nominal = {}
        ordinal = {}

        for col in data.columns:
            if isinstance(data[col].dtype, pd.CategoricalDtype):
                if data[col].cat.ordered:
                    ordinal[col] = data[col].cat.categories.to_list()
                else:
                    nominal[col] = set(data[col].cat.categories.to_list())
            elif pd.api.types.is_numeric_dtype(data[col]):
                continuous.add(col)
        
        return cls(continuous, nominal, ordinal)

    def copy(self) -> Self:

        return self.__class__(
            continuous = copy.deepcopy(self.continuous),
            ordinal = copy.deepcopy(self.ordinal),
            nominal = copy.deepcopy(self.nominal)
        )
    
    @classmethod
    def empty(cls) -> Self:
        return cls(continuous=set(), nominal = dict(), ordinal=dict())
    
    def _append_num(self, col_name: str) -> Self:
        
        if col_name in self.columns:
            raise ValueError(f"{col_name} is already in schema")
        
        schema_copy = self.copy()
        schema_copy.continuous.add(col_name)

        return schema_copy

    def _append_ord(self, col_name: str, ordinal_categories: list) -> Self:
        
        schema_copy = self.copy()
        
        if not len(set(ordinal_categories)) == len(ordinal_categories):
            raise ValueError("ordinal_categories contains duplicate categories")
        if col_name in schema_copy.columns:
            raise ValueError(f"{col_name} is already in schema")

        schema_copy.ordinal[col_name] = copy.deepcopy(ordinal_categories)

        return schema_copy

    def _append_unord(self, col_name: str, categories: list[str]) -> Self:
        
        schema_copy = self.copy()
        if not len(set(categories)) == len(categories):
            raise ValueError("categories contains duplicate categories")
        if col_name in schema_copy.columns:
            raise ValueError(f"{col_name} is already in schema")

        schema_copy.nominal[col_name] = set(categories)

        return schema_copy

    def _del_col(self, col_name: str) -> Self:
        
        schema_copy = self.copy()
        
        if col_name in self.continuous:
            schema_copy.continuous.remove(col_name)
        elif col_name in self.ordinal.keys():
            del schema_copy.ordinal[col_name]
        elif col_name in self.nominal.keys():
            del schema_copy.nominal[col_name]
        
        return schema_copy




# TODO: Make this more efficient?
def data_schema_validate(data: Optional[pd.DataFrame] = None, input_schema: Optional[DataSchema] = None) -> DataSchema:
    '''Function to validate that data and input_schema are of the correct type and are compatible'''

    if data is None and input_schema is None:
        raise TypeError("At least one of data or input_schema has not be non-null")
    elif data is not None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Non-null data must be of type pd.DataFrame")
        if input_schema is None:
            input_schema = DataSchema.build(data)
        else:
            if not isinstance(input_schema, DataSchema):
                raise TypeError("Non-null input_schema must be of type DataSchema")
            if input_schema != DataSchema.build(data):
                raise ValueError("input_schema and data are not compatible")
    else:
        if not isinstance(input_schema, DataSchema):
            raise TypeError("Non-null input_schema must be of type DataSchema")

    return input_schema 
