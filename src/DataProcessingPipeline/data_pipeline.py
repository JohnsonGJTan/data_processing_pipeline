import inspect
from typing import Self, Optional
from functools import partial
import pathlib
import joblib
import json

import pandas as pd

from . import data_processes
from .data_processes import ProcessOutput
from .data_schema import DataSchema, data_schema_validate


class DataPipe:

    PROCESS_REGISTRY = {
        'drop_col': data_processes._drop_col,
        'append_na': data_processes._append_na_mask,
        'impute': data_processes._impute,
        'cat_impute': data_processes._cat_impute,
        'num_impute': data_processes._num_impute,
        'outliers': data_processes._outliers,
        'ordinal_encode': data_processes._ordinal_encode,
        'one_hot_encode': data_processes._one_hot_encode,
        'target_encode': data_processes._target_encode,
        'label_encode': data_processes._label_encode,
        'cat_group': data_processes._cat_group,
        'num_bin': data_processes._num_bin
    }

    def __init__(self, process_str: str, params: dict):
        
        if process_str not in self.PROCESS_REGISTRY:
            raise ValueError(f"{process_str} not in process registry")
        if not isinstance(params, dict):
            raise TypeError("params must be of type dict")

        self.process_str = process_str
        self.params = params

        process_func = self.PROCESS_REGISTRY[process_str]

        sig = inspect.signature(process_func)
        for param_name in params.keys():
            if param_name not in sig.parameters:
                raise TypeError(f"'{param_name}' is not a valid parameter for {process_str} process")

        self.process = partial(process_func, **self.params)

    def __call__(self, data: Optional[pd.DataFrame] = None, schema: Optional[DataSchema] = None) -> ProcessOutput:
        '''Method for directly applying DataPipe to data and/or schema'''
        # Validates the types and compatibility between 
        input_schema = data_schema_validate(data=data, input_schema=schema)

        return self.process(input_schema=input_schema, data=data)


class DataPipeline:

    def __init__(
            self, 
            pipeline: list[DataPipe],
            pipeline_path: Optional[str | pathlib.Path] = None
            ):

        if pipeline_path is not None:
            if isinstance(pipeline_path, str):
                pipeline_path = pathlib.Path(pipeline_path)
            elif not isinstance(pipeline_path, pathlib.Path):
                raise TypeError("Non-null pipeline_path must be of type str or pathlib.Path")
            
            if not pipeline_path.is_file():
                raise ValueError("pipeline_path is not a file")
            
            with pipeline_path.open('r') as f:
                raw_pipeline = json.load(f)

            pipeline = []
            for raw_pipe in raw_pipeline:
                pipeline.append(DataPipe(process_str = raw_pipe['process'], params = raw_pipe['init_params']))
        else:
            if not isinstance(pipeline, list):
                raise TypeError("pipeline must be of type list[DataPipe]")
            for pipe in pipeline:
                if not isinstance(pipe, DataPipe):
                    raise TypeError("pipeline must be of type list[DataPipe]")
        
        self.pipeline = pipeline
        self.pipeline_ = []
        self.input_schema_ = None
        self.output_schema_ = None
        self._is_fitted = False 
                
    def fit(
        self, 
        ref_data: pd.DataFrame, 
        input_schema: Optional[DataSchema] = None, 
        ) -> Self:
        
        # validate input_schema and ref_data
        input_schema = data_schema_validate(data=ref_data, input_schema=input_schema)

        ## validate pipeline against input_schema and data
        data=ref_data.copy()
        pipeline_schema = input_schema.copy()
        fitted_pipeline = []
        for pipe in self.pipeline:
            output = pipe(data=data, schema=pipeline_schema)
            data = output.data.copy()
            pipeline_schema = output.schema.copy()
            fitted_pipeline.append(DataPipe(process_str=output.process, params=output.params))
    
        self.pipeline_ = fitted_pipeline
        self.output_schema_ = pipeline_schema
        self._is_fitted = True
        
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        
        if not self._is_fitted:
            raise RuntimeError("This DataPipeline instance is not yet fitted. Call 'fit' first.")
        
        # Validate that data is compatible with self.input_schema
        data_schema_validate(data=data, input_schema=self.input_schema_)        

        # Apply pipeline to data
        data = data.copy()
        for pipe in self.pipeline_:
            data = pipe(data=data).data

        return data

    def save(self, save_path: str | pathlib.Path):
        
        if not self._is_fitted:
            raise RuntimeError("Only fitted fitted pipelines can be saved.")

        # TODO: need to make this more secure...
        if isinstance(save_path, str):
            save_path = pathlib.Path(save_path)

        with save_path.open('wb') as f:
            joblib.dump(self, f)

    @classmethod
    def load(cls, load_path: str | pathlib.Path):
        if isinstance(load_path, str):
            load_path = pathlib.Path(load_path)
        
        with load_path.open('rb') as f:
            return joblib.load(f)