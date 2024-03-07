#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from abc import ABCMeta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from .core import _CumlModel
import pyspark
from pyspark import keyword_only
from pyspark.ml.clustering import KMeansModel as SparkKMeansModel
from pyspark.ml.clustering import _KMeansParams
from pyspark.ml.linalg import Vector
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    Row,
    StringType,
    StructField,
    StructType,
)

from .core import (
    CumlT,
    FitInputType,
    _ConstructFunc,
    _CumlEstimator,
    _CumlModelWithPredictionCol,
    _EvaluateFunc,
    _TransformFunc,
    param_alias,
)
from .metrics import EvalMetricInfo
from .params import HasFeaturesCols, P, _CumlClass, _CumlParams
from .utils import (
    _ArrayOrder,
    _concat_and_free,
    _get_spark_session,
    get_logger,
    java_uid,
)

class DBSCANClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {}
    
    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "‘euclidean’",
            "verbose": False,
            "max_mbytes_per_batch": None,
            "calc_core_sample_indices": True
        }
    
    def _pyspark_class(self) -> Optional[ABCMeta]:
        return None
    
class _DBSCANCumlParams(_CumlParams, HasFeaturesCols):
    def __init__(self) -> None:
        super().__init__()
        # restrict default seed to max value of 32-bit signed integer for cuML
        self._setDefault(seed=hash(type(self).__name__) & 0x07FFFFFFF)

    def getFeaturesCol(self) -> Union[str, List[str]]:  # type: ignore
        """
        Gets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`
        """
        if self.isDefined(self.featuresCols):
            return self.getFeaturesCols()
        elif self.isDefined(self.featuresCol):
            return self.getOrDefault("featuresCol")
        else:
            raise RuntimeError("featuresCol is not set")

    def setFeaturesCol(self: P, value: Union[str, List[str]]) -> P:
        """
        Sets the value of :py:attr:`featuresCol` or :py:attr:`featuresCols`.
        """
        if isinstance(value, str):
            self._set_params(featuresCol=value)
        else:
            self._set_params(featuresCols=value)
        return self

    def setFeaturesCols(self: P, value: List[str]) -> P:
        """
        Sets the value of :py:attr:`featuresCols`. Used when input vectors are stored as multiple feature columns.
        """
        return self._set_params(featuresCols=value)

    def setPredictionCol(self: P, value: str) -> P:
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        self._set_params(predictionCol=value)
        return self
    
class DBSCAN(DBSCANClass, _CumlEstimator, _DBSCANCumlParams):
    @keyword_only
    def __init__(
        self,
        *,
        featuresCol: str = "features",
        predictionCol: str = "prediction",
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        max_mbytes_per_batch: Optional[int] = None,
        calc_core_sample_indices: bool = True,
        verbose: Union[int, bool] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._set_params(**self._input_kwargs)

        max_records_per_batch_str = _get_spark_session().conf.get(
            "spark.sql.execution.arrow.maxRecordsPerBatch", "10000"
        )
        assert max_records_per_batch_str is not None
        self.max_records_per_batch = int(max_records_per_batch_str)
        self.BROADCAST_LIMIT = 8 << 30
    
    def _fit(self, dataset: DataFrame) -> _CumlModel:

        def _chunk_arr(
            arr: np.ndarray, BROADCAST_LIMIT: int = self.BROADCAST_LIMIT
        ) -> List[np.ndarray]:
            """Chunk an array, if oversized, into smaller arrays that can be broadcasted."""
            if arr.nbytes <= BROADCAST_LIMIT:
                return [arr]

            rows_per_chunk = BROADCAST_LIMIT // (arr.nbytes // arr.shape[0])
            num_chunks = (arr.shape[0] + rows_per_chunk - 1) // rows_per_chunk
            chunks = [
                arr[i * rows_per_chunk : (i + 1) * rows_per_chunk]
                for i in range(num_chunks)
            ]

            return chunks
        
        raw_data = np.array(dataset.toPandas())
        
        broadcast_raw_data = [
            spark.sparkContext.broadcast(chunk) for chunk in _chunk_arr(raw_data)
        ]

        spark = _get_spark_session()
        model = DBSCANModel(
            raw_data_=broadcast_raw_data,
            n_cols=len(raw_data[0]),
            dtype=type(raw_data[0][0]).__name__,
        )

        model._num_workers = self.num_workers
        model.eps = self.getOrDefault("eps")
        model.min_samples = self.getOrDefault("min_samples")
        model.metric = self.getOrDefault("metric")
        model.max_mbytes_per_batch = self.getOrDefault("max_mbytes_per_batch")
        model.calc_core_sample_indices = self.getOrDefault("calc_core_sample_indices")
        model.verbose = self.getOrDefault("verbose")

        return model

class DBSCANModel(DBSCANClass, _CumlModelWithPredictionCol, _DBSCANCumlParams):
    def __init__(
        self,
        n_cols: int,
        dtype: str,
    ):
        super(DBSCANModel, self).__init__(
            n_cols=n_cols, dtype=dtype
        )

        self._kmeans_spark_model: Optional[SparkKMeansModel] = None
    
    def _out_schema(self, input_schema: StructType) -> Union[StructType, str]:
        ret_schema = "int"
        return ret_schema

    def _transform_array_order(self) -> _ArrayOrder:
        return "C"

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:

        dtype = self.dtype
        n_cols = self.n_cols
        array_order = self._transform_array_order()

        def _construct_kmeans() -> CumlT:
            from cuml.cluster.dbscan_mg import DBSCANMG as CumlDBSCANMG

            dbscan = CumlDBSCANMG(output_type="cudf", eps=self.eps, )
            from spark_rapids_ml.utils import cudf_to_cuml_array

            dbscan.n_cols = n_cols
            dbscan.dtype = np.dtype(dtype)
            return dbscan

        def _transform_internal(
            dbscan: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.Series:
            res = list(dbscan.fit_predict(df).to_numpy())
            return pd.Series(res)

        return _construct_kmeans, _transform_internal, None
    
    def _transform(self, dataset: DataFrame) -> DataFrame:
        # Broadcast the dataset

        # Create ID dataset

        # MapInPandas with cuML

        # Return
