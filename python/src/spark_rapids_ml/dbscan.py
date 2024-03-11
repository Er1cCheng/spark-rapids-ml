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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast, Iterator

import numpy as np
import pandas as pd
from .core import _CumlModel, Pred, pred
import pyspark
from pyspark import keyword_only
from pyspark.ml.linalg import Vector
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lit, monotonically_increasing_id, row_number
from pyspark.sql.window import Window
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql.pandas.functions import pandas_udf
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
    _CumlCaller,
    _CumlEstimator,
    _CumlModelWithPredictionCol,
    _EvaluateFunc,
    _TransformFunc,
    param_alias,
)
from pyspark.ml.param.shared import (
    HasFeaturesCol,
    Param,
    Params,
    TypeConverters,
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
    
class _DBSCANCumlParams(_CumlParams, HasFeaturesCol, HasFeaturesCols):
    def __init__(self) -> None:
        super().__init__()
        self._setDefault(
            eps=0.5,
            min_samples=5,
            metric="euclidean",
            max_mbytes_per_batch=None,
            calc_core_sample_indices=True,
        )

    eps = Param(
        Params._dummy(),
        "eps",
        (
            f"The maximum distance between 2 points such they reside in the same neighborhood."
        ),
        typeConverter=TypeConverters.toFloat,
    )

    min_samples = Param(
        Params._dummy(),
        "min_samples",
        (
            f"The number of samples in a neighborhood such that this group can be considered as an important core point (including the point itself)."
        ),
        typeConverter=TypeConverters.toInt,
    )

    metric = Param(
        Params._dummy(),
        "metric",
        (
            f"The metric to use when calculating distances between points."
            f"If metric is ‘precomputed’, X is assumed to be a distance matrix and must be square."
            f"The input will be modified temporarily when cosine distance is used and the restored input matrix might not match completely due to numerical rounding."
        ),
        typeConverter=TypeConverters.toString,
    )

    max_mbytes_per_batch = Param(
        Params._dummy(),
        "max_mbytes_per_batch",
        (
            f"Calculate batch size using no more than this number of megabytes for the pairwise distance computation."
            f"This enables the trade-off between runtime and memory usage for making the N^2 pairwise distance computations more tractable for large numbers of samples."
            f"If you are experiencing out of memory errors when running DBSCAN, you can set this value based on the memory size of your device."
        ),
        typeConverter=TypeConverters.toInt,
    )

    calc_core_sample_indices = Param(
        Params._dummy(),
        "calc_core_sample_indices",
        (
            f"Indicates whether the indices of the core samples should be calculated."
            f"Setting this to False will avoid unnecessary kernel launches"
        ),
        typeConverter=TypeConverters.toBoolean,
    )

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

        self.verbose = verbose
    
    def setEps(self, value : float):
        return self._set_params(eps=value)
    
    def getEps(self):
        return self.getOrDefault("eps")
    
    def _fit(self, dataset: DataFrame) -> _CumlModel:
        spark = _get_spark_session()

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

        model = DBSCANModel(
            raw_data_=broadcast_raw_data,
            n_cols=len(raw_data[0]),
            dtype=type(raw_data[0][0][0]).__name__,
        )

        model._num_workers = self.num_workers
        model.eps = self.getOrDefault("eps")
        model.min_samples = self.getOrDefault("min_samples")
        model.metric = self.getOrDefault("metric")
        model.max_mbytes_per_batch = self.getOrDefault("max_mbytes_per_batch")
        model.calc_core_sample_indices = self.getOrDefault("calc_core_sample_indices")
        model.verbose = self.verbose

        model.input_schema = dataset.schema

        return model
    
    def _create_pyspark_model(self, result: Row) -> _CumlModel:
        raise NotImplementedError("DBSCAN does not support model creation from Row")
    
    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        raise NotImplementedError("DBSCAN does not can not fit and generate model")
    
    def _out_schema(self) -> Union[StructType, str]:
        return StructType()

class DBSCANModel(DBSCANClass, _CumlCaller, _CumlModelWithPredictionCol, _DBSCANCumlParams):
    def __init__(
        self,
        n_cols: int,
        dtype: str,
        raw_data_: List[pyspark.broadcast.Broadcast],
    ):
        super(DBSCANClass, self).__init__()

        super(_CumlModelWithPredictionCol, self).__init__(
            n_cols=n_cols, dtype=dtype, raw_data_=raw_data_,
        )

        super(_DBSCANCumlParams, self).__init__()

        self._dbscan_spark_model = None
    
    def _out_schema(self, *args) -> Union[StructType, str]:
        return StructType(
            [
                StructField(self._get_prediction_name(), IntegerType(), False),
            ]
        )
    
        # return self.input_schema.add(self._get_prediction_name(), IntegerType(), False)

    def _transform_array_order(self) -> _ArrayOrder:
        return "C"
    
    def _fit_array_order(self) -> _ArrayOrder:
        return "C"
    
    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        dtype = self.dtype
        n_cols = self.n_cols
        array_order = self._fit_array_order()

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.cluster.dbscan_mg import DBSCANMG as CumlDBSCANMG

            dbscan = CumlDBSCANMG(handle=params[param_alias.handle], 
                                  output_type="cudf", 
                                  eps=self.eps, 
                                  min_samples=self.min_samples,
                                  metric=self.metric,
                                  max_mbytes_per_batch=self.max_mbytes_per_batch,
                                  calc_core_sample_indices=self.calc_core_sample_indices)
            dbscan.n_cols = n_cols
            dbscan.dtype = np.dtype(dtype)
            
            df_list = [x for (x, _, _) in dfs]
            if isinstance(df_list[0], pd.DataFrame):
                concated = pd.concat(df_list)
            else:
                # features are either cp or np arrays here
                concated = _concat_and_free(df_list, order=array_order)
            data_df = concated[:, :-1]

            res = list(dbscan.fit_predict(data_df).to_numpy())

            return pd.Series(res)
        
        return _cuml_fit

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

        def _construct_dbscan() -> CumlT:
            from cuml.cluster.dbscan_mg import DBSCANMG as CumlDBSCANMG

            dbscan = CumlDBSCANMG(output_type="cudf", eps=self.eps, )
            from spark_rapids_ml.utils import cudf_to_cuml_array

            dbscan.n_cols = n_cols
            dbscan.dtype = np.dtype(dtype)
            return dbscan

        def _transform_internal(
            dbscan: CumlT, df: Union[pd.DataFrame, np.ndarray]
        ) -> pd.Series:
            # Delete the worker id column used for partition
            data_df = df[:, :-1]

            res = list(dbscan.fit_predict(data_df).to_numpy())
            return pd.Series(res)

        return _construct_dbscan, _transform_internal, None
    
    def _transform(self, dataset: DataFrame) -> DataFrame:
        dataset_copies = [dataset.alias(f"dataset_copy_{i}").withColumn("worker_id", lit(i)) for i in range(self.num_workers)]

        dataset_concat = dataset_copies[0]
        for df in dataset_copies[1:]:
            dataset_concat = dataset_concat.union(df)

        dataset_concat.repartition(self.num_workers, "worker_id")

        # Return
        rdd = self._call_cuml_fit_func(
            dataset=dataset_concat,
            partially_collect=False,
            paramMaps=None,
        )

        pred_name = self._get_prediction_name()
        pred_df = rdd.toDF()
        return pred_df
        window_spec = Window.orderBy(lit(1))
        pred_df = pred_df.withColumn("index", row_number().over(window_spec))
        dataset = dataset.withColumn("index", row_number().over(window_spec))

        # dataset.show()
        # pred_df.show()

        return pred_df

        return dataset.join(pred_df, "index").drop("index")
