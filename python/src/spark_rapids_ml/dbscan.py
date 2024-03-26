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

import json
import os
from abc import ABCMeta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import pyspark
from pyspark import RDD, keyword_only
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.linalg import Vector
from pyspark.ml.param.shared import HasFeaturesCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReader, DefaultParamsWriter, MLReader, MLWriter
from pyspark.sql import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col,
    lit,
    monotonically_increasing_id,
    row_number,
    spark_partition_id,
)
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    Row,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window

from .common.cuml_context import CumlContext
from .core import (
    CumlT,
    FitInputType,
    Pred,
    _ConstructFunc,
    _CumlCaller,
    _CumlCommon,
    _CumlEstimator,
    _CumlModel,
    _CumlModelReader,
    _CumlModelWithPredictionCol,
    _CumlModelWriter,
    _EvaluateFunc,
    _read_csr_matrix_from_unwrapped_spark_vec,
    _TransformFunc,
    _use_sparse_in_cuml,
    alias,
    param_alias,
    pred,
)
from .metrics import EvalMetricInfo
from .params import HasFeaturesCols, P, _CumlClass, _CumlParams
from .utils import (
    _ArrayOrder,
    _concat_and_free,
    _get_gpu_id,
    _get_spark_session,
    _is_local,
    _is_standalone_or_localcluster,
    dtype_to_pyspark_type,
    get_logger,
)

if TYPE_CHECKING:
    import cudf
    from pyspark.ml._typing import ParamMap


class DBSCANClass(_CumlClass):
    @classmethod
    def _param_mapping(cls) -> Dict[str, Optional[str]]:
        return {}

    def _get_cuml_params_default(self) -> Dict[str, Any]:
        return {
            "eps": 0.5,
            "min_samples": 5,
            "metric": "euclidean",
            "verbose": False,
            "max_mbytes_per_batch": None,
            "calc_core_sample_indices": True,
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
            idCol=alias.row_number,
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

    idCol = Param(
        Params._dummy(),
        "idCol",
        "id column name.",
        typeConverter=TypeConverters.toString,
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

    def setIdCol(self: P, value: str) -> P:
        """
        Sets the value of `idCol`. If not set, an id column will be added with column name `unique_id`. The id column is used to specify nearest neighbor vectors by associated id value.
        """
        self._set_params(idCol=value)
        return self

    def getIdCol(self) -> str:
        """
        Gets the value of `idCol`.
        """
        return self.getOrDefault("idCol")

    def _ensureIdCol(self, df: DataFrame) -> DataFrame:
        """
        Ensure an id column exists in the input dataframe. Add the column if not exists.
        """
        if not self.isSet("idCol") and self.getIdCol() in df.columns:
            raise ValueError(
                f"Cannot create a default id column since a column with the default name '{self.getIdCol()}' already exists."
                + "Please specify an id column"
            )

        id_col_name = self.getIdCol()
        df_withid = (
            df
            if self.isSet("idCol")
            else df.select(monotonically_increasing_id().alias(id_col_name), "*")
        )
        return df_withid


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

    def setEps(self: P, value: float) -> P:
        return self._set_params(eps=value)

    def getEps(self) -> float:
        return self.getOrDefault("eps")

    def setMin_samples(self: P, value: int) -> P:
        return self._set_params(min_samples=value)

    def getMin_samples(self) -> int:
        return self.getOrDefault("min_samples")

    def setMetric(self: P, value: str) -> P:
        return self._set_params(metric=value)

    def getMetric(self) -> str:
        return self.getOrDefault("metric")

    def _pre_process_data(self, dataset: DataFrame) -> Tuple[
        List[Column],
        Optional[List[str]],
        int,
        Union[Type[FloatType], Type[DoubleType]],
    ]:
        (
            select_cols,
            multi_col_names,
            dimension,
            feature_type,
        ) = _CumlCaller._pre_process_data(self, dataset)

        if self.hasParam("idCol") and self.isDefined("idCol"):
            id_col_name = self.getOrDefault("idCol")
            select_cols.append(col(id_col_name).alias(alias.row_number))
        else:
            select_cols.append(col(alias.row_number))

        return select_cols, multi_col_names, dimension, feature_type

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

        dataset = self._ensureIdCol(dataset)
        select_cols, multi_col_names, dimension, _ = self._pre_process_data(dataset)
        use_sparse_array = _use_sparse_in_cuml(dataset)
        input_dataset = dataset.select(*select_cols)
        pd_dataset: pd.DataFrame = input_dataset.toPandas()
        raw_data: np.ndarray = np.array(pd_dataset.drop(columns=[self.getIdCol()]))
        idCols: np.ndarray = np.array(pd_dataset[self.getIdCol()])
        input_col, input_cols = self._get_input_columns()

        broadcast_raw_data = [
            spark.sparkContext.broadcast(chunk) for chunk in _chunk_arr(raw_data)
        ]

        broadcast_idCol = [
            spark.sparkContext.broadcast(chunk) for chunk in _chunk_arr(idCols)
        ]

        get_logger(self.__class__).info(raw_data[0])

        model = DBSCANModel(
            eps = self.getOrDefault("eps"),
            min_samples = self.getOrDefault("min_samples"),
            metric = self.getOrDefault("metric"),
            max_mbytes_per_batch = self.getOrDefault("max_mbytes_per_batch"),
            calc_core_sample_indices = self.getOrDefault("calc_core_sample_indices"),
            raw_data_=broadcast_raw_data,
            idCols_=broadcast_idCol,
            n_cols=len(raw_data[0]),
            dtype=(
                type(raw_data[0][0][0]).__name__
                if isinstance(raw_data[0][0], List)
                or isinstance(raw_data[0][0], np.ndarray)
                else type(raw_data[0][0]).__name__
            ),
            processed_input_cols=input_dataset.drop(self.getIdCol()).columns,
            multi_col_names=multi_col_names,
            use_sparse_array=use_sparse_array,
            verbose=self.verbose,
            input_col=input_col,
            input_cols=input_cols,
        )

        model._num_workers = self.num_workers
        # model.eps = self.getOrDefault("eps")
        # model.min_samples = self.getOrDefault("min_samples")
        # model.metric = self.getOrDefault("metric")
        # model.max_mbytes_per_batch = self.getOrDefault("max_mbytes_per_batch")
        # model.calc_core_sample_indices = self.getOrDefault("calc_core_sample_indices")

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


class DBSCANModel(
    DBSCANClass, _CumlCaller, _CumlModelWithPredictionCol, _DBSCANCumlParams
):
    def __init__(
        self,
        eps: float,
        min_samples: int,
        metric: str,
        max_mbytes_per_batch: Optional[int],
        calc_core_sample_indices: bool,
        n_cols: int,
        dtype: str,
        raw_data_: List[pyspark.broadcast.Broadcast],
        idCols_: List[pyspark.broadcast.Broadcast],
        processed_input_cols: List[str],
        multi_col_names: List[str] | None,
        use_sparse_array: bool,
        verbose: int | bool,
        input_col: Optional[str],
        input_cols: Optional[List[str]],
    ):
        super(DBSCANClass, self).__init__()
        super(_CumlModelWithPredictionCol, self).__init__(
            n_cols=n_cols, dtype=dtype, raw_data_=raw_data_
        )
        super(_DBSCANCumlParams, self).__init__()

        self._setDefault(
            eps=0.5,
            min_samples=5,
            metric="euclidean",
            max_mbytes_per_batch=None,
            calc_core_sample_indices=True,
            idCol=alias.row_number,
        )

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.max_mbytes_per_batch = max_mbytes_per_batch
        self.calc_core_sample_indices = calc_core_sample_indices

        self._dbscan_spark_model = None
        # self.output_schema = StructType(output_schema.fields[:])
        self.processed_input_cols = processed_input_cols
        self.raw_data_ = raw_data_
        self.idCols_ = idCols_
        self.multi_col_names = multi_col_names
        self.use_sparse_array = use_sparse_array
        self.verbose = verbose

        if input_col is not None:
            self.setFeaturesCol(input_col)

        if input_cols is not None:
            self.setFeaturesCols(input_cols)

    def _pre_process_data(self, dataset: DataFrame) -> Tuple[  # type: ignore
        List[Column],
        Optional[List[str]],
        int,
        Union[Type[FloatType], Type[DoubleType]],
    ]:
        (
            select_cols,
            multi_col_names,
            dimension,
            feature_type,
        ) = _CumlCaller._pre_process_data(self, dataset)

        if self.hasParam("idCol") and self.isDefined("idCol"):
            id_col_name = self.getOrDefault("idCol")
            select_cols.append(col(id_col_name).alias(alias.row_number))
        else:
            select_cols.append(col(alias.row_number))

        return select_cols, multi_col_names, dimension, feature_type

    def _out_schema(
        self, input_schema: StructType = StructType()
    ) -> Union[StructType, str]:
        return StructType(
            [
                StructField(self._get_prediction_name(), IntegerType(), False),
                StructField(self.getIdCol(), LongType(), False),
            ]
        )

    def _transform_array_order(self) -> _ArrayOrder:
        return "C"

    def _fit_array_order(self) -> _ArrayOrder:
        return "C"

    def _require_nccl_ucx(self) -> Tuple[bool, bool]:
        return (True, True)

    def _get_cuml_fit_func(
        self,
        dataset: DataFrame,
        extra_params: Optional[List[Dict[str, Any]]] = None,
    ) -> Callable[
        [FitInputType, Dict[str, Any]],
        Dict[str, Any],
    ]:
        import cupy as cp
        import cupyx

        dtype = self.dtype
        n_cols = self.n_cols
        array_order = self._fit_array_order()
        pred_name = self._get_prediction_name()
        idCol_name = self.getIdCol()
        logger = get_logger(self.__class__)

        cuda_managed_mem_enabled = (
            _get_spark_session().conf.get("spark.rapids.ml.uvm.enabled", "false")
            == "true"
        )

        inputs = []

        idCol = list(
            self.idCols_[0].value
            if len(self.idCols_) == 1
            else np.concatenate([chunk.value for chunk in self.idCols_])
        )

        for pdf_bc in self.raw_data_:
            pdf = pd.DataFrame(data=pdf_bc.value, columns=self.processed_input_cols)

            if self.multi_col_names:
                features = np.array(pdf[self.multi_col_names], order=array_order)
            elif self.use_sparse_array:
                # sparse vector
                features = _read_csr_matrix_from_unwrapped_spark_vec(pdf)
            else:
                # dense vector
                features = np.array(list(pdf[alias.data]), order=array_order)

            # experiments indicate it is faster to convert to numpy array and then to cupy array than directly
            # invoking cupy array on the list
            if cuda_managed_mem_enabled:
                features = (
                    cp.array(features)
                    if self.use_sparse_array is False
                    else cupyx.scipy.sparse.csr_matrix(features)
                )

            inputs.append(features)

        if isinstance(inputs[0], pd.DataFrame):
            concated = pd.concat(inputs)
        else:
            # features are either cp or np arrays here
            concated = _concat_and_free(inputs, order=array_order)

        def _cuml_fit(
            dfs: FitInputType,
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
            from cuml.cluster.dbscan_mg import DBSCANMG as CumlDBSCANMG
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            partition_id = context.partitionId()

            logger = get_logger(self.__class__)

            dbscan = CumlDBSCANMG(
                handle=params[param_alias.handle],
                output_type="cudf",
                eps=self.eps,
                min_samples=self.min_samples,
                metric=self.metric,
                max_mbytes_per_batch=self.max_mbytes_per_batch,
                calc_core_sample_indices=self.calc_core_sample_indices,
                verbose=self.verbose,
            )
            dbscan.n_cols = params[param_alias.num_cols]
            dbscan.dtype = np.dtype(dtype)

            res = list(dbscan.fit_predict(concated).to_numpy())

            if partition_id == 0:
                return {
                    idCol_name: idCol,
                    pred_name: res,
                }
            else:
                return {
                    idCol_name: [],
                    pred_name: [],
                }

        return _cuml_fit

    def fit_post_process(
        self, fit_result: Dict[str, Any], partition_id: int
    ) -> pd.DataFrame:
        if partition_id == 0:
            df = pd.DataFrame(fit_result)

            return df

        return pd.DataFrame([])

    def _get_cuml_transform_func(
        self, dataset: DataFrame, eval_metric_info: Optional[EvalMetricInfo] = None
    ) -> Tuple[
        _ConstructFunc,
        _TransformFunc,
        Optional[_EvaluateFunc],
    ]:
        raise NotImplementedError(
            "DBSCAN does not can not have a separate transform UDF"
        )

    def _transform(self, dataset: DataFrame) -> DataFrame:
        logger = get_logger(self.__class__)
        dataset = self._ensureIdCol(dataset)
        self.features_df = dataset.toPandas()
        # self.output_schema.add(self._get_prediction_name(), IntegerType(), False)
        idCol_name = self.getIdCol()

        default_num_partitions = dataset.rdd.getNumPartitions()

        # Return
        rdd = self._call_cuml_fit_func(
            dataset=dataset,
            partially_collect=False,
            paramMaps=None,
        )
        rdd = rdd.repartition(default_num_partitions)

        pred_df = rdd.toDF()

        return dataset.join(pred_df, idCol_name).drop(idCol_name)

    def _get_model_attributes(self) -> Optional[Dict[str, Any]]:
        """
        Override parent method to bring broadcast variables to driver before JSON serialization.
        """

        self._model_attributes["idCols_"] = [chunk.value for chunk in self.idCols_]
        self._model_attributes["raw_data_"] = [chunk.value for chunk in self.raw_data_]

        self._model_attributes["eps"] = self.eps
        self._model_attributes["min_samples"] = self.min_samples
        self._model_attributes["metric"] = self.metric
        self._model_attributes["max_mbytes_per_batch"] = self.max_mbytes_per_batch
        self._model_attributes["calc_core_sample_indices"] = (
            self.calc_core_sample_indices
        )
        self._model_attributes["verbose"] = self.verbose
        self._model_attributes["processed_input_cols"] = self.processed_input_cols
        self._model_attributes["multi_col_names"] = self.multi_col_names
        self._model_attributes["use_sparse_array"] = self.use_sparse_array
        self._model_attributes["input_col"], self._model_attributes["input_cols"]= self._get_input_columns()

        return self._model_attributes

    def write(self) -> MLWriter:
        return _CumlModelWriterNumpy(self)

    @classmethod
    def read(cls) -> MLReader:
        return _CumlModelReaderNumpy(cls)


class _CumlModelWriterNumpy(_CumlModelWriter):
    """
    Override parent writer to save numpy objects of _CumlModel to the file
    """

    def saveImpl(self, path: str) -> None:
        DefaultParamsWriter.saveMetadata(
            self.instance,
            path,
            self.sc,
            extraMetadata={
                "_cuml_params": self.instance._cuml_params,
                "_num_workers": self.instance._num_workers,
                "_float32_inputs": self.instance._float32_inputs,
            },
        )
        data_path = os.path.join(path, "data")
        model_attributes = self.instance._get_model_attributes()

        if not os.path.exists(data_path):
            os.makedirs(data_path)
        assert model_attributes is not None
        for key, value in model_attributes.items():
            if isinstance(value, list) and isinstance(value[0], np.ndarray):
                paths = []
                for idx, chunk in enumerate(value):
                    array_path = os.path.join(data_path, f"{key}_{idx}.npy")
                    np.save(array_path, chunk, allow_pickle=True)
                    paths.append(array_path)
                model_attributes[key] = paths

        metadata_file_path = os.path.join(data_path, "metadata.json")
        print(model_attributes)
        model_attributes_str = json.dumps(model_attributes)
        self.sc.parallelize([model_attributes_str], 1).saveAsTextFile(
            metadata_file_path
        )


class _CumlModelReaderNumpy(_CumlModelReader):
    """
    Override parent reader to instantiate numpy objects of _CumlModel from file
    """

    def load(self, path: str) -> "_CumlEstimator":
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        data_path = os.path.join(path, "data")
        metadata_file_path = os.path.join(data_path, "metadata.json")

        model_attr_str = self.sc.textFile(metadata_file_path).collect()[0]
        model_attr_dict = json.loads(model_attr_str)

        for key, value in model_attr_dict.items():
            if isinstance(value, list) and value[0].endswith(".npy"):
                arrays = []
                spark = _get_spark_session()
                for array_path in value:
                    array = np.load(array_path, allow_pickle=True)
                    arrays.append(spark.sparkContext.broadcast(array))
                model_attr_dict[key] = arrays

        instance = self.model_cls(**model_attr_dict)
        # DefaultParamsReader.getAndSetParams(instance, metadata)
        instance._cuml_params = metadata["_cuml_params"]
        instance._num_workers = metadata["_num_workers"]
        instance._float32_inputs = metadata["_float32_inputs"]
        return instance
