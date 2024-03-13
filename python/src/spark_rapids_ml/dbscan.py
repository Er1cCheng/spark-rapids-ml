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
from pyspark.sql import Column
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import lit, monotonically_increasing_id, row_number, spark_partition_id
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    FloatType,
    IntegerType,
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
    _CumlModelWithPredictionCol,
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
            "metric": "‘euclidean’",
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

    def setEps(self: P, value: float) -> P:
        return self._set_params(eps=value)

    def getEps(self) -> float:
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

        select_cols, multi_col_names, dimension, _ = self._pre_process_data(dataset)
        use_sparse_array = _use_sparse_in_cuml(dataset)
        input_dataset = dataset.select(*select_cols)
        raw_data: np.ndarray = np.array(input_dataset.toPandas())

        broadcast_raw_data = [
            spark.sparkContext.broadcast(chunk) for chunk in _chunk_arr(raw_data)
        ]

        model = DBSCANModel(
            raw_data_=broadcast_raw_data,
            n_cols=len(raw_data[0]),
            dtype=type(raw_data[0][0][0]).__name__,
            output_schema=dataset.schema,
            input_cols=input_dataset.columns,
            multi_col_names=multi_col_names,
            use_sparse_array=use_sparse_array,
        )

        model._num_workers = self.num_workers
        model.eps = self.getOrDefault("eps")
        model.min_samples = self.getOrDefault("min_samples")
        model.metric = self.getOrDefault("metric")
        model.max_mbytes_per_batch = self.getOrDefault("max_mbytes_per_batch")
        model.calc_core_sample_indices = self.getOrDefault("calc_core_sample_indices")

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
        n_cols: int,
        dtype: str,
        raw_data_: List[pyspark.broadcast.Broadcast],
        output_schema: StructType,
        input_cols: List[str],
        multi_col_names: List[str] | None,
        use_sparse_array: bool
    ):
        super(DBSCANClass, self).__init__()

        super(_CumlModelWithPredictionCol, self).__init__(
            n_cols=n_cols, dtype=dtype, raw_data_=raw_data_
        )

        super(_DBSCANCumlParams, self).__init__()

        self._dbscan_spark_model = None
        self.output_schema = output_schema
        self.input_cols = input_cols
        self.raw_data_ = raw_data_
        self.multi_col_names = multi_col_names
        self.use_sparse_array = use_sparse_array

    def _pre_process_data(self, dataset: DataFrame) -> Tuple[
        List[Column],
        Optional[List[str]],
        int,
        Union[Type[FloatType], Type[DoubleType]],
    ]:
        return _CumlCaller._pre_process_data(self, dataset)

    def _out_schema(
        self, input_schema: StructType = StructType()
    ) -> Union[StructType, str]:
        return self.output_schema

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
        logger = get_logger(self.__class__)

        cuda_managed_mem_enabled = (
            _get_spark_session().conf.get("spark.rapids.ml.uvm.enabled", "false")
            == "true"
        )

        inputs = []

        for pdf_bc in self.raw_data_:
            pdf = pd.DataFrame(data=pdf_bc.value, columns=self.input_cols)

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
                    if use_sparse_array is False
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

            dbscan = CumlDBSCANMG(
                handle=params[param_alias.handle],
                output_type="cudf",
                eps=self.eps,
                min_samples=self.min_samples,
                metric=self.metric,
                max_mbytes_per_batch=self.max_mbytes_per_batch,
                calc_core_sample_indices=self.calc_core_sample_indices,
            )
            dbscan.n_cols = n_cols
            dbscan.dtype = np.dtype(dtype)

            res = list(dbscan.fit_predict(concated).to_numpy())

            return pd.Series(res)

        return _cuml_fit

    def fit_post_process(
        self, fit_result: Dict[str, Any]
    ) -> pd.DataFrame:
        self.features_df[self._get_prediction_name()] = fit_result

        return self.features_df

    def _call_cuml_fit_func_df(
        self,
        dataset: DataFrame,
        partially_collect: bool = True,
        paramMaps: Optional[Sequence["ParamMap"]] = None,
    ) -> RDD:
        """
        Fits a model to the input dataset. This is called by the default implementation of fit.

        Parameters
        ----------
        dataset : :py:class:`pyspark.sql.DataFrame`
            input dataset

        Returns
        -------
        :class:`Transformer`
            fitted model
        """
        self._validate_parameters()

        cls = self.__class__

        select_cols, multi_col_names, dimension, _ = self._pre_process_data(dataset)

        num_workers = self.num_workers

        dataset = dataset.select(*select_cols)

        if dataset.rdd.getNumPartitions() != num_workers:
            dataset = self._repartition_dataset(dataset)

        is_local = _is_local(_get_spark_session().sparkContext)

        cuda_managed_mem_enabled = (
            _get_spark_session().conf.get("spark.rapids.ml.uvm.enabled", "false")
            == "true"
        )
        if cuda_managed_mem_enabled:
            get_logger(cls).info("CUDA managed memory enabled.")

        # parameters passed to subclass
        params: Dict[str, Any] = {
            param_alias.cuml_init: self.cuml_params,
        }

        # Convert the paramMaps into cuml paramMaps
        fit_multiple_params = []
        if paramMaps is not None:
            for paramMap in paramMaps:
                tmp_fit_multiple_params = {}
                for k, v in paramMap.items():
                    name = self._get_cuml_param(k.name, False)
                    assert name is not None
                    tmp_fit_multiple_params[name] = self._get_cuml_mapping_value(
                        name, v
                    )
                fit_multiple_params.append(tmp_fit_multiple_params)
        params[param_alias.fit_multiple_params] = fit_multiple_params

        cuml_fit_func = self._get_cuml_fit_func(
            dataset, None if len(fit_multiple_params) == 0 else fit_multiple_params
        )

        fit_post_process_func = (
            self.fit_post_process
            if hasattr(self.__class__, "fit_post_process")
            else None
        )

        array_order = self._fit_array_order()

        cuml_verbose = self.cuml_params.get("verbose", False)

        use_sparse_array = _use_sparse_in_cuml(dataset)

        (enable_nccl, require_ucx) = self._require_nccl_ucx()

        def _train_udf(pdf_iter: Iterator[pd.DataFrame]) -> pd.DataFrame:
            import cupy as cp
            import cupyx
            from pyspark import BarrierTaskContext

            context = BarrierTaskContext.get()
            partition_id = context.partitionId()
            logger = get_logger(cls)

            # set gpu device
            _CumlCommon._set_gpu_device(context, is_local)

            if cuda_managed_mem_enabled:
                import rmm
                from rmm.allocators.cupy import rmm_cupy_allocator

                rmm.reinitialize(
                    managed_memory=True,
                    devices=_CumlCommon._get_gpu_device(context, is_local),
                )
                cp.cuda.set_allocator(rmm_cupy_allocator)

            _CumlCommon._initialize_cuml_logging(cuml_verbose)

            # handle the input
            # inputs = [(X, Optional(y)), (X, Optional(y))]
            logger.info("Loading data into python worker memory")
            inputs = []
            sizes = []

            for pdf in pdf_iter:
                sizes.append(pdf.shape[0])
                if multi_col_names:
                    features = np.array(pdf[multi_col_names], order=array_order)
                elif use_sparse_array:
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
                        if use_sparse_array is False
                        else cupyx.scipy.sparse.csr_matrix(features)
                    )

                label = pdf[alias.label] if alias.label in pdf.columns else None
                row_number = (
                    pdf[alias.row_number] if alias.row_number in pdf.columns else None
                )
                inputs.append((features, label, row_number))

            if len(sizes) == 0 or all(sz == 0 for sz in sizes):
                raise RuntimeError(
                    "A python worker received no data.  Please increase amount of data or use fewer workers."
                )

            logger.info("Initializing cuml context")
            with CumlContext(
                partition_id, num_workers, context, enable_nccl, require_ucx
            ) as cc:
                params[param_alias.handle] = cc.handle
                params[param_alias.part_sizes] = sizes
                params[param_alias.num_cols] = dimension
                params[param_alias.loop] = cc._loop

                logger.info("Invoking cuml fit")

                # call the cuml fit function
                # *note*: cuml_fit_func may delete components of inputs to free
                # memory.  do not rely on inputs after this call.
                result = cuml_fit_func(inputs, params)
                result_df = (
                    fit_post_process_func(result)
                    if fit_post_process_func is not None
                    else pd.DataFrame(data=result)
                )
                logger.info("Cuml fit complete")

            if partially_collect == True:
                if enable_nccl:
                    context.barrier()

                if context.partitionId() == 0:
                    yield result_df
            else:
                yield result_df

        pipelined_rdd = (
            dataset.mapInPandas(_train_udf, schema=self._out_schema())  # type: ignore
            .rdd.barrier()
            .mapPartitions(lambda x: x)
        )

        return pipelined_rdd

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

            dbscan = CumlDBSCANMG(
                output_type="cudf",
                eps=self.eps,
            )
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
        logger = get_logger(self.__class__)
        self.features_df = dataset.toPandas()
        self.output_schema.add(self._get_prediction_name(), IntegerType(), False)

        default_num_partitions = dataset.rdd.getNumPartitions()

        # Return
        rdd = self._call_cuml_fit_func_df(
            dataset=dataset,
            partially_collect=False,
            paramMaps=None,
        )
        rdd = rdd.repartition(default_num_partitions)

        pred_df = rdd.toDF()

        return pred_df

        # window_spec = Window.orderBy(lit(1))
        # pred_df = pred_df.withColumn("index", row_number().over(window_spec))
        # dataset = dataset.withColumn("index", row_number().over(window_spec))

        # dataset.show()
        # pred_df.show()

        # return dataset.join(pred_df, "index").drop("index")
