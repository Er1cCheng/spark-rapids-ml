#
# Copyright (c) 2022, NVIDIA CORPORATION.
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
from typing import Any, Tuple

import numpy as np
import pytest

from sparkcuml.linear_model.linear_regression import (
    SparkCumlLinearRegression,
    SparkCumlLinearRegressionModel,
)
from sparkcuml.tests.sparksession import CleanSparkSession
from sparkcuml.tests.utils import (
    array_equal,
    create_pyspark_dataframe,
    cuml_supported_data_types,
    idfn,
    make_regression_dataset,
    pyspark_supported_feature_types,
)


# @lru_cache(4) TODO fixme: TypeError: Unhashable Type” Numpy.Ndarray
def train_with_cuml_linear_regression(
    X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float
) -> Any:
    if alpha == 0:
        from cuml import LinearRegression as cuLinearRegression

        lr = cuLinearRegression(output_type="numpy")
    else:
        if l1_ratio == 0.0:
            from cuml import Ridge as cuLinearRegression

            lr = cuLinearRegression(output_type="numpy", alpha=alpha)
        else:
            raise NotImplementedError("Lassor and ElasticNet have not been implemented")

    lr.fit(X, y)
    return lr


@pytest.mark.parametrize("l2", [0.0, 0.7])
def test_linear_regression_estimator_basic(tmp_path: str, l2: float) -> None:
    # test estimator default param
    lr = SparkCumlLinearRegression()
    assert lr.getOrDefault("algorithm") == "eig"
    assert lr.getOrDefault("solver") == "eig"
    assert lr.getOrDefault("fit_intercept")
    assert not lr.getOrDefault("normalize")
    assert lr.getRegParam() == 0.0

    def assert_params(linear_reg: SparkCumlLinearRegression, l2: float) -> None:
        assert linear_reg.getOrDefault("algorithm") == "svd"
        assert not linear_reg.getOrDefault("fit_intercept")
        assert linear_reg.getOrDefault("normalize")
        assert linear_reg.getRegParam() == l2

    lr = SparkCumlLinearRegression(algorithm="svd", fit_intercept=False, normalize=True)
    lr.setRegParam(l2)
    assert_params(lr, l2)

    # Estimator persistence
    path = tmp_path + "/linear_regression_tests"
    estimator_path = f"{path}/linear_regression"
    lr.write().overwrite().save(estimator_path)
    lr_loaded = SparkCumlLinearRegression.load(estimator_path)

    assert_params(lr_loaded, l2)


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("data_shape", [(10, 2)], ids=idfn)
@pytest.mark.parametrize("l2", [0.0, 0.7])
def test_linear_regression_model_basic(
    tmp_path: str,
    feature_type: str,
    data_type: np.dtype,
    data_shape: Tuple[int, int],
    l2: float,
) -> None:
    # Train a toy model
    X, _, y, _ = make_regression_dataset(data_type, data_shape[0], data_shape[1])
    with CleanSparkSession() as spark:
        df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X, y
        )

        lr = SparkCumlLinearRegression()
        lr.setRegParam(l2)
        lr.setFeaturesCol(features_col)
        assert label_col is not None
        lr.setLabelCol(label_col)

        assert lr.getFeaturesCol() == features_col
        assert lr.getLabelCol() == label_col

        def assert_model(
            lhs: SparkCumlLinearRegressionModel, rhs: SparkCumlLinearRegressionModel
        ) -> None:
            assert lhs.coef == rhs.coef
            assert lhs.intercept == lhs.intercept
            assert lhs.dtype == np.dtype(data_type).name
            assert lhs.dtype == rhs.dtype
            assert lhs.n_cols == rhs.n_cols
            assert lhs.n_cols == data_shape[1]

        # train a model
        lr_model = lr.fit(df)

        # model persistence
        path = tmp_path + "/linear_regression_tests"
        model_path = f"{path}/linear_regression_model"
        lr_model.write().overwrite().save(model_path)

        lr_model_loaded = SparkCumlLinearRegressionModel.load(model_path)
        assert_model(lr_model, lr_model_loaded)


@pytest.mark.parametrize("feature_type", pyspark_supported_feature_types)
@pytest.mark.parametrize("data_shape", [(1000, 20)], ids=idfn)
@pytest.mark.parametrize("data_type", cuml_supported_data_types)
@pytest.mark.parametrize("max_record_batch", [100, 10000])
@pytest.mark.parametrize("alpha", [0.0, 0.7])  # equal to reg parameter
@pytest.mark.parametrize("l1_ratio", [0.0])  # equal to elastic_net parameter
def test_linear_regression(
    gpu_number: int,
    feature_type: str,
    data_shape: Tuple[int, int],
    data_type: np.dtype,
    max_record_batch: int,
    alpha: float,
    l1_ratio: float,
) -> None:
    X_train, X_test, y_train, _ = make_regression_dataset(
        data_type, data_shape[0], data_shape[1]
    )

    cu_lr = train_with_cuml_linear_regression(X_train, y_train, alpha, l1_ratio)
    cu_expected = cu_lr.predict(X_test)

    conf = {"spark.sql.execution.arrow.maxRecordsPerBatch": str(max_record_batch)}
    with CleanSparkSession(conf) as spark:
        train_df, features_col, label_col = create_pyspark_dataframe(
            spark, feature_type, data_type, X_train, y_train
        )
        assert label_col is not None
        slr = SparkCumlLinearRegression(num_workers=gpu_number, verbose=7)
        slr.setRegParam(alpha)
        slr.setElasticNetParam(l1_ratio)
        slr.setFeaturesCol(features_col)
        slr.setLabelCol(label_col)
        slr_model = slr.fit(train_df)

        assert array_equal(cu_lr.coef_, slr_model.coef, 1e-3, 1e-3)

        test_df, _, _ = create_pyspark_dataframe(spark, feature_type, data_type, X_test)

        result = slr_model.transform(test_df).collect()
        pred_result = [row.prediction for row in result]
        assert array_equal(cu_expected, pred_result, 1e-3, 1e-3)
