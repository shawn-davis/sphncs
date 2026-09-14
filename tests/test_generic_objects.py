from dataclasses import dataclass

import numpy as np
import pytest

from sphncs import SphncsClusterer


@dataclass
class Measurement:
    value: float
    group: str


def measurement_distance(left: Measurement, right: Measurement) -> float:
    return abs(left.value - right.value)


def test_clusters_arbitrary_objects_with_a_callable_metric():
    pytest.importorskip("KDEpy")
    objects = [
        Measurement(0.0, "low"), Measurement(0.1, "low"), Measurement(0.2, "low"),
        Measurement(10.0, "high"), Measurement(10.1, "high"), Measurement(10.2, "high"),
    ]
    model = SphncsClusterer(
        metric=measurement_distance, bandwidth=0.15, grid_points=128, random_state=0,
    ).fit(objects)
    assert model.objects_ == objects
    assert all(isinstance(value, Measurement) for value in model.representatives_)
    assert model.predict([Measurement(0.15, "low")]).shape == (1,)


def test_callable_partition_feature_routes_arbitrary_objects():
    pytest.importorskip("KDEpy")
    objects = [
        Measurement(0.0, "low"), Measurement(0.2, "low"), Measurement(9.8, "high"), Measurement(10.0, "high"),
    ]
    model = SphncsClusterer(
        metric=measurement_distance,
        partitioning=True,
        partitioning_feature=lambda value: value.value,
        partitioning_bandwidth=0.15,
        bandwidth=0.15,
        grid_points=128,
        random_state=0,
    ).fit(objects)
    assert np.allclose(model.partition_values_, [0.0, 0.2, 9.8, 10.0])
    assert model.transform([Measurement(0.1, "low")]).shape == (1, 1)


def test_metric_is_required_for_the_generic_estimator():
    with pytest.raises(ValueError, match="metric is required"):
        SphncsClusterer().fit([object(), object()])
