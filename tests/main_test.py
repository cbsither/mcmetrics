import pytest
import numpy as np

from mcmetrics.main import MCMetrics 


# some fake data
faux_model_name = 'faux_model'

faux_data = np.array([[100,10],
                      [10,100]])

faux_class_names = ['positive', 'negative']

faux_data_shape = faux_data.shape

faux_prior = 1

n_samples = 1000
n_additional_samples = 100

def test_mcmetrics_import():
    """Test if MCMetrics can be imported."""
    assert MCMetrics is not None, "MCMetrics cannot be imported"

def test_mcmetrics_class_instance():
    """Test if an instance of MCMetrics can be created."""
    obj = MCMetrics()  # add required args if any
    assert isinstance(obj, MCMetrics), "Should create an instance of MCMetrics"


def test_mcmetrics_initialization():
    """Test if MCMetrics initializes with correct attributes."""
    obj = MCMetrics(model_name=faux_model_name, data=faux_data, class_names=faux_class_names, prior=faux_prior)
    assert obj.model_name == faux_model_name, "Model name should match"
    assert np.array_equal(obj.data, faux_data), "Data should match"
    assert obj.class_names == faux_class_names, "Class names should match"

def test_mcmetrics_sampler():
    """Test if the sampler method works correctly."""
    obj = MCMetrics(model_name=faux_model_name, data=faux_data, class_names=faux_class_names, prior=faux_prior)
    obj.sampler(n=1000)  # assuming sampler returns some samples
    assert isinstance(obj.posterior_samples, np.ndarray), "Sampler should return a numpy array"
    assert obj.posterior_samples.shape[0] > 0, "Sampler should return non-empty samples"

def test_myclass_some_behavior():
    obj = MCMetrics(model_name=faux_model_name, data=faux_data, class_names=faux_class_names, prior=faux_prior)
    obj.sampler(n=100_000)  # assuming sampler returns some samples

        # calculate sensitivity
    obj.calculate_metric(metric='sensitivity', averaging=None)

    # assert some expected behavior
    obj.calc_mean(metric='sensitivity'), obj.cm[0,0] / (obj.cm[0,0] + obj.cm[0,1]), 
    obj.cm[1,1] / (obj.cm[1,1] + obj.cm[1,0])
    # Check that calc_mean output for both classes is close to analytical values
    mean_dict = obj.calc_mean(metric='sensitivity')
    expected_0 = obj.cm[0,0] / (obj.cm[0,0] + obj.cm[0,1])
    expected_1 = obj.cm[1,1] / (obj.cm[1,1] + obj.cm[1,0])
    assert np.isclose(mean_dict['Class 0'], expected_0, atol=0.01), f"Class 0 mean {mean_dict['Class 0']} != expected {expected_0}"
    assert np.isclose(mean_dict['Class 1'], expected_1, atol=0.01), f"Class 1 mean {mean_dict['Class 1']} != expected {expected_1}"
