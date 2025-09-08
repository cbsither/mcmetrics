import pytest
import numpy as np

from tests.test_data import TestData
from mcmetrics.metrics import Metrics

# --- Fixtures ---------------------------------------------------------------

@pytest.fixture(scope="module")
def cm():
    """Confusion matrix test data (module-scoped: loaded once per test module)."""
    # If tests might mutate it, return a copy() in each test instead or use function scope.
    return TestData.load_data()

@pytest.fixture
def metrics():
    """Fresh Metrics object per test (function-scoped)."""
    return Metrics()

@pytest.fixture
def metrics_with_cm(metrics, cm):
    """Metrics with its cm set (function-scoped)."""
    metrics.cm = cm
    return metrics

# --- Tests ------------------------------------------------------------------


def test_metrics_import():
    assert Metrics is not None, "Metrics cannot be imported"

def test_data_load(cm):
    assert isinstance(cm, np.ndarray), "Data should be a numpy array"
    assert cm.shape == (3, 3), "Data shape should be (3, 3)"

def test_class_positives_and_negatives(metrics, cm):
    for j in range(cm.shape[0]):
        P_ = metrics.P_(cm, j)
        N_ = metrics.N_(cm, j)
        assert P_ == cm[:, j].sum(), "P_ calculation is incorrect"
        assert N_ == cm.sum() - cm[:, j].sum(), "N_ calculation is incorrect"

def test_class_proportions(metrics_with_cm, cm):
    proportions = metrics_with_cm.class_proportions()
    expected = cm.sum(axis=1) / cm.sum()
    assert np.allclose(proportions, expected), "Class proportions calculation is incorrect"

def test_prevalence(metrics, cm):
    for j in range(cm.shape[0]):
        prevalence = metrics.prevalence(cm, j)
        p_i = metrics.P_(cm, j)
        n_i = metrics.N_(cm, j)
        expected = p_i / (p_i + n_i)
        assert np.isclose(prevalence, expected), f"Prevalence incorrect for class {j}"

@pytest.mark.parametrize("j", [0, 1, 2])
def test_accuracy(metrics, cm, j):
    accuracy = metrics.accuracy(cm, j)

    tp_i = metrics.TP(cm, j)
    tn_i = metrics.TN(cm, j)
    fp_i = metrics.FP(cm, j)
    fn_i = metrics.FN(cm, j)
    numerator_ = tp_i + tn_i
    demoninator_ = tp_i + tn_i + fp_i + fn_i

    assert np.isclose(accuracy, numerator_ / demoninator_), f"Accuracy calculation is incorrect for class {j}"

@pytest.mark.parametrize("j", [0, 1, 2])
def test_sensitivity(metrics, cm, j):
    sensitivity = metrics.sensitivity(cm, j)

    tp_i = metrics.TP(cm, j)
    fn_i = metrics.FN(cm, j)
    numerator_ = tp_i
    demoninator_ = tp_i + fn_i

    assert np.isclose(sensitivity, numerator_ / demoninator_), f"Sensitivity calculation is incorrect for class {j}"

@pytest.mark.parametrize("j", [0, 1, 2])
def test_specificity(metrics, cm, j):
    specificity = metrics.specificity(cm, j)

    tn_i = metrics.TN(cm, j)
    fp_i = metrics.FP(cm, j)
    numerator_ = tn_i
    demoninator_ = tn_i + fp_i

    assert np.isclose(specificity, numerator_ / demoninator_), f"Specificity calculation is incorrect for class {j}"

@pytest.mark.parametrize("j", [0, 1, 2])
def test_true_positive_rate():
    pass

@pytest.mark.parametrize("j", [0, 1, 2])
def test_false_negative_rate():
    pass

@pytest.mark.parametrize("j", [0, 1, 2])
def test_true_negative_rate():
    pass


@pytest.mark.parametrize("j", [0, 1, 2])
def test_balanced_accuracy(metrics, cm, j):
    balanced_accuracy = metrics.balanced_accuracy(cm, j)

    tpr = metrics.tpr(cm, j)
    tnr = metrics.tnr(cm, j)
    numerator_ = tpr + tnr
    demoninator_ = 2

    assert np.isclose(balanced_accuracy, numerator_ / demoninator_), f"Balanced accuracy calculation is incorrect for class {j}"


def test_precision():
    pass


def test_positive_predictive_value():
    pass



##### OLD ####


def test_metrics_import():
    """Test if Metrics can be imported."""
    assert Metrics is not None, "Metrics cannot be imported"


def test_data_load():
    """Test if test data loads correctly."""
    data = TestData.load_data()
    assert isinstance(data, np.ndarray), "Data should be a numpy array"
    assert data.shape == (3, 3), "Data shape should be (3, 3)"


def test_class_positives_and_negatives():

    Metrics_obj = Metrics()
    cm = TestData.load_data()
    indices = range(cm.shape[0])

    for j in indices:
        P_ = Metrics_obj.P_(cm, j)
        N_ = Metrics_obj.N_(cm, j)
        assert P_ == cm[:, j].sum(), "P_ calculation is incorrect"
        assert N_ == cm.sum() - cm[:, j].sum(), "N_ calculation is incorrect"


def test_class_proportions():

    Metrics_obj = Metrics()
    cm = TestData.load_data()
    Metrics_obj.cm = cm  # set confusion matrix for class_proportions method
    proportions = Metrics_obj.class_proportions()
    expected_proportions = cm.sum(axis=1) / cm.sum()
    assert np.allclose(proportions, expected_proportions), "Class proportions calculation is incorrect"


def test_prevalence():
    
    Metrics_obj = Metrics()
    cm = TestData.load_data()
    indices = range(cm.shape[0])

    for j in indices:
        prevalence = Metrics_obj.prevalence(cm, j)
        p_i = Metrics_obj.P_(cm, j)
        n_i = Metrics_obj.N_(cm, j)
        expected_prevalence = p_i / (p_i + n_i)
        assert np.isclose(prevalence, expected_prevalence), f"Prevalence calculation is incorrect for class {j}"


def test_accuracy():
    pass


def test_balanced_accuracy():
    pass


def test_precision():
    pass


def test_positive_predictive_value():
    pass

def test_false_discovery_rate():
    pass

def test_f1_score():
    pass

def test_false_omission_rate():
    pass

def test_negative_predictive_rate():
    pass

def test_fowlkes_mallows_index():
    pass

def test_positive_likelihood_ratio():
    pass

def test_negative_likelihood_ratio():
    pass

def test_diagnostic_odds_ratio():
    pass

def test_markedness():
    pass

def test_informedness():
    pass

def test_matthews_correlation_coefficient():
    pass

def test_prevalence_threshold():
    pass

def test_prevalence_threshold_point_est():
    pass

def test_screening_coefficient():
    pass

def test_threat_score():
    pass

def test_jaccard_index():
    pass

def test_critical_success_index():
    pass

def test_bayesian_positive_predictive_value():
    pass

def test_binomial_me():
    pass

def test_cohens_kappa():
    pass