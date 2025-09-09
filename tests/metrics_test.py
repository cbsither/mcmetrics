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
def test_true_positive_rate(metrics, cm, j):
    tpr = metrics.true_positive_rate(cm, j)
    sensitivity = metrics.sensitivity(cm, j)
    assert np.isclose(tpr, sensitivity), f"True Positive Rate calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_false_negative_rate(metrics, cm, j):
    
    fnr = metrics.false_negative_rate(cm, j)

    fn_i = metrics.FN(cm, j)
    tp_i = metrics.TP(cm, j)
    numerator_ = fn_i
    demoninator_ = fn_i + tp_i

    assert np.isclose(fnr, numerator_ / demoninator_), f"False Negative Rate calculation is incorrect for class {j}"
    

@pytest.mark.parametrize("j", [0, 1, 2])
def test_true_negative_rate(metrics, cm, j):
    # same as specificity
    tnr = metrics.true_negative_rate(cm, j)
    tn_i = metrics.TN(cm, j)
    fp_i = metrics.FP(cm, j)
    numerator_ = tn_i
    demoninator_ = tn_i + fp_i
    assert np.isclose(tnr, numerator_ / demoninator_), f"True Negative Rate calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_balanced_accuracy(metrics, cm, j):

    balanced_accuracy = metrics.balanced_accuracy(cm, j)

    tpr = metrics.tpr(cm, j)
    tnr = metrics.tnr(cm, j)
    
    numerator_ = tpr + tnr
    demoninator_ = 2

    assert np.isclose(balanced_accuracy, numerator_ / demoninator_), f"Balanced accuracy calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_precision(metrics, cm, j):
    precision = metrics.precision(cm, j)

    tp_i = metrics.TP(cm, j)
    fp_i = metrics.FP(cm, j)
    numerator_ = tp_i
    demoninator_ = tp_i + fp_i

    assert np.isclose(precision, numerator_ / demoninator_), f"Precision calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_positive_predictive_value(metrics, cm, j):
    ppv = metrics.positive_predictive_value(cm, j)
    precision = metrics.precision(cm, j)
    assert np.isclose(ppv, precision), f"PPV calculation is incorrect for class {j}"



@pytest.mark.parametrize("j", [0, 1, 2])
def test_false_discovery_rate(metrics, cm, j):
    # same as 1 - positive predictive value
    fdr = metrics.false_discovery_rate(cm, j)
    ppv = metrics.positive_predictive_value(cm, j)
    assert np.isclose(fdr, 1 - ppv), f"FDR calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_false_omission_rate(metrics, cm, j):
    
    fomr = metrics.false_omission_rate(cm, j)

    fn_i = metrics.FN(cm, j)
    tn_i = metrics.TN(cm, j)
    numerator_ = fn_i
    demoninator_ = fn_i + tn_i

    assert np.isclose(fomr, numerator_ / demoninator_), f"False Omission Rate calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_negative_predictive_rate(metrics, cm, j):
    npr = metrics.negative_predictive_rate(cm, j)

    tn_i = metrics.TN(cm, j)
    fn_i = metrics.FN(cm, j)
    numerator_ = tn_i
    demoninator_ = tn_i + fn_i

    assert np.isclose(npr, numerator_ / demoninator_), f"Negative Predictive Rate calculation is incorrect for class {j}"


####### complex metrics tests below - contain combinations of other metrics ########

@pytest.mark.parametrize("j", [0, 1, 2])
def test_f1_score(metrics, cm, j):
    
    f1 = metrics.f1_score(cm, j)

    tp_i = metrics.TP(cm, j)
    fp_i = metrics.FP(cm, j)
    fn_i = metrics.FN(cm, j)
    numerator_ = 2 * tp_i
    demoninator_ = 2 * tp_i + fp_i + fn_i

    assert np.isclose(f1, numerator_ / demoninator_), f"F1 Score calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_fowlkes_mallows_index(metrics, cm, j):
    fmi = metrics.fowlkes_mallows_index(cm, j)

    ppv = metrics.positive_predictive_value(cm, j)
    tpr = metrics.true_positive_rate(cm, j)
    expected = np.sqrt(ppv * tpr)

    assert np.isclose(fmi, expected), f"Fowlkes-Mallows Index calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_positive_likelihood_ratio(metrics, cm, j):
    plr = metrics.positive_likelihood_ratio(cm, j)

    tpr = metrics.true_positive_rate(cm, j)
    fpr = metrics.true_negative_rate(cm, j)

    assert np.isclose(plr, tpr / fpr), f"Positive Likelihood Ratio calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_negative_likelihood_ratio(metrics, cm, j):
    nlr = metrics.negative_likelihood_ratio(cm, j)

    fnr = metrics.false_negative_rate(cm, j)
    tnr = metrics.true_negative_rate(cm, j)

    assert np.isclose(nlr, fnr / tnr), f"Negative Likelihood Ratio calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_diagnostic_odds_ratio(metrics, cm, j):
    dor = metrics.diagnostic_odds_ratio(cm, j)

    plr = metrics.positive_likelihood_ratio(cm, j)
    nlr = metrics.negative_likelihood_ratio(cm, j)

    assert np.isclose(dor, plr / nlr), f"Diagnostic Odds Ratio calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_markedness(metrics, cm, j):
    mk = metrics.markedness(cm, j)

    ppv = metrics.positive_predictive_value(cm, j)
    npv = metrics.negative_predictive_rate(cm, j)

    assert np.isclose(mk, ppv + npv - 1), f"Markedness calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_informedness(metrics, cm, j):
    
    im = metrics.informedness(cm, j)

    tpr = metrics.true_positive_rate(cm, j)
    tnr = metrics.true_negative_rate(cm, j)

    assert np.isclose(im, tpr + tnr - 1), f"Informedness calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_matthews_correlation_coefficient(metrics, cm, j):
    pass



@pytest.mark.parametrize("j", [0, 1, 2])
def test_prevalence_threshold(metrics, cm, j):

    pt = metrics.prevalence_threshold(cm, j)

    sens = metrics.sensitivity(cm, j)
    spec = metrics.specificity(cm, j)

    numerator_ = (np.sqrt(sens * (-spec + 1)) + spec - 1)
    demoninator_ = sens + spec - 1

    assert np.isclose(pt, numerator_ / demoninator_), f"Prevalence Threshold calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_prevalence_threshold_point_est(metrics, cm, j):
    
    ptpe = metrics.prevalence_threshold_point_est(metrics.sensitivity(cm, j), 
                                                  metrics.specificity(cm, j))
    
    sens = metrics.sensitivity(cm, j)
    spec = metrics.specificity(cm, j)

    numerator_ = (np.sqrt(sens * (-spec + 1)) + spec - 1)
    demoninator_ = sens + spec - 1

    assert np.isclose(ptpe, numerator_ / demoninator_), f"Prevalence Threshold Point Estimation calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_screening_coefficient(metrics, cm, j):

    sc = metrics.screening_coefficient(cm, j)

    sens = metrics.sensitivity(cm, j)
    spec = metrics.specificity(cm, j)

    assert np.isclose(sc, sens + spec), f"Screening Coefficient calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_threat_score(metrics, cm, j):
    
    ts = metrics.threat_score(cm, j)

    tp_i = metrics.TP(cm, j)
    fp_i = metrics.FP(cm, j)
    fn_i = metrics.FN(cm, j)

    numerator_ = tp_i
    demoninator_ = tp_i + fn_i + fp_i
    assert np.isclose(ts, numerator_ / demoninator_), f"Threat Score calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_jaccard_index(metrics, cm, j):
    # same as threat score
    ji = metrics.jaccard_index(cm, j)
    ts = metrics.threat_score(cm, j)

    assert np.isclose(ji, ts), f"Jaccard Index calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_critical_success_index(metrics, cm, j):
    # same as threat score
    csi = metrics.critical_success_index(cm, j)
    ts = metrics.threat_score(cm, j)

    assert np.isclose(csi, ts), f"Critical Success Index calculation is incorrect for class {j}"


@pytest.mark.parametrize("j", [0, 1, 2])
def test_bayesian_positive_predictive_value(metrics, cm, j, prior=0.5):
    # using confusion matrix

    # using a prespecificed prior


    pass


@pytest.mark.parametrize("j", [0, 1, 2])
def test_binomial_me(metrics, cm, j):
    pass


@pytest.mark.parametrize("j", [0, 1, 2])
def test_cohens_kappa(metrics, cm, j):
    
    pass