# **Monte Carlo Metrics**

<p align="center">
  <a href="https://github.com/cbsither/mcmetrics">
    <img src="data/figures/screening_curve.png">
  </a>
</p>

Figure: Example of a screening curve.

## **Table of Contents**

Github Repository: https://github.com/cbsither/mcmetrics

<!--ts-->
* [Overview](#overview)
* [Notes and Future Additions](#notes-and-future-additions)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [List of Metrics](#list-of-metrics)
<!--te-->

## **Overview**

Monte Carlo Metrics (MCMetrics) is an easy to use Python library that leverages Bayesian inference to calculate conditional performance metric uncertainty with respect to sample size. MCMetrics differs from other conditional performance metric libraries by providing credible intervals (CIs) for each metric, which are implicitly calculated from the posterior distribution of the metric. This allows users to understand the uncertainty in their metric estimates, which is particularly useful in fields like medical diagnostics, machine learning, and any domain where performance metrics are critical. Additionally, MCMetrics is built on top of NumPy and Scikit-Learn, making it easy to integrate into existing workflows.

MCMetrics calculates CIs by sampling a posterior Dirichlet distribution with a default or specified prior distribution combined with the data. MCMetrics then calculates statistics like accuracy, recall (i.e., sensitivity), positive predictive value (i.e., precision), among others from the samples (see the [List of Metrics](#list-of-metrics) section).

MCMetrics takes a relatively straight forward approach when calculating conditional performance metrics and CIs. First, it samples $n$ times from the posterior Dirichlet distribution comprised of the prior and confusion matrix. After samples are generated, the user can specify which conditional performance metrics to calculate. Then, each metric can be summarized by its posterior distribution in the form of summary statistics and CIs. Once a metric is calculated, it is permanently stored and only removed if another $n$ samples are called and the ```resample``` parameter set to ```True```.

MCMetrics works by automating calling several independent functions in a workflow. As such, MCMetrics functions can be called independently and function similar to the Scikit-Learn metric functions by specifying either a confusion matrix or the raw parameter values (i.e., TPs, FPs, FNs, and FPs, or the predicted y and actual y values). 

## **Notes and Future Additions**

Note 1: Currently, MCMetrics only fully supports binary classification confusion matrices. It is possible to calculate metrics for multiclass confusion matrices, but the statistics and CIs are equivalent to micro-averaging since each sample from the posterior is weighted equally. Future versions will support macro-averaging and specifying weights. 

Note 2: MCMetrics is currently in active development and all features are not fully implemented. Some of the API may change in future versions, but the core functionality and workflow specified in the [Quick Start](#quick-start) will remain the same.

## **Installation**

Stable
------
The ```pip``` version is the last stable release. Version: *0.1.1*
```sh
pip install mcmetrics
```

## **Quick Start**

```python
import numpy as np
import mcmetrics

# Create a 2x2 confusion matrix. By default 
# mcmetrics interpets the 'actual' condition 
# as rows and 'predicted' condition as columns.

cm1 = np.array([[100,10],
                [10,100]])

# initiate the Metrics class
mc = MCMetrics(model_name="model_1", cm=cm1) # prior is set to 1 by default

# sample posterior
mc.sample(n=1_000) # default n = 100_000

# calculate and return metric and credible intervals

# calculate sensitivity
mc.calculate_metric(metric='sensitivity', averaging=None)
# calculate specificity
mc.calculate_metric(metric='specificity', averaging=None)

# loop calculations
metrics = ['accuracy', 'positive predictive value', 'prevalence threshold']
mc.calculate_metric(metric=metrics, averaging=None)

# calculate summary statistics
mc.calc_mean(metric='sensitivity') # mean
mc.calc_std(metric='sensitivity') # standard deviation
mc.calc_median(metric='sensitivity') # median
mc.calc_mode(metric='sensitivity') # mode
mc.calc_var(metric='sensitivity', ddof=0) # population variance

# chain together calculations
mc.calculate_metric(metric='specificity').calc_mean()

# calculate credible intervals
mc.calculate_ci(metric='sensitivity', ci=0.95)

# the upper and lower bounds of the credible interval can be explicited specified
mc.calculate_ci(metric='sensitivity', cil=0.005, ciu=0.99)

# Remember, the credible intervals are a property of the posterior distribution 
# for a  metric, along with the mean, median, and mode. All of these statistics 
# describe the conditional performance metric uncertainty with respect to 
# sample size.

```

## **List of Metrics**

A full list of metrics is included below. Each metric is called based by the exact command below in the `metric` parameter. For instance, to call precision, you would call `metric='precision'`. Or for a metric with spaces, you would call `metric='positive predictive value'`.

* `prevalence`
* `accuracy`
* `balanced accuracy`
* `precision`
* `positive predictive value`
Same as `precision`.
* `false discovery rate`
* `f1 score`
* `false omission rate`
* `negative predictive value`
* `fowlkes-mallows index`
* `informedness`
* `sensitivity`
* `true positive rate`
Same as `sensitivity`.
* `recall`
Same as `sensitivity`.
* `false positive rate`
* `positive likelihood ratio`
* `markedness`
* `true negative rate`
* `matthews correlation coefficient`
* `prevalence threshold`
* `false negative rate`
* `specificity`
Same as `true negative rate`.
* `negative likelihood ratio`
* `diagnostic odds ratio`
* `jaccard index`
* `critical success index`
Same as `Jaccard Index`.
* `threat score`
Same as `Jaccard Index`.
* `screening coefficient`