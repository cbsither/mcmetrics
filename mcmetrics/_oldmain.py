import numpy as np
import csv
from sklearn.metrics import confusion_matrix
import pickle
import multiprocessing as mp


class MCMetrics:

    def __init__(self, model_name, prior=1, cm=None, 
                 y_true=None, y_pred=None, class_names=None, threads=4):
    
        if threads < mp.cpu_count():
            self.threads = threads
        else:
            raise ValueError("Number of threads must be less than the number of available CPU cores.")

        if cm is not None:
            self.cm = np.array(cm).astype('float')
        elif y_true is not None and y_pred is not None:
            self.cm = confusion_matrix(y_true, y_pred)
        else:
            raise ValueError("Either confusion matrix (cm) or true "
                             "and predicted labels (y_true, y_pred) must be provided.")

        self.model_name = model_name

        if isinstance(prior, (int, float)):
            self.prior = np.ones_like(self.cm) * prior
        elif isinstance(prior, np.ndarray) and prior.shape == self.cm.shape:
            self.prior = prior
        else:
            raise ValueError("Prior must be a scalar or a numpy array " \
                             "with the same shape as the confusion matrix.")
        
        self.matrix_size = self.cm.shape[0]

        if class_names is None:
            self.class_names = [f"Class {i}" for i in range(self.matrix_size)]
        elif isinstance(class_names, list) and all(isinstance(name, str) for name in class_names):
            self.class_names = class_names

        if len(self.class_names) != self.matrix_size:
            raise ValueError("Number of class names must match the number " \
                             "of classes in the confusion matrix.")

        self.posterior_samples = None
        self.sample_size = None
        self.metrics = {}

        # reference dictionary for summary statistic functions
        self.summary_stats = {'mean': self.calc_mean,
                              'mode': self.calc_mode,
                              'median': self.calc_median,
                              'variance': self.calc_var,
                              'var': self.calc_var,
                              'standard deviation': self.calc_std,
                              'std': self.calc_std}

        # reference dictionary for metrics functions and data arrays
        self.metric_metadata = {'prevalence': {'function': self.prev,
                                               'data array': 'prevalence'},

                                'accuracy': {'function': self.acc,
                                               'data array': 'accuracy'},

                                'balanced accuracy': {'function': self.ba,
                                                      'data array': 'balanced_accuracy'},

                                'precision': {'function': self.prec,
                                              'data array': 'precision'},

                                'positive predictive value': {'function': self.ppv,
                                              'data array': 'precision'}, # ppv is the same as precision

                                'false discovery rate': {'function': self.fdr,
                                              'data array': 'false_discovery_rate'},

                                'f1 score': {'function': self.f1,
                                              'data array': 'f1_score'},

                                'false omission rate': {'function': self.for_,
                                              'data array': 'false_omission_rate'},

                                'negative predictive value': {'function': self.npv,
                                              'data array': 'negative_predictive_value'},

                                'fowlkes-mallows index': {'function': self.fm,
                                              'data array': 'fowkes_mallows_index'},

                                'informedness': {'function': self.informedness,
                                              'data array': 'informedness'},

                                'sensitivity': {'function': self.sens,
                                              'data array': 'sensitivity'}, # sensitivity is the same as tpr

                                'true positive rate': {'function': self.sens,
                                              'data array': 'sensitivity'}, # tpr is the same as sensitivity

                                'recall': {'function': self.sens,
                                              'data array': 'sensitivity'}, # recall is the same as sensitivity

                                'false positive rate': {'function': self.fpr,
                                              'data array': 'false_positive_rate'},

                                'positive likelihood ratio': {'function': self.pos_LR,
                                              'data array': 'positive_likelihood_ratio'},

                                'markedness': {'function': self.markedness,
                                              'data array': 'markedness'},

                                'true negative rate': {'function': self.spec,
                                              'data array': 'specificity'}, # tnr is the same as specificity

                                'matthews correlation coefficient': {'function': self.mcc,
                                              'data array': 'matthews_correlation_coefficient'},

                                'prevalence threshold': {'function': self.pt,
                                              'data array': 'prevalence_threshold'},

                                'false negative rate': {'function': self.fnr,
                                              'data array': 'false_negative_rate'},

                                'specificity': {'function': self.spec,
                                              'data array': 'specificity'},

                                'negative likelihood ratio': {'function': self.neg_LR,
                                              'data array': 'negative_likelihood_ratio'},

                                'diagnostic odds ratio': {'function': self.dor,
                                              'data array': 'diagnotic_odds_ratio'},

                                'jaccard index': {'function': self.jaccard_index,
                                              'data array': 'jaccard_index'},

                                'critical success index': {'function': self.jaccard_index,
                                              'data array': 'jaccard_index'}, # same as jaccard index

                                'threat score': {'function': self.jaccard_index,
                                              'data array': 'jaccard_index'}, # same as jaccard index

                                'screening coefficient': {'function': self.screening_coefficient,
                                              'data array': 'screening_coefficient'},

                                'cohens kappa': {'function': self.cohens_kappa,
                                                 'data array': 'cohens_kappa'}

                                }
        

    """ Error Functions """

    def validate_prior(self):
        if isinstance(self.prior, (int, float)):
            self.prior = np.ones_like(self.cm) * self.prior
        elif isinstance(self.prior, np.ndarray) and self.prior.shape == self.cm.shape:
            self.prior = self.prior
        else:
            raise ValueError("Prior must be a scalar or a numpy array " \
                             "with the same shape as the confusion matrix.")

    def validate_posterior_samples(self):
        if self.posterior_samples is None:
            raise ValueError("Posterior samples not generated. Call sample() first.")

    def validate_sample_size(self):
        if self.sample_size is None:
            raise AttributeError("Posterior samples not generated. Call sample() first.")

    def validate_metric(self, metric):
        metric_name = self.metric_metadata[metric]['data array'] 
        if metric_name not in self.metrics:
            raise ValueError(
                f"metric {metric} not found in metrics. Call calculate_metric() first."
            )

    def metric_check(self, metrics, models):
        if len(metrics) > 0:
            for metric_ in metrics:
                for model_ in models:
                    if metric_ not in model_.metrics:
                        model_.calculate_metric(metric=metric_)



    """ Sampler and Helper Functions """

    def division_of_labor(self, samples):
        """
        Determines the number of samples to be assigned to each thread.
        """

        min_samples = samples // self.threads
        thread_spool = [min_samples] * self.threads

        for i in range(0, samples % self.threads):
            thread_spool[i] += 1

        return thread_spool

    def retrieve_metrics_list(self):
        """
        Returns:
            list: a list of all metrics in the metric_metadata dictionary
        """
        return list(self.metric_metadata.keys())


    def dirichlet_samples(self, alpha, size):
        """
        Draw multiple samples from a Dirichlet distribution with parameters `alpha`.

        Args:
            alpha (array-like): 1D array of shape (k,) containing the parameters (alpha_i > 0).
            size (int):         Number of samples to draw.

        Returns:
            np.ndarray:
                A (size, k) array where each row is a k-dimensional Dirichlet sample.
        """
        # Prepare an empty (size x k) array for the Gamma draws
        arr = np.zeros((size, len(alpha)))

        # For each component alpha_i, draw `size` Gamma samples and store as a column
        for i, alpha_ in enumerate(alpha):
            gamma_draws = np.random.gamma(shape=alpha_, scale=1.0, size=size)
            arr[:, i] = gamma_draws

        # Normalize rows to sum to 1
        row_sums = arr.sum(axis=1, keepdims=True)
        return arr / row_sums

    def significant_digits(self):
        """
        Calculate the number of significant digits in the total number of samples.

        This method computes the number of significant digits in the sum of all elements
        in the confusion matrix (self.cm). If the total number of samples is zero, it returns 0.
        Otherwise, it returns the number of digits in the total sample count.

        Returns:
            int: The number of significant digits in the total number of samples.
        """

        total_samples = self.cm.sum()

        if total_samples == 0:
            return 0
        
        return int(np.floor(np.log10(total_samples))) + 1

    def posterior_check(self):
        """
        Checks if posterior samples have been generated.

        This method verifies whether the posterior samples have been generated by checking
        if the `posterior_samples` attribute is not None. 

        Raises:
            ValueError: If posterior samples have not been generated.

        Returns:
            bool: True if posterior samples have been generated.
        """
        
        if self.posterior_samples is None:
            raise ValueError("Posterior samples not generated. Call sample() first.")
        else:
            return True

    def alpha_(self):
        """
        Returns:
            numpy array: returns the alpha parameter values for the Dirichlet distribution
        """
        return self.cm + self.prior
    
    def probability_matrix(self, include_prior=True):
        """
        
        """
        if include_prior:
            return self.alpha_() / self.alpha_().sum()
        else:
            return self.cm / self.cm.sum()

    def dirichlet_counts(self):
        """
        Returns:
            NumPy Array: a vector containing the alpha parameters without the prior
        """
        return self.cm.flatten()
    
    def dirichlet_prob(self):
        """
        Returns:
            NumPy Array: a vector containing the normalized probabilities for each 
        """
        return self.probability_matrix().flatten()


    def sample(self, n=100_000, include_prior=True, resample=False):

        if (not isinstance(self.posterior_samples, np.ndarray) and self.sample_size is None) or resample is True:
        
            self.sample_size = n

            alpha_array = self.alpha_() if include_prior == True else self.cm
            if alpha_array.min() == 0:
                raise ValueError("All alpha values must be gre+ater than 0.")
            else:
                self.posterior_samples = self.dirichlet_samples(alpha_array.flatten(), 
                                                            size=n).reshape(-1, *self.cm.shape)
        
        else:
            alpha_array = self.alpha_() if include_prior == True else self.cm
            if alpha_array.min() == 0:
                raise ValueError("All alpha values must be gre+ater than 0.")
            else:
                self.sample_size += n
                new_samples = self.dirichlet_samples(alpha_array.flatten(), size=n).reshape(-1, *self.cm.shape)
                self.posterior_samples = np.concatenate([self.posterior_samples, new_samples], axis=0)


    def ci(self, arr, cil, ciu):
        return np.quantile(a=arr, q=[cil, ciu], axis=0)

    def ci_range(self, ci, cil, ciu):
        """
        Checks to ensure the credible interval ranges are specified correctly.
        """
        if cil is None and ciu is None:
            cil = (1 - ci) / 2
            ciu = 1 - cil
        elif cil is None and ciu is not None:
            cil = 1 - ciu
        elif cil is not None and ciu is None:
            ciu = 1 - cil
        else:
            pass
        return cil, ciu

    def smoothed_hist_calc_mode(self, metric, axis: int = 0) -> np.ndarray:

        # Access the metric data
        data = self.metrics[self.metric_metadata[metric]['data array']]

        metric_dict = {} 

        # Calculate the histogram
        for cc in range(data.shape[1]):
            bins = np.linspace(data[:,cc].min(), data[:,cc].max(), 1000)
            hist, bins = np.histogram(data[:,cc], bins=bins, density=True)
            smoothed_hist = self.smooth_histogram(hist=hist, w=51, polyorder=3)
            est_mode = self.calc_mode_alt(arr=smoothed_hist, bins=bins)
            metric_dict[self.class_names[cc]] = est_mode

        return metric_dict
    
    def calc_mode(self, metric: str = None, axis: int = 0) -> np.ndarray:
        """
        Estimate the mode of a (unimodal) sample via the half-sample-mode algorithm.
        Returns an array of modes along the specified axis.
        """
        if metric is None:
            if not hasattr(self, '_metric'):
                raise ValueError("No metric specified. Please specify a metric to calculate the mean.")
            metric = self._metric

        if len(metric[0]) == 1:

            # Access the metric data
            data = self.metrics[self.metric_metadata[metric[0]]['data array']]

            def hsm_recursive(arr):
                n = len(arr)
                # Base case: if 1 or 2 points remain, return their midpoint
                if n <= 2:
                    return 0.5 * (arr[0] + arr[-1])
                # Number of points in the half
                half = (n + 1) // 2  # ceiling of n/2
                # Find sub-interval of length 'half' that has the smallest range
                min_width = np.inf
                min_idx = 0
                for i in range(n - half + 1):
                    width = arr[i + half - 1] - arr[i]
                    if width < min_width:
                        min_width = width
                        min_idx = i
                # Restrict to that smallest sub-interval
                new_arr = arr[min_idx : min_idx + half]
                return hsm_recursive(new_arr)

            # Apply the half-sample-mode algorithm along the specified axis
            if axis == 0:
                temp_metrics = np.apply_along_axis(lambda x: hsm_recursive(np.sort(x)), axis, data)
                # add class labels to the dictionary
                metric_dict = {self.class_names[i]: temp_metrics[i] for i in range(len(temp_metrics))}
                return metric_dict
            elif axis == 1:
                temp_metrics = np.apply_along_axis(lambda x: hsm_recursive(np.sort(x)), axis, data)
                # add class labels to the dictionary
                metric_dict = {f'row_{i}': temp_metrics[i] for i in range(len(temp_metrics))}
                return metric_dict
            else:
                raise ValueError("Axis must be 0 or 1")
            
        else:
            raise ValueError("Mode calculation is only supported for a single metric at a time. Please specify a single metric.")


    def fit_beta(self, metric=None):
        """
        Fits a Beta distribution to the posterior samples for a metric.

        Returns:
        """
        if metric is None:
            if not hasattr(self, '_metric'):
                raise ValueError("No metric specified. Please specify a metric to calculate the mean.")
            metric = self._metric

        if len(metric) == 1:

            mu_est = self.calc_mean(metric)
            var_est = self.calc_var(metric, ddof=0)

            alpha_hat = {}
            beta_hat = {}

            for key in mu_est:
                

                conc_param = mu_est[key]*(1-mu_est[key])/var_est[key] - 1
                prop_alpha_hat = mu_est[key] * conc_param
                prop_beta_hat  = (1-mu_est[key]) * conc_param

                alpha_hat[key] = prop_alpha_hat
                beta_hat[key] = prop_beta_hat

            # reorganize the dictionaries to match class names

            class_hat = {}

            for i in range(len(self.class_names)):
                class_hat[self.class_names[i]] = {
                    'alpha': alpha_hat.get(self.class_names[i], 0),
                    'beta': beta_hat.get(self.class_names[i], 0)
                }

            return class_hat
        else:
            raise ValueError("Beta fitting is only supported for a single metric at a time. Please specify a single metric.")


    def calc_mean(self, metric=None, calc_ci=True, ci=0.95, cil=None, ciu=None):
        """
        Calculate the mean of a metric.
        """

        if metric is None:
            if not hasattr(self, '_metric'):
                raise ValueError("No metric specified. Please specify a metric to calculate the mean.")
            metric = self._metric

        if len(metric) == 1:
            temp_metrics = self.metrics[self.metric_metadata[metric[0]]['data array']].mean(axis=0)
            # add class labels to the dictionary
            metric_dict = {self.class_names[i]: temp_metrics[i] for i in range(len(temp_metrics))}
            return metric_dict
        
        else:
            raise ValueError("Mean calculation is only supported for a single metric at a time. Please specify a single metric.")
    

    def calc_var(self, metric=None, ddof=1):
        """
        Calculate the variance of a metric.
        """
        if metric is None:
            if not hasattr(self, '_metric'):
                raise ValueError("No metric specified. Please specify a metric to calculate the mean.")
            metric = self._metric

        if len(metric) == 1:
            temp_metrics = self.metrics[self.metric_metadata[metric[0]]['data array']].var(ddof=ddof, axis=0)
            # add class labels to the dictionary
            metric_dict = {self.class_names[i]: temp_metrics[i] for i in range(len(temp_metrics))}

            return metric_dict
    
        else:
            raise ValueError("Variance calculation is only supported for a single metric at a time. Please specify a single metric.")
    

    def calc_std(self, metric: str = None, ddof: int = 1):
        """
        Calculate the standard deviation of a metric.
        """
        if metric is None:
            if not hasattr(self, '_metric'):
                raise ValueError("No metric specified. Please specify a metric to calculate the mean.")
            metric = self._metric

        if len(metric) == 1:
            temp_metrics = self.metrics[self.metric_metadata[metric[0]]['data array']].std(ddof=ddof, axis=0)
            # add class labels to the dictionary
            metric_dict = {self.class_names[i]: temp_metrics[i] for i in range(len(temp_metrics))}
            return metric_dict
        
        else:
            raise ValueError("Standard deviation calculation is only supported for a single metric at a time. Please specify a single metric.")


    def calc_median(self, metric: str = None):
        """
        Calculate the median of a metric.
        """
        if metric is None:
            if not hasattr(self, '_metric'):
                raise ValueError("No metric specified. Please specify a metric to calculate the mean.")
            metric = self._metric

        if len(metric) == 1:
            # calculate a temporary array for row medians
            temp_metrics = self.metrics[self.metric_metadata[metric[0]]['data array']].median(axis=0)
            # add class labels to the dictionary
            metric_dict = {self.class_names[i]: temp_metrics[i] for i in range(len(temp_metrics))}
            return metric_dict
        else:
            raise ValueError("Median calculation is only supported for a single metric at a time. Please specify a single metric.")


    def retrieve_metric_samples(self, metric):
        """
        Retrieve a metric from the metrics dictionary.
        """
        return self.metrics[self.metric_metadata[metric]['data array']]


    """ Posterior Distributions """

    def calculate_metric(self, metric, averaging=None):
        
        # check if all metrics should be calculated
        if metric == 'all':
            return self

        # check to ensure samples are present
        self.validate_sample_size()

        # make the metric a list if its a single string
        if isinstance(metric, str):
            metric = [metric]

        # check if the metric is valid
        for m in metric:
            if m not in self.metric_metadata:
                raise ValueError(f"Metric '{m}' is not defined in metric_metadata.")
        
        self._metric = metric
        
        for m in metric:

            # if a metric already exists, don't recalculate
            if self.metric_metadata[m]['data array'] in self.metrics:
                continue

            # calculate metric
            temp_arr = np.empty((len(self.posterior_samples), self.matrix_size))

            # rows
            for ii in range(0, self.sample_size):
                # columns
                for jj in range(0, self.matrix_size):
                    temp_arr[ii, jj] = self.metric_metadata[m]['function'](cm=self.posterior_samples[ii], j=jj)

            self.metrics[self.metric_metadata[m]['data array']] = temp_arr

        return self
    

    def calculate_ci(self, metric='accuracy', ci=0.95, cil=None, ciu=None):
        
        self.validate_metric(metric=metric)

        cil, ciu = self.ci_range(ci, cil, ciu)

        cis = {}

        metric_name = self.metric_metadata[metric]['data array']

        if self.class_names is not None:
            for jj in range(self.matrix_size):
                cis[self.class_names[jj]] = self.ci(arr=self.metrics[metric_name][:,jj],
                                                    cil=cil, ciu=ciu)
        else:
            for jj in range(self.matrix_size):
                cis[f'class_{jj}'] = self.ci(arr=self.metrics[metric_name][:,jj],
                                            cil=cil, ciu=ciu)
    
        return cis
    