import numpy as np

class Metrics:
    """
    Functions for all metrics.
    """

    def TP(self, cm, j):
        """
        Returns the true positives for a given class j in a square matrix size m.
        """
        return cm[j,j]
    
    def FP(self, cm, j):
        """
        Returns the false positives for a given class j in a square matrix size m.
        """
        return cm[:, j].sum() - self.TP(cm, j)

    def FN(self, cm, j):
        """
        Returns the false negatives for a given class j in a square matrix size m.
        """
        return cm[j, :].sum() - self.TP(cm, j)

    def TN(self, cm, j):
        """
        Returns the true negatives for a given class j in a square matrix size m.
        """
        return cm.sum() - (self.TP(cm, j) + self.FP(cm, j) + self.FN(cm, j))

    def N_(self, cm, j):
        return self.FN(cm, j) + self.TN(cm , j)
    
    def P_(self, cm, j):
        return self.TP(cm, j) + self.FP(cm , j) 
    
    def class_proportions(self):
        return self.cm.sum(axis=1) / self.cm.sum()
    

    """ Metric functions """

    def prevalence(self, cm, j):
            """
            The prevalence here is the conditional prevalence for class j given
            the entire dataset. 

            *IMPORTANT*: This is not the same as the population prevalence. Rather,
            it is contingent on the sampling method/bias used to create the dataset.

            Definition:
                Prevalance = P_j / (P + N)

                *the occurence of positives for class j divided by all
                occurences.
            """
            p_i = self.P_(cm, j)
            n_i = self.N_(cm, j)
            return p_i / (p_i + n_i)
    

    def accuracy(self, cm, j):
        """
        Accuracy

        Definition:
            Accuracy = (TP + TN) / (P + N)
        """
        return (self.TP(cm, j)+self.TN(cm, j)) / (self.P_(cm, j)+self.N_(cm, j))

    def balanced_accuracy(self, cm, j):
        """
        Balanced Accuracy

        Definition:
            Bal. Accuracy = (TPR + TNR) / 2
        """
        return (self.tpr(cm, j) + self.tnr(cm, j)) / 2

    def precision(self, cm, j):
        """
        Precision or Positive Predictive Value

        Definition:
            Precision = TP / (TP + FP)

            or:

            Precision = 1 - False Discovery Rate
        """
        return self.TP(cm, j) / (self.TP(cm, j) + self.FP(cm, j))
    
    def positive_predictive_value(self, cm, j):
        """
        Positive Predictive Value

        Definition:
            See 'Precision'
        """
        return self.prec(cm, j)
    
    def false_discovery_rate(self, cm, j):
        """
        False Discovery Rate

        Definition:

            FDR = FP / (TP + FP) = 1 - PPV

        """
        return 1 - self.ppv(cm, j)

    def f1_score(self, cm, j):
        return (2*self.TP(cm, j)) / (2*self.TP(cm, j)+self.FP(cm, j)+self.FN(cm, j))

    def false_omission_rate(self, cm, j):
        return self.FN(cm, j) / (self.TN(cm, j)+self.FN(cm, j))
    
    def negative_predictive_rate(self, cm, j):
        return self.TN(cm, j) / (self.TN(cm, j)+self.FN(cm, j))
    
    def fowlkes_mallows_index(self, cm, j):
        return np.sqrt(self.ppv(cm, j) * self.tpr(cm , j))

    def sensitivity(self, cm, j):
        """
        Sensitivity (or true positive rate)

        Probability of detection; hit rate; power

        Definition:
            Sensitivity = TP / P

            or:

            Sensitivity = 1 - False Negative Rate
        """
        return self.TP(cm, j) / self.P_(cm, j)

    def specificity(self, cm, j):
        return self.TN(cm, j) / self.N_(cm, j)

    def true_positive_rate(self, cm, j):
        return self.sens(cm, j)

    def false_positive_rate(self, cm, j):
        return self.FP(cm, j) / self.N_(cm, j)

    def false_negative_rate(self, cm, j):
        return self.FN(cm, j) / self.P_(cm, j)

    def true_negative_rate(self, cm, j):
        return self.spec(cm, j)

    def positive_likelihood_ratio(self, cm, j):
        return self.tpr(cm, j) / self.fpr(cm, j)
    
    def negative_likelihood_ratio(self, cm, j):
        return self.fnr(cm, j) / self.tnr(cm, j)
    
    def diagnostic_odds_ratio(self, cm, j):
        return self.pos_LR(cm, j) / self.neg_LR(cm, j)
    
    def markedness(self, cm, j):
        return self.ppv(cm, j) + self.npv(cm, j) - 1
    
    def informedness(self, cm, j):
        return self.tpr(cm, j) + self.tnr(cm, j) - 1
    
    def matthews_correlation_coefficient(self, cm, j):
        """
        Matthews correlation coefficient

        Measures the quality of binary classifications
        """
        numerator_ = np.sqrt(self.tpr(cm, j) * self.tnr(cm, j) * self.ppv(cm, j) * self.npv(cm, j))
        demoninator_ = np.sqrt(self.fnr(cm, j) * self.fpr(cm, j) * self.for_(cm, j) * self.fdr(cm, j))
        return numerator_ - demoninator_
    
    def prevalence_threshold(self, cm, j):
        sens_ = self.sens(cm, j)
        spec_ = self.spec(cm, j)

        numerator_ = (np.sqrt(sens_ * (-spec_ + 1)) + spec_ - 1)
        demoninator_ = sens_ + spec_ - 1
        if demoninator_ == 0:
            return np.nan
        else:
            return numerator_ / demoninator_

    def prevalence_threshold_point_est(self, sens_, spec_):
        return (np.sqrt(sens_*(-spec_+1))+spec_-1) / (sens_+spec_-1)

    def _prevalence_threshold_old(self, cm, j):
        """
        Same as from Balayla (2020) "Prevalence threshold (phi_e) and the geometry of screening curves" 
        but using fpr and tpr rather than sensitivity and specificity.

        ***Not the same as from Balayla (2020) when classes are imbalanced. Retired due to this reason.
        """

        a_ = np.sqrt(self.tpr(cm, j) * self.fpr(cm, j)) - self.fpr(cm, j)
        b_ = self.tpr(cm, j) - self.fpr(cm, j)

        if b_ == 0:
            return np.nan
        else:
            return a_ / b_

    def screening_coefficient(self, cm, j):
        """
        From: Balayla (2020): Prevalence threshold (phi_e) and the geometry of screening curves
        
        The screening coefficient is the sum of sensitivities and specificites and can exist between
        0 and 2.
        """
        return self.sens(cm, j) + self.spec(cm, j)

    def threat_score(self, cm, j):
        return self.TP(cm, j) / (self.TP(cm, j) + self.FN(cm, j) + self.FP(cm, j))
    
    def jaccard_index(self, cm, j):
        return self.ts(cm, j)

    def critical_success_index(self, cm, j):
        return self.ts(cm, j)
    
    def bayesian_positive_predictive_value(self, sens, spec, prior, use_cm=False):
        """
        Bayesian Positive Predictive Value (PPV)

        Parameters:
        sens (float): Sensitivity of the test.
        spec (float): Specificity of the test.
        prior (float): Prior probability of the condition being true.

        Returns:
        float: Bayesian PPV.

        The `use_cm` parameter will use the prevalence derived from the confusion matrix
        if set to True, otherwise it will use the specified prior probability.
        """
        a_ = sens
        b_ = spec
        if use_cm == True:
            c_ = self.prev(self.cm, j)
        else:
            c_ = prior
        return (a_*c_) / ((a_*c_)+(b_*c_)-b_-c_+1)

    def binomial_me(self, z_, p_, me_):
        return (z_**2 * p_ * (1 - p_)) / (me_)**2

    def cohens_kappa(self, cm, j):
        """
        From: https://en.wikipedia.org/wiki/Cohen%27s_kappa
        """
        return (2 * (self.TP(cm, j) * self.TN(cm, j) - self.FN(cm, j) * self.FP(cm, j))) / \
               ((self.TP(cm, j) + self.FP(cm, j)) * (self.FP(cm, j) + self.TN(cm, j)) + (self.TP(cm, j) + self.FN(cm, j)) * (self.FN(cm, j) + self.TN(cm, j)))



    