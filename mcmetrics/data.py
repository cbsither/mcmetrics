from dataclasses import dataclass
import numpy as np

@dataclass
class Samples:
    pass

@dataclass
class Tables:
    pass


@dataclass
class Plots:
    pass

@dataclass
class Metrics:
    summary_stats: dict = {'mean': 'calc_mean',
                           'mode': 'calc_mode',
                           'median': 'calc_median',
                           'variance': 'calc_var',
                           'var': 'calc_var',
                           'standard deviation': 'calc_std',
                           'std': 'calc_std'}
    
    metric_metadata: dict = {'prevalence': {'function': self.prev,
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
