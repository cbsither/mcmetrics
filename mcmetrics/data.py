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
class Metadata:
    summary_stats: dict = {'mean': 'calc_mean',
                           'mode': 'calc_mode',
                           'median': 'calc_median',
                           'variance': 'calc_var',
                           'var': 'calc_var',
                           'standard deviation': 'calc_std',
                           'std': 'calc_std'}
    
    metric_metadata: dict = {'prevalence': {'function': 'prevalence',
                                            'data array': 'prevalence'},

                            'accuracy': {'function': 'accuracy',
                                            'data array': 'accuracy'},

                            'balanced accuracy': {'function': 'balanced_accuracy',
                                                    'data array': 'balanced_accuracy'},

                            'precision': {'function': 'precision',
                                            'data array': 'precision'},

                            'positive predictive value': {'function': 'positive_predictive_value',
                                            'data array': 'precision'}, # ppv is the same as precision

                            'false discovery rate': {'function': 'false_discovery_rate',
                                            'data array': 'false_discovery_rate'},

                            'f1 score': {'function': 'f1_score',
                                            'data array': 'f1_score'},

                            'false omission rate': {'function': 'false_omission_rate',
                                            'data array': 'false_omission_rate'},

                            'negative predictive value': {'function': 'negative_predictive_rate',
                                            'data array': 'negative_predictive_value'},

                            'fowlkes-mallows index': {'function': 'fowlkes_mallows_index',
                                            'data array': 'fowkes_mallows_index'},

                            'informedness': {'function': self.informedness,
                                            'data array': 'informedness'},

                            'sensitivity': {'function': 'sensitivity',
                                            'data array': 'sensitivity'}, # sensitivity is the same as tpr

                            'true positive rate': {'function': 'sensitivity',
                                            'data array': 'sensitivity'}, # tpr is the same as sensitivity

                            'recall': {'function': 'sensitivity',
                                            'data array': 'sensitivity'}, # recall is the same as sensitivity

                            'false positive rate': {'function': 'false_positive_rate',
                                            'data array': 'false_positive_rate'},

                            'positive likelihood ratio': {'function': 'positive_likelihood_ratio',
                                            'data array': 'positive_likelihood_ratio'},

                            'markedness': {'function': 'markedness',
                                            'data array': 'markedness'},

                            'true negative rate': {'function': 'specificity',
                                            'data array': 'specificity'}, # tnr is the same as specificity

                            'matthews correlation coefficient': {'function': 'matthews_correlation_coefficient',
                                            'data array': 'matthews_correlation_coefficient'},

                            'prevalence threshold': {'function': 'prevalence_threshold',
                                            'data array': 'prevalence_threshold'},

                            'false negative rate': {'function': 'false_negative_rate',
                                            'data array': 'false_negative_rate'},

                            'specificity': {'function': 'specificity',
                                            'data array': 'specificity'},

                            'negative likelihood ratio': {'function': 'negative_likelihood_ratio',
                                            'data array': 'negative_likelihood_ratio'},

                            'diagnostic odds ratio': {'function': 'diagnotic_odds_ratio',
                                            'data array': 'diagnotic_odds_ratio'},

                            'jaccard index': {'function': 'jaccard_index',
                                            'data array': 'jaccard_index'},

                            'critical success index': {'function': 'jaccard_index',
                                            'data array': 'jaccard_index'}, # same as jaccard index

                            'threat score': {'function': 'jaccard_index',
                                            'data array': 'jaccard_index'}, # same as jaccard index

                            'screening coefficient': {'function': 'screening_coefficient',
                                            'data array': 'screening_coefficient'},

                            'cohens kappa': {'function': 'cohens_kappa',
                                                'data array': 'cohens_kappa'}

                            }
