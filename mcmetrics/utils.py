import pickle

class MCMetricsError(Exception):
    """Base class for all exceptions in this module."""
    pass

class InvalidMetricError(MCMetricsError, ValueError):
    """Exception raised when an invalid metric is encountered."""
    
    def __init__(self, metric_name):
        super().__init__(f"Invalid metric: {metric_name}")
        self.metric_name = metric_name


class InvalidConfusionMatrixError(Exception):
    """Exception raised for errors in the confusion matrix input."""
    
    def __init__(self, message="Confusion matrix must be a square 2D numpy array."):
        super().__init__(message)


class Utilities:

    """ Save and Load Functions """

    def save_class(self, file_path):
        """
        Save the current instance of the class to a file using pickle.

        Args:
            file_path (str): The path where the class instance should be saved.
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)


    def load_class(file_path):
        """
        Load a saved instance of the class from a file using pickle.

        Args:
            file_path (str): The path from where the class instance should be loaded.

        Returns:
            MCMetrics: The loaded instance of the MCMetrics class.
        """
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    """ Table Generation Functions """

    def generate_table(self, metrics, class_, sumstats, savepath, ci=0.95, cil=None, ciu=None):
        """
        Generates a table of metrics with credible intervals and summary statistics.

        Parameters
        ----------
        metrics : list
            List of metrics to include in the table.
        sumstats : str
            List of summary statistics to include in the table.
        savepath : str
            Path to save the table.
        ci : float, optional
            Credible interval level, by default 0.95.
        cil : float, optional
            Lower bound of the credible interval, by default None.
        ciu : float, optional
            Upper bound of the credible interval, by default None.
        """
        # check which class to use and check if it exists
        if class_ is None:
            class_ = self.class_names
        else:
            class_ = class_

        if isinstance(class_, list):
            for cls in class_:
                if cls not in self.class_names:
                    raise ValueError(f"Class '{cls}' not found in class names.")
        else:
            if class_ not in self.class_names:
                raise ValueError(f"Class '{class_}' not found in class names.")
        
        ### check if the metric has been calculated
        metrics = [metric.replace(' ', '_') for metric in metrics]
        calc_list = set(metrics) - set(list(self.metrics))

        if len(calc_list) > 0:
            for metric_ in calc_list:
                self.calculate_metric(metric=metric_)

        """
        Table Format:

                  | Metric 1 (95% CI) | Metric 2 (95% CI)
                  | Value | CIL | CIU | Value | CIL | CIU
        |---------|-------------------|------------------
        | Class 1 |       |     |     |       |     |
        |---------|-------------------|------------------
        | Class 2 |       |     |     |       |     |

        """

        results = {}

        # find CI range
        cil, ciu = self.ci_range(ci, cil, ciu)

        # calculate and save tables for each class (if applicable)
        for curr_class in class_:

            # Create a dictionary to store the results
            results[curr_class] = {}

            for metric in metrics:

                results[curr_class][metric] = {}

                cil_value, ciu_value = self.calculate_ci(metric=metric, cil=cil, ciu=ciu)[curr_class]
                sumstat_value = self.summary_stats[sumstats](metric=metric)[curr_class]

                results[curr_class][metric]['CIL'] = cil_value
                results[curr_class][metric]['CIU'] = ciu_value 
                results[curr_class][metric][sumstats] = sumstat_value

        # Create a .csv file with the results
        with open(savepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            metrics_list = ['' for i in range(len(metrics)*3)]

            metric_indx = 0
            for i in range(len(metrics_list)):
                if i % 3 == 0:
                    metrics_list[i] = f'{metrics[metric_indx].capitalize()}'
                    metric_indx += 1
                else:
                    pass

            header = ['Class'] + metrics_list
            stats_row = [f'{sumstats.capitalize()}'] + [f'CI ({np.round(cil*100, 3)}%)'] + [f'CI ({np.round(ciu*100, 3)}%)']

            second_row = ['']
            for i in range(len(metrics)):
                second_row = second_row + stats_row

            # Write header
            #header = ['Class'] + [f'{metric.capitalize()}, (CI{np.round(cil*100, 3)}%, CI{np.round(ciu*100, 3)}%)' for metric in metrics]
            writer.writerow(header)
            writer.writerow(second_row)

            # Write data
            for curr_class in class_:
                row = [curr_class]
                for metric in metrics:
                    value = results[curr_class][metric][sumstats]
                    cil_val_ = float(results[curr_class][metric]['CIL'])
                    ciu_val_ = float(results[curr_class][metric]['CIU'])
                    row.append(f'{value:.4f}')      # sumstat value
                    row.append(f'{cil_val_:.4f}')   # CIL
                    row.append(f'{ciu_val_:.4f}')   # CIU

                writer.writerow(row)