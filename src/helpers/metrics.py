from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, f1_score

# Base class for metrics
class Metric(ABC):
    @abstractmethod
    def compute(self, true_labels_matrix, pred_labels_matrix):
        """Compute the given metrics on the reference labels in true_labels_matrix
        and predicted labels in pr"ed_labels_matrix. 

        Args:
            true_labels_matrix (List[List[int 0, int 1, int 2]]): reference labels.
            pred_labels_matrix (List[List[int 0, int 1, int 2]]): predicted labels.
        """
        pass

class HammingLossMetric(Metric):
    def compute(self, true_labels_matrix, pred_labels_matrix):
        accuracies = [] # lists accuracies for each review.
        for t_labels, labels in zip(true_labels_matrix, pred_labels_matrix):
            accuracies.append(1 - hamming_loss(t_labels, labels))
        return np.mean(accuracies)

class SubsetAccuracyMetric(Metric):
    def compute(self, true_labels_matrix, pred_labels_matrix):
        subset_accuracies = []
        for t_labels, labels in zip(true_labels_matrix, pred_labels_matrix):
            subset_accuracies.append(t_labels == labels)
        return np.mean(subset_accuracies)

class F1MacroMetric(Metric):
    def compute(self, true_labels_matrix, pred_labels_matrix):
        f1s = []
        t_matrix, p_matrix = np.array(true_labels_matrix), np.array(pred_labels_matrix)
        for j in range(t_matrix.shape[1]):
            f1s.append(f1_score(t_matrix[:, j], p_matrix[:, j], average='macro'))
        return np.mean(f1s)

class MetricFactory:
    _metrics = {
        "hamming_loss": HammingLossMetric,
        "subset_accuracy": SubsetAccuracyMetric,
        "f1_macro": F1MacroMetric,
    }

    @classmethod
    def get_metric(cls, metric_name):
        if metric_name not in cls._metrics:
            raise ValueError(f"Metric '{metric_name}' not recognized. Available metrics: {list(cls._metrics.keys())}")
        return cls._metrics[metric_name]()
    
def evaluate(y_true, y_pred, metric_name):
    metric = MetricFactory.get_metric(metric_name)
    return metric.compute(y_true, y_pred)