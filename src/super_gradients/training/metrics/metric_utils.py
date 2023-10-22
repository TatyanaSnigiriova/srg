import numpy as np
import torch
from torchmetrics import MetricCollection
from super_gradients.training.utils.utils import AverageMeter


def get_logging_values(loss_loggings: AverageMeter, metrics: MetricCollection, criterion=None):
    """
    @param loss_loggings: AverageMeter running average for the loss items
    @param metrics: MetricCollection object for running user specified metrics
    @param criterion the object loss_loggings average meter is monitoring, when set to None- only the metrics values are
    computed and returned.

    @return: tuple of the computed values
    """
    if criterion is not None:
        loss_loggingg_avg = loss_loggings.average
        if not isinstance(loss_loggingg_avg, tuple):
            loss_loggingg_avg = tuple([loss_loggingg_avg])
        logging_vals = loss_loggingg_avg + get_metrics_results_tuple(metrics)
    else:
        logging_vals = get_metrics_results_tuple(metrics)

    return logging_vals


def get_metrics_titles(metrics_collection: MetricCollection, experiment_name=''):
    """

    @param metrics_collection: MetricCollection object for running user specified metrics
    @return: list of all the names of the computed values list(str)
    """
    titles = []
    for metric_name, metric in metrics_collection.items():
        if metric_name == "additional_items":
            continue
        elif hasattr(metric, "component_names"):
            titles += [f'{experiment_name}__{component}' for component in metric.component_names] if experiment_name \
                else metric.component_names
        elif hasattr(metric, "reduction") and (
                (isinstance(metric.reduction, str) and metric.reduction.lower() == "none")
                or metric.reduction is None
            ):
            titles += [f'{experiment_name}__{metric_name}'] +\
                    [f'{experiment_name}__{metric_name}_{class_idx}' for class_idx in range(metric.num_classes)]\
                if experiment_name else [f'{metric_name}'] +\
                    [f'{metric_name}_{class_idx}' for class_idx in range(metric.num_classes)]

        else:
            titles.append(f'{experiment_name}__{metric_name}' if experiment_name else metric_name)

    return titles


def get_metrics_results_tuple(metrics_collection: MetricCollection):
    """

    @param metrics_collection: metrics collection of the user specified metrics
    @type metrics_collection
    @return: tuple of metrics values
    """
    if metrics_collection is None:
        results_tuple = ()
    else:
        results_tuple = tuple(flatten_metrics_dict(metrics_collection.compute()).values())
    return results_tuple


def flatten_metrics_dict(metrics_dict: dict):
    """
    :param metrics_dict - dictionary of metric values where values can also be dictionaries containing subvalues
    (in the case of compound metrics)

    @return: flattened dict of metric values i.e {metric1_name: metric1_value...}
    """
    flattened = {}
    for metric_name, metric_val in metrics_dict.items():
        if metric_name == "additional_items":
            continue
        # COLLECT ALL OF THE COMPONENTS IN THE CASE OF COMPOUND METRICS
        elif isinstance(metric_val, dict):
            for sub_metric_name, sub_metric_val in metric_val.items():
                if isinstance(sub_metric_val, torch.Tensor):
                    sub_metric_val = sub_metric_val.detach().cpu().numpy()
                    try:
                        flattened[sub_metric_name] = np.float32(sub_metric_val.mean().item())

                        for class_idx, class_val in enumerate(sub_metric_val):
                            flattened[sub_metric_name + f'_{class_idx}'] = np.float32(class_val.item())
                    except TypeError:
                        flattened[sub_metric_name] = np.float32(sub_metric_val.item())
                else:
                    flattened[sub_metric_name] = sub_metric_val
        else:
            if isinstance(metric_val, torch.Tensor):
                metric_val = metric_val.detach().cpu().numpy()
                try:
                    flattened[metric_name] = np.float32(metric_val.mean().item())

                    for class_idx, class_val in enumerate(metric_val):
                        flattened[metric_name + f'_{class_idx}'] = np.float32(class_val.item())

                except TypeError:
                    flattened[metric_name] = np.float32(metric_val.item())
            else:
                flattened[metric_name] = metric_val
    return flattened


def get_metrics_dict(metrics_tuple, metrics_collection, loss_logging_item_names, experiment_name=''):
    """
    Returns a dictionary with the epoch results as values and their names as keys.
    @param metrics_tuple: the result tuple
    @param metrics_collection: MetricsCollection
    @param loss_logging_item_names: loss component's names.
    @return: dict
    """
    keys = loss_logging_item_names + get_metrics_titles(metrics_collection, experiment_name)
    metrics_dict = dict(zip(keys, list(metrics_tuple)))
    return metrics_dict


def get_train_loop_description_dict(metrics_tuple, metrics_collection, loss_logging_item_names, experiment_name='',
                                    **log_items):
    """
    Returns a dictionary with the epoch's logging items as values and their names as keys, with the purpose of
     passing it as a description to tqdm's progress bar.

    @param metrics_tuple: the result tuple
    @param metrics_collection: MetricsCollection
    @param loss_logging_item_names: loss component's names.
    @param log_items additional logging items to be rendered.
    @return: dict
    """
    log_items.update(get_metrics_dict(metrics_tuple, metrics_collection, loss_logging_item_names, experiment_name))
    # Tensor values have been processed before in flatten_metrics_dict

    return log_items
