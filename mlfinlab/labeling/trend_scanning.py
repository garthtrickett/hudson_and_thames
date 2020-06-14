"""
Implementation of Trend-Scanning labels described in `Advances in Financial Machine Learning: Lecture 3/10
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_
"""

import pandas as pd
import numpy as np

from mlfinlab.structural_breaks.sadf import get_betas

import numba
from numba import njit, prange

import time


# with this set to true 547.9137012958527
@njit
def get_labels_numba(price_series_np_array, t_events_np_array_epoch,
                     look_forward_window, min_sample_length, step):
    t1_array_numba = np.zeros(
        len(t_events_np_array_epoch))  # Array of label end times
    t1_array_numba[:] = np.nan
    t_values_array_numba = np.zeros(
        len(t_events_np_array_epoch))  # Array of trend t-values
    t_values_array_numba[:] = np.nan
    for i in prange(len(t_events_np_array_epoch)):
        subset_np_array = price_series_np_array[i:i + look_forward_window]
        subset_np_array_epoch = t_events_np_array_epoch[i:i +
                                                        look_forward_window]
        if len(subset_np_array) >= look_forward_window:
            # Loop over possible look-ahead windows to get the one which yields maximum t values for b_1 regression coef
            max_abs_t_value = -np.inf  # Maximum abs t-value of b_1 coefficient among l values
            max_t_value_index = None  # Index with maximum t-value
            max_t_value = None  # Maximum t-value signed

            for forward_window in range(min_sample_length,
                                        len(subset_np_array), step):
                y_subset = subset_np_array[:forward_window].reshape(
                    -1, 1)  # y{t}:y_{t+l}

                y_subset_np_array_epoch = subset_np_array_epoch[:
                                                                forward_window]

                # Array of [1, 0], [1, 1], [1, 2], ... [1, l] # b_0, b_1 coefficients
                X_subset = np.ones((y_subset.shape[0], 2))
                X_subset[:, 1] = np.arange(y_subset.shape[0])

                # Get regression coefficients estimates
                # start = time.time()
                b_mean_, b_std_ = get_betas(X_subset, y_subset)
                # end = time.time()
                # print("get_betas time")
                # print(end - start)
                # Check if l gives the maximum t-value among all values {0...L}
                t_beta_1 = (b_mean_[1] / np.sqrt(b_std_[1, 1]))[0]
                if abs(t_beta_1) > max_abs_t_value:
                    max_abs_t_value = abs(t_beta_1)
                    max_t_value = t_beta_1
                    max_t_value_index = forward_window

            label_endtime_index = y_subset_np_array_epoch[max_t_value_index -
                                                          1]
            # import pdb
            # pdb.set_trace()
            t1_array_numba[i] = label_endtime_index
            t_values_array_numba[i] = max_t_value
        else:
            t1_array_numba[i] = np.nan
            t_values_array_numba[i] = np.nan
    return t1_array_numba, t_values_array_numba


def trend_scanning_labels(price_series: pd.Series,
                          t_events: list = None,
                          look_forward_window: int = 20,
                          min_sample_length: int = 5,
                          step: int = 1) -> pd.DataFrame:
    """
    `Trend scanning <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3257419>`_ is both a classification and
    regression labeling technique.

    That can be used in the following ways:

    1. Classification: By taking the sign of t-value for a given observation we can set {-1, 1} labels to define the
       trends as either downward or upward.
    2. Classification: By adding a minimum t-value threshold you can generate {-1, 0, 1} labels for downward, no-trend,
       upward.
    3. The t-values can be used as sample weights in classification problems.
    4. Regression: The t-values can be used in a regression setting to determine the magnitude of the trend.

    The output of this algorithm is a DataFrame with t1 (time stamp for the farthest observation), t-value, returns for
    the trend, and bin.

    :param price_series: (pd.Series) Close prices used to label the data set
    :param t_events: (list) Filtered events, array of pd.Timestamps
    :param look_forward_window: (int) Maximum look forward window used to get the trend value
    :param min_sample_length: (int) Minimum sample length used to fit regression
    :param step: (int) Optimal t-value index is searched every 'step' indices
    :return: (pd.DataFrame) Consists of t1, t-value, ret, bin (label information). t1 - label endtime, tvalue,
        ret - price change %, bin - label value based on price change sign
    """
    # pylint: disable=invalid-name

    if t_events is None:
        t_events = price_series.index

    ### My numba version

    price_series_np_array = np.asarray(price_series)
    t_events_np_array_epoch = np.asarray(t_events.astype(np.int64) // 1000000)

    start = time.time()
    t1_array_numba, t_values_array_numba = get_labels_numba(
        price_series_np_array, t_events_np_array_epoch, look_forward_window,
        min_sample_length, step)
    end = time.time()
    print("get_labels time")
    print(end - start)

    start = time.time()
    t1_array_numba = pd.to_datetime(t1_array_numba, unit="ms")

    labels_numba = pd.DataFrame(
        {
            't1': t1_array_numba,
            't_value': t_values_array_numba
        },
        index=t_events)
    labels_numba = labels_numba.dropna()
    labels_numba.loc[:, 'ret'] = price_series.loc[
        labels_numba.t1].values / price_series.loc[
            labels_numba.index].values - 1


    end = time.time()
    print("rest of script time")
    print(end - start)

    return labels_numba
