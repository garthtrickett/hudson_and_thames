"""
Logic regarding labeling from chapter 3. In particular the Triple Barrier Method and Meta-Labeling.
"""

import numpy as np
np.set_printoptions(suppress=True)  # don't use scientific notati
import time
import pandas as pd
import sys
import os
from itertools import chain
from pathlib import Path

from numba import njit, prange

home = str(Path.home())
sys.path.append(home + "/ProdigyAI/third_party_libraries/hudson_and_thames")

# Snippet 3.1, page 44, Daily Volatility Estimates
from mlfinlab.util.multiprocess import mp_pandas_obj


@njit  # prange i loop ?
def get_events_subloop_numba(events_t1_index_as_epoch, events_t1_as_epoch,
                             close_np_array, close_index_as_epoch,
                             out_epoch_index, side_np_array,
                             stop_loss_np_array, profit_taking_np_array,
                             out_sl, out_pt):
    for i in range(len(events_t1_index_as_epoch)):
        path_prices_for_given_trade = np.zeros(len(close_np_array))
        path_prices_for_given_trade[:] = np.nan
        path_prices_for_given_trade_index = np.zeros(len(close_np_array))
        path_prices_for_given_trade_index[:] = np.nan
        for j in range(len(close_index_as_epoch)):

            if close_index_as_epoch[j] >= events_t1_index_as_epoch[
                    i] and close_index_as_epoch[j] <= events_t1_as_epoch[i]:
                path_prices_for_given_trade_index[j] = close_index_as_epoch[j]
                path_prices_for_given_trade[j] = close_np_array[j]

            if close_index_as_epoch[j] == events_t1_index_as_epoch[i]:
                loc_index_in_epoch = j

        path_prices_for_given_trade_index = path_prices_for_given_trade_index[
            ~np.isnan(path_prices_for_given_trade_index)]
        path_prices_for_given_trade = path_prices_for_given_trade[
            ~np.isnan(path_prices_for_given_trade)]

        cum_returns = (
            path_prices_for_given_trade / close_np_array[loc_index_in_epoch] -
            1) * side_np_array[i]  # Path returns

        for k in range(len(out_epoch_index)):
            if out_epoch_index[k] == events_t1_index_as_epoch[i]:
                out_loc = k

        for l in range(len(cum_returns)):
            if cum_returns[l] < stop_loss_np_array[out_loc]:
                out_sl[out_loc] = path_prices_for_given_trade_index[l]
                break
            if cum_returns[l] > profit_taking_np_array[out_loc]:
                out_pt[out_loc] = path_prices_for_given_trade_index[l]
                break

    return out_sl, out_pt


# Snippet 3.2, page 45, Triple Barrier Labeling Method
def apply_pt_sl_on_t1(close, events, pt_sl, molecule):  # pragma: no cover
    """
    Snippet 3.2, page 45, Triple Barrier Labeling Method

    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.

    :param close: (series) close prices
    :param events: (series) of indices that signify "events" (see cusum_filter function
    for more details)
    :param pt_sl: (array) element 0, indicates the profit taking level; element 1 is stop loss level
    :param molecule: (an array) a set of datetime index values for processing
    :return: DataFrame of timestamps of when first barrier was touched
    """
    # Apply stop loss/profit taking, if it takes place before t1 (end of event)

    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)
    numba_out = events_[["t1"]].copy(deep=True)

    profit_taking_multiple = pt_sl[0]
    stop_loss_multiple = pt_sl[1]

    # Profit taking active
    if profit_taking_multiple > 0:
        profit_taking = profit_taking_multiple * events_["trgt"]
    else:
        profit_taking = pd.Series(index=events.index)  # NaNs

    # Stop loss active
    if stop_loss_multiple > 0:
        stop_loss = -stop_loss_multiple * events_["trgt"]
    else:
        stop_loss = pd.Series(index=events.index)  # NaNs

    # print("starting sub loop prep")

    start_time = time.time()

    out_epoch_index = np.asarray(out.index.astype(np.int64)) // 1000000

    stop_loss_np_array = np.asarray(stop_loss)
    profit_taking_np_array = np.asarray(profit_taking)

    out_sl = np.zeros(len(out_epoch_index))
    out_sl[:] = np.nan
    out_pt = np.zeros(len(out_epoch_index))
    out_pt[:] = np.nan

    events_t1_fill_na_with_last_close_index = events_["t1"].fillna(
        close.index[-1])

    events_side_fill_na_with_last_close_index = events_["side"].fillna(
        close.index[-1])

    side_np_array = np.asarray(events_side_fill_na_with_last_close_index)

    events_t1_index_as_epoch = np.asarray(
        events_t1_fill_na_with_last_close_index.index.astype(
            np.int64)) // 1000000
    events_t1_as_epoch = np.asarray(
        events_t1_fill_na_with_last_close_index.astype(np.int64)) // 1000000

    close_index_as_epoch = np.asarray(close.index.astype(np.int64)) // 1000000
    close_np_array = np.asarray(close)

    # print("finished sub loop prep")

    # print("starting subloop in apply_pt_sl")

    out_sl, out_pt = get_events_subloop_numba(
        events_t1_index_as_epoch, events_t1_as_epoch, close_np_array,
        close_index_as_epoch, out_epoch_index, side_np_array,
        stop_loss_np_array, profit_taking_np_array, out_sl, out_pt)

    numba_out["sl"] = pd.to_datetime(out_sl, unit="ms")
    numba_out["pt"] = pd.to_datetime(out_pt, unit="ms")

    # Get events original code
    # for loc, vertical_barrier in events_["t1"].fillna(
    #         close.index[-1]).iteritems():
    #     closing_prices = close[
    #         loc:vertical_barrier]  # Path prices for a given trade

    #     cum_returns = (closing_prices / close[loc] -
    #                    1) * events_.at[loc, "side"]  # Path returns-
    #     out.loc[loc,
    #             "sl"] = cum_returns[cum_returns < stop_loss[loc]].index.min(
    #             )  # Earliest stop loss date
    #     out.loc[loc, "pt"] = cum_returns[
    #         cum_returns > profit_taking[loc]].index.min(
    #         )  # Earliest profit taking date

    end_time = time.time()

    print("finished subloop in apply_pt_sl" + str(end_time - start_time))

    return numba_out


# Snippet 3.4 page 49, Adding a Vertical Barrier
def add_vertical_barrier(t_events,
                         close,
                         num_days=0,
                         num_hours=0,
                         num_minutes=0,
                         num_seconds=0):
    """
    Snippet 3.4 page 49, Adding a Vertical Barrier

    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    :param t_events: (series) series of events (symmetric CUSUM filter)
    :param close: (series) close prices
    :param num_days: (int) number of days to add for vertical barrier
    :param num_hours: (int) number of hours to add for vertical barrier
    :param num_minutes: (int) number of minutes to add for vertical barrier
    :param num_seconds: (int) number of seconds to add for vertical barrier
    :return: (series) timestamps of vertical barriers
    """
    timedelta = pd.Timedelta(
        "{} days, {} hours, {} minutes, {} seconds".format(
            num_days, num_hours, num_minutes, num_seconds))
    # Find index to closest to vertical barrier
    nearest_index = close.index.searchsorted(t_events + timedelta)

    # Exclude indexes which are outside the range of close price index
    nearest_index = nearest_index[nearest_index < close.shape[0]]

    # Find price index closest to vertical barrier time stamp
    nearest_timestamp = close.index[nearest_index]
    filtered_events = t_events[:nearest_index.shape[0]]

    vertical_barriers = pd.Series(data=nearest_timestamp,
                                  index=filtered_events)
    return vertical_barriers


# Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
def get_events(close,
               t_events,
               pt_sl,
               target,
               min_ret,
               num_threads,
               vertical_barrier_times=False,
               side_prediction=None,
               split_by=10000):
    """
    Snippet 3.6 page 50, Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    :param close: (series) Close prices
    :param t_events: (series) of t_events. These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        Eg: CUSUM Filter
    :param pt_sl: (2 element array) element 0, indicates the profit taking level; element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    :param target: (series) of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (series) A pandas series with the timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    :param side_prediction: (series) Side of the bet (long/short) as decided by the primary model
    :return: (data frame) of events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['trgt'] is event's target
            -events['side'] (optional) implies the algo's position side
            -events['pt'] Profit taking multiple
            -events['sl'] Stop loss multiple
    """

    # 1) Get target
    target = target.reindex(t_events)
    # target = target[target > min_ret]  # min_ret ### this isnt commented out in the original repo

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

    # 3) Form events object, apply stop loss on vertical barrier
    if side_prediction is None:
        side_ = pd.Series(1.0, index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side_prediction.reindex(
            target.index)  # Subset side_prediction on target index.
        pt_sl_ = pt_sl[:2]

    # Create a new df with [v_barrier, target, side] and drop rows that are NA in target
    events = pd.concat(
        {
            "t1": vertical_barrier_times,
            "trgt": target,
            "side": side_
        }, axis=1)
    events = events.dropna(subset=["trgt"])

    print("apply_pt_sl_on_t1 loop started")

    start_time = time.time()

    first_touch_dates_list = []
    padding = (len(close) - len(events)) * 2

    number_of_splits = len(events.index) // split_by
    for i in range(number_of_splits):
        if i == 0:
            print("split_nuber started " + str(i))

            events_sub_index = events.index[:(i + 1) * split_by]
            events_sub = events.loc[events_sub_index]
            close_sub = close.iloc[:((i + 1) * split_by) + (padding * 2)]

        elif i < number_of_splits - 1:
            print("split_nuber started " + str(i))
            events_sub_index = events.index[i * split_by:(i + 1) * split_by]
            events_sub = events.loc[events_sub_index]
            close_sub = close.iloc[(i * split_by) -
                                   (padding * 2):((i + 1) * split_by) +
                                   (padding * 2)]

        elif i == number_of_splits - 1:
            print("last")
            print("split_nuber started " + str(i))
            events_sub_index = events.index[i * split_by:]
            events_sub = events.loc[events_sub_index]
            close_sub = close.iloc[(i * split_by) - (padding * 2):]

        # Apply Triple Barrier
        first_touch_dates = mp_pandas_obj(
            func=apply_pt_sl_on_t1,
            pd_obj=("molecule", events_sub_index),
            num_threads=num_threads,
            close=close_sub,
            events=events_sub,
            pt_sl=pt_sl_,
        )

        first_touch_dates_list.append(first_touch_dates)

        print("split_nuber finished " + str(i))

    end_time = time.time()

    print("apply_pt_sl_on_t1 loop finished " + str(end_time - start_time))

    start_time = time.time()
    first_touch_dates = pd.concat(first_touch_dates_list)

    end_time = time.time()

    print("pd concat time " + str(end_time - start_time))

    print("apply_pt_sl_on_t1 finished")

    events_t1_epoch_from_first_touch_dates_list = []
    padding = (len(close) - len(events)) * 2

    number_of_splits = len(events.index) // split_by
    for i in range(number_of_splits):
        if i == 0:
            start_index = None  # 0
            end_index = (i + 1) * split_by
        elif i < number_of_splits - 1:
            start_index = i * split_by
            end_index = (i + 1) * split_by
        elif i == number_of_splits - 1:
            start_index = i * split_by
            end_index = None  # -1

        print("fill_events_t1_with_first_touches pre work started")
        start_time = time.time()

        events_index_as_epoch = np.asarray(
            events.index[start_index:end_index].astype(np.int64))
        events_t1 = events.reindex(events.t1)
        events_t1_as_epoch = np.asarray(
            events_t1.index[start_index:end_index].astype(np.int64))

        first_touch_dates_index_as_epoch = np.asarray(
            first_touch_dates.index[start_index:end_index].astype(np.int64))
        first_touch_dates_t1 = first_touch_dates.reindex(first_touch_dates.t1)
        first_touch_dates_t1_as_epoch = np.asarray(
            first_touch_dates_t1.index[start_index:end_index].astype(np.int64))

        first_touch_dates_pt = first_touch_dates.reindex(first_touch_dates.pt)
        first_touch_dates_pt_as_epoch = np.asarray(
            first_touch_dates_pt.index[start_index:end_index].astype(np.int64))

        first_touch_dates_sl = first_touch_dates.reindex(first_touch_dates.sl)
        first_touch_dates_sl_as_epoch = np.asarray(
            first_touch_dates_sl.index[start_index:end_index].astype(np.int64))

        end_time = time.time()
        print("fill_events_t1_with_first_touches pre work finished" +
              str(end_time - start_time))

        print("fill_events_t1_with_first_touches numba started")
        start_time = time.time()

        events_t1_epoch_from_first_touch_dates = fill_events_t1_with_first_touches(
            events_index_as_epoch,
            first_touch_dates_index_as_epoch,
            events_t1_as_epoch,
            first_touch_dates_t1_as_epoch,
            first_touch_dates_pt_as_epoch,
            first_touch_dates_sl_as_epoch,
        )

        events_t1_epoch_from_first_touch_dates_list.append(
            events_t1_epoch_from_first_touch_dates)

        end_time = time.time()
        print("fill_events_t1_with_first_touches numba finished" +
              str(end_time - start_time))

    print(len(events_t1_epoch_from_first_touch_dates_list))

    for i in range(len(events_t1_epoch_from_first_touch_dates_list)):
        print(i)
        if i == 0:
            appended_events_t1_epoch_from_first_touch_dates = np.append(
                events_t1_epoch_from_first_touch_dates_list[0],
                events_t1_epoch_from_first_touch_dates_list[1])
        elif i < len(events_t1_epoch_from_first_touch_dates_list) - 1:
            appended_events_t1_epoch_from_first_touch_dates = np.append(
                appended_events_t1_epoch_from_first_touch_dates,
                events_t1_epoch_from_first_touch_dates_list[i + 1])

    # for ind in events.index:
    #     # find the index  where index == ind then update t1
    #     events.loc[ind, "t1"] = first_touch_dates.loc[ind, :].dropna().min()

    if side_prediction is None:
        events = events.drop("side", axis=1)

    # Add profit taking and stop loss multiples for vertical barrier calculations
    events["pt"] = pt_sl[0]
    events["sl"] = pt_sl[1]

    events.t1 = pd.to_datetime(appended_events_t1_epoch_from_first_touch_dates)

    return events


@njit(parallel=True)
def fill_events_t1_with_first_touches(
    events_index_as_epoch,
    first_touch_dates_index_as_epoch,
    events_t1_as_epoch,
    first_touch_dates_t1_as_epoch,
    first_touch_dates_pt_as_epoch,
    first_touch_dates_sl_as_epoch,
):
    for i in prange(len(events_index_as_epoch)):
        for j in prange(len(first_touch_dates_index_as_epoch)):
            if events_index_as_epoch[i] >= first_touch_dates_index_as_epoch[j]:

                if (first_touch_dates_pt_as_epoch[j] < 0
                        and first_touch_dates_sl_as_epoch[j] < 0):
                    events_t1_as_epoch[i] = first_touch_dates_t1_as_epoch[j]
                elif (first_touch_dates_pt_as_epoch[j] > 0
                      and first_touch_dates_sl_as_epoch[j] > 0):
                    if (first_touch_dates_pt_as_epoch[j] <
                            first_touch_dates_sl_as_epoch[j]):
                        events_t1_as_epoch[i] = first_touch_dates_pt_as_epoch[
                            j]
                    elif (first_touch_dates_pt_as_epoch[j] >
                          first_touch_dates_sl_as_epoch[j]):
                        events_t1_as_epoch[i] = first_touch_dates_sl_as_epoch[
                            j]

                elif (first_touch_dates_pt_as_epoch[j] < 0
                      and first_touch_dates_sl_as_epoch[j] > 0):
                    if (first_touch_dates_sl_as_epoch[j] <
                            first_touch_dates_t1_as_epoch[j]):
                        events_t1_as_epoch[i] = first_touch_dates_sl_as_epoch[
                            j]
                    else:
                        events_t1_as_epoch[i] = first_touch_dates_t1_as_epoch[
                            j]
                elif (first_touch_dates_pt_as_epoch[j] > 0
                      and first_touch_dates_sl_as_epoch[j] < 0):
                    if (first_touch_dates_pt_as_epoch[j] <
                            first_touch_dates_t1_as_epoch[j]):
                        events_t1_as_epoch[i] = first_touch_dates_pt_as_epoch[
                            j]
                    else:
                        events_t1_as_epoch[i] = first_touch_dates_t1_as_epoch[
                            j]

    return events_t1_as_epoch


@njit(parallel=True)
def barrier_touched_numba(out_df_ret_as_array, out_df_trgt_as_array,
                          events_pt_as_array, events_sl_as_array):
    store = np.zeros(len(out_df_ret_as_array))

    for i in prange(len(out_df_ret_as_array)):
        ret = out_df_ret_as_array[i]
        target = out_df_trgt_as_array[i]

        pt_level_reached = ret > np.log(1 + target) * events_pt_as_array[i]
        sl_level_reached = ret < -np.log(1 + target) * events_sl_as_array[i]

        # for date_time, values in out_df.iterrows():
        #     ret = values["ret"]
        #     target = values["trgt"]

        #     pt_level_reached = ret > np.log(1 + target) * events.loc[date_time,
        #                                                              'pt']
        #     sl_level_reached = ret < -np.log(1 + target) * events.loc[date_time,
        #                                                               'sl']
        if ret > 0.0 and pt_level_reached:
            # Top barrier reached
            store[i] = 1
        elif ret < 0.0 and sl_level_reached:
            # Bottom barrier reached
            store[i] = -1
        else:
            # Vertical barrier reached
            store[i] = 0

    return store


# Snippet 3.9, pg 55, Question 3.3
def barrier_touched(out_df, events):
    """
    Snippet 3.9, pg 55, Question 3.3
    Adjust the getBins function (Snippet 3.7) to return a 0 whenever the vertical barrier is the one touched first.

    Top horizontal barrier: 1
    Bottom horizontal barrier: -1
    Vertical barrier: 0

    :param out_df: (DataFrame) containing the returns and target
    :param events: (DataFrame) The original events data frame. Contains the pt sl multiples needed here.
    :return: (DataFrame) containing returns, target, and labels
    """

    out_df_ret_as_array = out_df.ret.values
    out_df_trgt_as_array = out_df.trgt.values
    events_pt_as_array = events.pt.values
    events_sl_as_array = events.sl.values

    store = barrier_touched_numba(out_df_ret_as_array, out_df_trgt_as_array,
                                  events_pt_as_array, events_sl_as_array)

    # Save to 'bin' column and return
    out_df["bin"] = store
    return out_df


# Snippet 3.4 -> 3.7, page 51, Labeling for Side & Size with Meta Labels
def get_bins(triple_barrier_events, close):
    """
    Snippet 3.7, page 51, Labeling for Side & Size with Meta Labels

    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:

    Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
    a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
    The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
    to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

    :param triple_barrier_events: (data frame)
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
                Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    :param close: (series) close prices
    :return: (data frame) of meta-labeled events
    """

    import time
    start_time = time.time()

    # 1) Align prices with their respective events
    events_ = triple_barrier_events.dropna(subset=["t1"])
    all_dates = events_.index.union(
        other=events_["t1"].array).drop_duplicates()
    prices = close.reindex(all_dates, method="bfill")

    # 2) Create out DataFrame
    out_df = pd.DataFrame(index=events_.index)
    # Need to take the log returns, else your results will be skewed for short positions
    out_df["ret"] = np.log(prices.loc[events_["t1"].array].array) - np.log(
        prices.loc[events_.index])
    out_df["trgt"] = events_["trgt"]

    # Meta labeling: Events that were correct will have pos returns
    if "side" in events_:
        out_df["ret"] = out_df["ret"] * events_["side"]  # meta-labeling

    end_time = time.time()

    print("p1=" + str(end_time - start_time))

    # Added code: label 0 when vertical barrier reached
    out_df = barrier_touched(out_df, triple_barrier_events)

    end_time = time.time()

    print("p2=" + str(end_time - start_time))

    start_time = time.time()

    # Meta labeling: label incorrect events with a 0
    if "side" in events_:
        out_df.loc[out_df["ret"] <= 0, "bin"] = 0

    # Transform the log returns back to normal returns.
    out_df["ret"] = np.exp(out_df["ret"]) - 1

    # Add the side to the output. This is useful for when a meta label model must be fit
    tb_cols = triple_barrier_events.columns
    if "side" in tb_cols:
        out_df["side"] = triple_barrier_events["side"]

    end_time = time.time()

    print("p3=" + str(end_time - start_time))

    return out_df


# Snippet 3.8 page 54
def drop_labels(events, min_pct=0.05):
    """
    Snippet 3.8 page 54

    This function recursively eliminates rare observations.

    :param events: (data frame) events.
    :param min_pct: (float) a fraction used to decide if the observation occurs less than that fraction.
    :return: (data frame) of events.
    """
    # Apply weights, drop labels with insufficient examples
    while True:
        df0 = events["bin"].value_counts(normalize=True)

        if df0.min() > min_pct or df0.shape[0] < 3:
            break

        print("dropped label: ", df0.idxmin(), df0.min())
        events = events[events["bin"] != df0.idxmin()]

    return events
