import numpy as np
from typing import Optional


def get_unit_rates(
    unit_idx,
    units_spk_time,
    units_spk_time_index,
    alignment_times,
    evoked_window,
    pre_alignment_window
):
    spk_idx_lb = 0 if unit_idx == 0 else units_spk_time_index[unit_idx-1]
    spks = units_spk_time[spk_idx_lb:units_spk_time_index[unit_idx]]
    evoked_spk_rate = []
    prestim_spk_rate = []
    for alignment_time in alignment_times:
        evoked_spk_rate.append(
            sum(
                (spks >= alignment_time + evoked_window[0]) & (spks < alignment_time + evoked_window[1])
            ) / (evoked_window[1] - evoked_window[0])
        )
        prestim_spk_rate.append(
            sum(
                (spks >= alignment_time - pre_alignment_window[1]) & (spks < alignment_time - pre_alignment_window[0])
            ) / (pre_alignment_window[1] - pre_alignment_window[0])
        )
    return evoked_spk_rate, prestim_spk_rate


def get_unit_stats(
    evoked_spk_rate,
    prestim_spk_rate
):
    valid_trials = np.array(prestim_spk_rate) != 0.0
    if any(valid_trials):
        avg_evoked_spk_rate = np.mean(evoked_spk_rate, where=valid_trials)
        avg_prestim_spk_rate = np.mean(prestim_spk_rate, where=valid_trials)
        std_prestim_spk_rate = np.std(prestim_spk_rate, where=valid_trials)
    else:
        avg_evoked_spk_rate = np.nan
        avg_prestim_spk_rate = np.nan
        std_prestim_spk_rate = np.nan
    return avg_evoked_spk_rate, avg_prestim_spk_rate, std_prestim_spk_rate


def get_unit_stats_per_cat(
    evoked_spk_rate,
    prestim_spk_rate,
    cat_data,
    unique_cats
):
    valid_trials = np.array(prestim_spk_rate) != 0.0
    avg_evoked_spk_rate_per_cat = {x: np.nan for x in unique_cats}
    avg_prestim_spk_rate_per_cat = {x: np.nan for x in unique_cats}
    std_prestim_spk_rate_per_cat = {x: np.nan for x in unique_cats}
    if any(valid_trials):
        for unique_cat in unique_cats:
            valid_cat = valid_trials & cat_data == unique_cat
            if any(valid_cat):
                avg_evoked_spk_rate_per_cat[unique_cat] = np.mean(evoked_spk_rate, where=valid_cat)
                avg_prestim_spk_rate_per_cat[unique_cat] = np.mean(prestim_spk_rate, where=valid_cat)
                std_prestim_spk_rate_per_cat[unique_cat] = np.std(prestim_spk_rate, where=valid_cat)
            else:
                avg_evoked_spk_rate_per_cat[unique_cat] = np.nan
                avg_prestim_spk_rate_per_cat[unique_cat] = np.nan
                std_prestim_spk_rate_per_cat[unique_cat] = np.nan
    else:
        for unique_cat in unique_cats:
            avg_evoked_spk_rate_per_cat[unique_cat] = np.nan
            avg_prestim_spk_rate_per_cat[unique_cat] = np.nan
            std_prestim_spk_rate_per_cat[unique_cat] = np.nan
    return avg_evoked_spk_rate_per_cat, avg_prestim_spk_rate_per_cat, std_prestim_spk_rate_per_cat


def calculate_unit_response(
    avg_evoked_spk_rate,
    avg_prestim_spk_rate,
    std_prestim_spk_rate,
    std_threshold: float = 1e-6
):
    if std_prestim_spk_rate <= std_threshold:
        avg_response = np.nan
    else:
        avg_response = (avg_evoked_spk_rate - avg_prestim_spk_rate) / std_prestim_spk_rate
    return avg_response


def calculate_unit_response_per_cat(
    std_prestim_spk_rate,
    unique_cats,
    avg_evoked_spk_rate_per_cat,
    avg_prestim_spk_rate_per_cat,
    std_prestim_spk_rate_per_cat,
    std_threshold: float = 1e-6
):
    if std_prestim_spk_rate <= std_threshold:
        response_per_cat = np.nan
        max_response = np.nan
    else:
        response_per_cat = {x: np.nan for x in unique_cats}
        for unique_cat in unique_cats:
            if std_prestim_spk_rate_per_cat[unique_cat] == 0.0:
                response_per_cat[unique_cat] = np.nan
            else:
                response_per_cat[unique_cat] = (
                    avg_evoked_spk_rate_per_cat[unique_cat] - avg_prestim_spk_rate_per_cat[unique_cat]
                ) / std_prestim_spk_rate_per_cat[unique_cat]
        max_response = max(response_per_cat.values())
    return response_per_cat, max_response


def calculate_response(
    units,
    trials,
    pre_alignment_window,
    evoked_window,
    event_name: str,
    cat: Optional[str] = None,
    std_threshold: float = 1e-6
):
    """
    Calculate the reponse of units from the table of the NWBFile.

    Evaluates as the difference between pre-trial and post-presentation
    spiking rates scaled by the pre-stimulus noise levels.

    Parameters
    ----------
    nwbfile : NWBFile
        Source of the units table.
    pre_alignment_window : [start, end]
        Length of time to evaluate pre-event spiking over (relative to chosen alignment), in seconds.
    evoked_window : [start, end]
        Length of time to evaluate evoked spiking over, in seconds.
    event_name : str
        Name of event from trials table to align to.
    cat : str, optional
        Categorial column of the trials table to take condition responsitivity over.
        If not None, will return maximum response over categories instead of unconditional.
    std_threshold : float, optional
        Sets all values with a standard deviation in spiking rate below this threshold to NaN.
    """
    units_spk_time = units.spike_times.data[:]
    units_spk_time_index = units.spike_times_index.data[:]
    alignment_times = getattr(trials, event_name).data[:]
    unique_cats = set()
    if cat is not None:
        cat_data = getattr(trials, cat).data[:]
        unique_cats.update(cat_data)

    all_avg_evoked_spk_rate = []
    all_avg_prestim_spk_rate = []
    all_std_prestim_spk_rate = []
    all_avg_evoked_spk_rate_per_cat = []
    all_avg_prestim_spk_rate_per_cat = []
    all_std_prestim_spk_rate_per_cat = []
    all_avg_responses = []
    all_responses_per_cat = []
    all_max_responses = []
    for j, _ in enumerate(units_spk_time_index):
        evoked_spk_rate, prestim_spk_rate = get_unit_rates(
            unit_idx=j,
            units_spk_time=units_spk_time,
            units_spk_time_index=units_spk_time_index,
            alignment_times=alignment_times,
            evoked_window=evoked_window,
            pre_alignment_window=pre_alignment_window
        )
        avg_evoked_spk_rate, avg_prestim_spk_rate, std_prestim_spk_rate = get_unit_stats(
            evoked_spk_rate,
            prestim_spk_rate
        )
        avg_response = calculate_unit_response(
            avg_evoked_spk_rate,
            avg_prestim_spk_rate,
            std_prestim_spk_rate,
            std_threshold=std_threshold
        )
        all_avg_evoked_spk_rate.append(avg_evoked_spk_rate)
        all_avg_prestim_spk_rate.append(avg_prestim_spk_rate)
        all_std_prestim_spk_rate.append(std_prestim_spk_rate)
        all_avg_responses.append(avg_response)

        if cat is not None:
            (
                avg_evoked_spk_rate_per_cat,
                avg_prestim_spk_rate_per_cat,
                std_prestim_spk_rate_per_cat
            ) = get_unit_stats_per_cat(
                evoked_spk_rate,
                prestim_spk_rate,
                cat_data,
                unique_cats
            )
            response_per_cat, max_response = calculate_unit_response_per_cat(
                std_prestim_spk_rate,
                unique_cats,
                avg_evoked_spk_rate_per_cat,
                avg_prestim_spk_rate_per_cat,
                std_prestim_spk_rate_per_cat,
                std_threshold=std_threshold
            )
            all_avg_evoked_spk_rate_per_cat.append(avg_evoked_spk_rate_per_cat)
            all_avg_prestim_spk_rate_per_cat.append(avg_prestim_spk_rate_per_cat)
            all_std_prestim_spk_rate_per_cat.append(std_prestim_spk_rate_per_cat)
            all_responses_per_cat.append(response_per_cat)
            all_max_responses.append(max_response)

    internals = dict(
        avg_evoked_spk_rate=avg_evoked_spk_rate,
        avg_prestim_spk_rate=avg_prestim_spk_rate,
        all_avg_evoked_spk_rate_per_cat=all_avg_evoked_spk_rate_per_cat,
        all_avg_prestim_spk_rate_per_cat=all_avg_prestim_spk_rate_per_cat,
        all_std_prestim_spk_rate_per_cat=all_std_prestim_spk_rate_per_cat,
        std_prestim_spk_rate=std_prestim_spk_rate,
        all_responses_per_cat=all_responses_per_cat
    )
    if cat is None:
        internals.update(all_max_responses=all_max_responses)
        return np.array(all_avg_responses), internals
    else:
        internals.update(all_avg_responses=all_avg_responses)
        return np.array(all_max_responses), internals
