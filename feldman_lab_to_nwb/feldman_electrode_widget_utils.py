"""Authors: Cody Baker."""
import numpy as np
from typing import Optional, Iterable

from pynwb.misc import Units
from pynwb.file import TrialTable


def get_unit_rates(
    unit_idx: int,
    units_spk_time: Iterable[float],
    units_spk_time_index: Iterable[int],
    alignment_times: Iterable[float],
    pre_alignment_window: Iterable[float],
    post_alignment_window: Iterable[float],
) -> (Iterable[float], Iterable[float]):
    """
    Extract spiking rates within pre- and post- alignment intervals from a units spike time list.

    Parameters
    ----------
    unit_idx : int
        Index of the units_spk_time_index array whose rates are being extracted.
    units_spk_time : Iterable[float]
        Array-like object with global raw spike times for all units.
    units_spk_time_index : Iterable[int]
        Array-like object mapping the unit index of each spike time from the units_spk_time object.
    alignment_times : Iterable[float]
        Array-like object specifying the time around which to extract rates.
        Usually indicates trial times, or event times within trials.
    pre_alignment_window : Iterable[float]
        Array-like of the form [start, end] flipped with respect to the value of the alignment time, in units seconds.
        E.g., pre_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
        will calculate spiking activity over the windows [[0.7, 1.1], [1.8, 2.2], [2.9, 3.3]] respectively.
    post_alignment_window : Iterable[float]
        Array-like of the form [start, end] with respect to the value of the alignment time, in units seconds.
        E.g., post_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
        will calculate spiking activity over the windows [[1.3, 1.7], [2.4, 2.8], [3.5, 3.9]] respectively.

    Returns
    -------
    evoked_spk_rate : Iterable[float]
        List of spiking rates over the post_alignment_window for each element of the alignment_times.
    prestim_spk_rate : Iterable[float]
        List of spiking rates over the pre_alignment_window for each element of the alignment_times.
    """
    spk_idx_lb = 0 if unit_idx == 0 else units_spk_time_index[unit_idx-1]
    spks = units_spk_time[spk_idx_lb:units_spk_time_index[unit_idx]]
    evoked_spk_rate = []
    prestim_spk_rate = []
    for alignment_time in alignment_times:
        evoked_spk_rate.append(
            sum(
                (spks >= alignment_time + post_alignment_window[0]) & (spks < alignment_time + post_alignment_window[1])
            ) / (post_alignment_window[1] - post_alignment_window[0])
        )
        prestim_spk_rate.append(
            sum(
                (spks >= alignment_time - pre_alignment_window[1]) & (spks < alignment_time - pre_alignment_window[0])
            ) / (pre_alignment_window[1] - pre_alignment_window[0])
        )
    return evoked_spk_rate, prestim_spk_rate


def calculate_unit_response(
    unit_idx: int,
    units_spk_time: Iterable[float],
    units_spk_time_index: Iterable[int],
    alignment_times: Iterable[float],
    pre_alignment_window: Iterable[float],
    post_alignment_window: Iterable[float],
    std_threshold: float = 1e-6
) -> (Iterable[float], Iterable[float]):
    """
    Calculate the responsitivity of a single unit from a list of global spike times and their unit indices.

    Responsitivity is defined here as (X - mu) / sigma, where X is the spiking rate of the unit during the
    post_alignment_window and mu is the baseline spiking rate estimated over the pre_alignment_window. Sigma is the
    baseline standard deviation of spiking rates during the pre_alignment_window.

    Parameters
    ----------
    unit_idx : int
        Index of the units_spk_time_index array whose rates are being extracted.
    units_spk_time : Iterable[float]
        Array-like object with global raw spike times for all units.
    units_spk_time_index : Iterable[int]
        Array-like object mapping the unit index of each spike time from the units_spk_time object.
    alignment_times : Iterable[float]
        Array-like object specifying the time around which to extract rates.
        Usually indicates trial times, or event times within trials.
    pre_alignment_window : Iterable[float]
        Array-like of the form [start, end] flipped with respect to the value of the alignment time, in units seconds.
        E.g., pre_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
        will calculate spiking activity over the windows [[0.7, 1.1], [1.8, 2.2], [2.9, 3.3]] respectively.
    post_alignment_window : Iterable[float]
        Array-like of the form [start, end] with respect to the value of the alignment time, in units seconds.
        E.g., post_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
        will calculate spiking activity over the windows [[1.3, 1.7], [2.4, 2.8], [3.5, 3.9]] respectively.
    std_threshold : float, optional (defaults to 1e-6)
        Any units with pre-alignment noise levels less than this threshold have their response set to NaN.

    Returns
    -------
    evoked_spk_rate : Iterable[float]
        List of spiking rates over the post_alignment_window for each element of the alignment_times.
    prestim_spk_rate : Iterable[float]
        List of spiking rates over the pre_alignment_window for each element of the alignment_times.
    """
    evoked_spk_rate, prestim_spk_rate = get_unit_rates(
        unit_idx=unit_idx,
        units_spk_time=units_spk_time,
        units_spk_time_index=units_spk_time_index,
        alignment_times=alignment_times,
        post_alignment_window=post_alignment_window,
        pre_alignment_window=pre_alignment_window
    )

    avg_evoked_spk_rate = np.mean(evoked_spk_rate)
    avg_prestim_spk_rate = np.mean(prestim_spk_rate)
    std_prestim_spk_rate = np.std(prestim_spk_rate)
    if std_prestim_spk_rate <= std_threshold or avg_prestim_spk_rate == 0.:
        avg_response = np.nan
    else:
        avg_response = (avg_evoked_spk_rate - avg_prestim_spk_rate) / std_prestim_spk_rate
    return avg_response


def calculate_all_responses(
    units: Units,
    trials: TrialTable,
    pre_alignment_window: Iterable[float],
    post_alignment_window: Iterable[float],
    event_name: str,
    std_threshold: float = 1e-6
) -> Iterable[float]:
    """
    Calculate the reponses of units from a UnitsTable.

    Evaluates as the difference between pre-trial and post-presentation
    spiking rates scaled by the pre-stimulus noise levels.

    Parameters
    ----------
    units : Units
        A NWBFile Units Table.
    trials : TrialTable
        A NWBFile Trials Table.
    pre_alignment_window : Iterable[float]
        Array-like of the form [start, end] flipped with respect to the value of the alignment time, in units seconds.
        E.g., pre_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
        will calculate spiking activity over the windows [[0.7, 1.1], [1.8, 2.2], [2.9, 3.3]] respectively.
    post_alignment_window : Iterable[float]
        Array-like of the form [start, end] with respect to the value of the alignment time, in units seconds.
        E.g., post_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
        will calculate spiking activity over the windows [[1.3, 1.7], [2.4, 2.8], [3.5, 3.9]] respectively.
    event_name : str
        Name of event from trials table to align to.
    cat : str, optional
        Categorial column of the trials table to take condition responsitivity over.
        If not None, will return maximum response over categories instead of unconditional.
    std_threshold : float, optional (defaults to 1e-6)
        Any units with pre-alignment noise levels less than this threshold have their response set to NaN.

    Returns
    -------
    responses : Iterable[float]
        Array of responsitivity for each unit in the Units Table.
    """
    units_spk_time = units.spike_times.data[:]
    units_spk_time_index = units.spike_times_index.data[:]
    alignment_times = getattr(trials, event_name).data[:]

    responses = []
    for j, _ in enumerate(units_spk_time_index):
        responses.append(
            calculate_unit_response(
                unit_idx=j,
                units_spk_time=units_spk_time,
                units_spk_time_index=units_spk_time_index,
                alignment_times=alignment_times,
                pre_alignment_window=pre_alignment_window,
                post_alignment_window=post_alignment_window,
                std_threshold=std_threshold
            )
        )
    return np.array(responses)
