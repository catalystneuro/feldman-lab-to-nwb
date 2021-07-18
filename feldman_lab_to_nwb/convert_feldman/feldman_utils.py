"""Authors: Alessio Buccino and Cody Baker."""
from tqdm import tqdm
import numpy as np

from spikeextractors import SpikeGLXRecordingExtractor


def get_trials_info(recording_nidq: SpikeGLXRecordingExtractor, trial_ongoing_channel: int, event_channel: int):
    """
    Parse trial number, stimulus number, and segment number for each trial.

    First, the 'trial ongoing signal' is used to find start and stop index for each trial.
    Then, for each trial, the hex digits corresponding to trial number, stimulus number,
    and segment number are extracted and converted to integer.

    Parameters
    ----------
    recording_nidq: se.SpikeGLXRecordingExtractor
        The NIDQ recording extractor
    trial_ongoing_channel: int
        The channel_id corresponding to the trial ongoing signal
    event_channel: int
        The channel id correponding to the event message signal

    Returns
    -------
    trial_numbers: list
        List with trial id for each trial
    stimulus_numbers: list
        List with stimulus id for each trial
    segment_numbers: list
        List with segment id for each trial
    trial_times: list
        List with t_start and t_stop for each trial
    """
    # define hex base and conversion
    hex_base = 16
    voltage_range = 4.5 * 1e6
    hex_dict = {0: "0", 1: "1", 2: "2", 3: "3",
                4: "4", 5: "5", 6: "6", 7: "7",
                8: "8", 9: "9", 10: "a", 11: "b",
                12: "c", 13: "d", 14: "e", 15: "f"}

    # get fs and define 10ms interval in samples (used to extract hex digits)
    nidq_fs = recording_nidq.get_sampling_frequency()
    tenms_interval = int(0.01 * nidq_fs)

    # get trial and event traces
    tr_trial = recording_nidq.get_traces(channel_ids=[trial_ongoing_channel])[0]
    tr_events = recording_nidq.get_traces(channel_ids=[event_channel])[0]

    # discretize trial ongoing signal
    tr_trial_bin = np.zeros(tr_trial.shape, dtype=int)
    tr_trial_bin[tr_trial > np.max(tr_trial) // 2] = 1

    # get trial start and trial stop indices
    t_start_idxs = np.where(np.diff(tr_trial_bin) > 0)[0]
    t_stop_idxs = np.where(np.diff(tr_trial_bin) < 0)[0]

    # discard first stop event if it comes before a start event
    if t_stop_idxs[0] < t_start_idxs[0]:
        print("Discarding first trial")
        t_stop_idxs = t_stop_idxs[1:]

    # discard last start event if it comes after last stop event
    if t_start_idxs[-1] > t_stop_idxs[-1]:
        print("Discarding last trial")
        t_start_idxs = t_start_idxs[:-1]

    assert len(t_start_idxs) == len(t_stop_idxs), "Found a different number of trial start and trial stop indices"

    trial_numbers = []
    stimulus_numbers = []
    segment_numbers = []
    trial_times = []

    # Extract hex digits for each trial
    for t in tqdm(range(len(t_start_idxs)), desc="Parsing hex signals"):
        start_idx = t_start_idxs[t]
        stop_idx = t_stop_idxs[t]

        trial_times.append(recording_nidq.frame_to_time(np.array([start_idx, stop_idx])))

        i_start = start_idx
        trial_digits = ""
        stimulus_digits = ""
        segment_digits = ""

        # First 4 digits (10ms each) are the trial number
        for i in range(4):
            median_value = np.median(tr_events[i_start + 10:i_start + tenms_interval - 10])
            digit = (median_value * (hex_base - 1) / voltage_range)
            if digit > 0:
                digit = int(np.floor(digit))
            else:
                digit = int(np.ceil(digit))
            trial_digits += hex_dict[digit]
            i_start += tenms_interval
        trial_numbers.append(int(trial_digits, hex_base))

        # Second 4 digits (10ms each) are the stimulus number
        for i in range(4):
            median_value = np.median(tr_events[i_start + 10:i_start + tenms_interval - 10])
            digit = (median_value * (hex_base - 1) / voltage_range)
            if digit > 0:
                digit = int(np.floor(digit))
            else:
                digit = int(np.ceil(digit))
            stimulus_digits += hex_dict[digit]
            i_start += tenms_interval
        stimulus_numbers.append(int(stimulus_digits, hex_base))

        # Third 4 digits (10ms each) are the segment number
        for i in range(4):
            median_value = np.median(tr_events[i_start + 10:i_start + tenms_interval - 10])
            digit = (median_value * (hex_base - 1) / voltage_range)
            if digit > 0:
                digit = int(np.floor(digit))
            else:
                digit = int(np.ceil(digit))
            segment_digits += hex_dict[digit]
            i_start += tenms_interval
        segment_numbers.append(int(segment_digits, hex_base))

    return trial_numbers, stimulus_numbers, segment_numbers, trial_times
