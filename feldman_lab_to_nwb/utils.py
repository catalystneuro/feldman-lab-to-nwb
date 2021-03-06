"""Authors: Alessio Buccino and Cody Baker."""
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Union
from scipy.io import savemat
import json
from typing import Iterable

from pynwb import NWBHDF5IO
from spikeextractors import SpikeGLXRecordingExtractor, RecordingExtractor, SubRecordingExtractor

PathType = Union[str, Path]


def get_trials_info(
    recording_nidq: SpikeGLXRecordingExtractor,
    trial_ongoing_channel: int,
    event_channel: int
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Parse trial number, stimulus number, and segment number for each trial.

    First, the 'trial ongoing signal' is used to find start and stop index for each trial.
    Then, for each trial, the hex digits corresponding to trial number, stimulus number,
    and segment number are extracted and converted to integer.

    Parameters
    ----------
    recording_nidq: spikeextractors.SpikeGLXRecordingExtractor
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
    hex_base = 16
    voltage_range = 4.5 * 1e6
    hex_dict = {
        0: "0", 1: "1", 2: "2", 3: "3",
        4: "4", 5: "5", 6: "6", 7: "7",
        8: "8", 9: "9", 10: "a", 11: "b",
        12: "c", 13: "d", 14: "e", 15: "f"
    }

    nidq_fs = recording_nidq.get_sampling_frequency()
    tenms_interval = int(0.01 * nidq_fs)

    tr_trial = recording_nidq.get_traces(channel_ids=[trial_ongoing_channel])[0]
    tr_events = recording_nidq.get_traces(channel_ids=[event_channel])[0]
    scaled_tr_events = tr_events * (hex_base - 1) / voltage_range
    scaled_tr_events = (scaled_tr_events - np.median(scaled_tr_events)) / np.max(scaled_tr_events) * (hex_base - 1)

    tr_trial_bin = np.zeros(tr_trial.shape, dtype=int)
    tr_trial_bin[tr_trial > np.max(tr_trial) // 2] = 1

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
    trial_times = []

    for t in tqdm(range(len(t_start_idxs)), desc="Parsing hex signals"):
        start_idx = t_start_idxs[t]
        stop_idx = t_stop_idxs[t]

        trial_times.append(recording_nidq.frame_to_time(np.array([start_idx, stop_idx])))

        i_start = start_idx
        trial_digits = ""
        stimulus_digits = ""

        for i in range(4):
            digit = np.median(scaled_tr_events[i_start + 10:i_start + tenms_interval - 10])
            digit = int(round(digit))
            trial_digits += hex_dict[digit]
            i_start += tenms_interval
        trial_numbers.append(int(trial_digits, hex_base))

        for i in range(4):
            digit = np.median(scaled_tr_events[i_start + 10:i_start + tenms_interval - 10])
            digit = int(round(digit))
            stimulus_digits += hex_dict[digit]
            i_start += tenms_interval
        stimulus_numbers.append(int(stimulus_digits, hex_base))
    return np.array(trial_numbers), np.array(stimulus_numbers), np.array(trial_times)


def clip_trials(
    trial_numbers: Iterable,
    stimulus_numbers: Iterable,
    trial_times: Iterable
) -> (Iterable, Iterable, Iterable):
    """
    Clip the trial information from get_trials_info to align correctly.

    It was found in the test data that sometimes the nidq and recording start at unreset trial values; this clips that
    end of the recording and returns all data and times starting from the new point (trial number zero).

    Parameters
    ----------
    trial_numbers : Iterable
        Array of sequential trial numbers.
    stimulus_numbers : Iterable
        Array of sequential stimulus numbers.
    trial_times : Iterable
        Array of start and stop times for the trial_numbers.
    """
    clip_idx = list(trial_numbers).index(0)
    trial_times = trial_times[clip_idx:, :]
    trial_times = trial_times - trial_times[0, 0]
    return trial_numbers[clip_idx:], stimulus_numbers[clip_idx:], trial_times


def clip_recording(
    trial_numbers: Iterable,
    trial_times: Iterable,
    recording: RecordingExtractor = None
) -> SubRecordingExtractor:
    """
    Clip the recording to align with the trials information.

    It was found in the test data that sometimes the nidq and recording start at unreset trial values; this clips that
    end of the recording and returns all data and times starting from the new point (trial number zero).

    Parameters
    ----------
    recording : RecordingExtractor
        If passed, will return a SubRecordingExtractor clipped to align with the trials info.
    """
    return SubRecordingExtractor(
        parent_recording=recording,
        start_frame=recording.time_to_frame(times=trial_times[list(trial_numbers).index(0)][0])
    )


def convert_nwb_to_spikes_mat(
    nwbfile_path: PathType,
    matfile_path: PathType,
    write_waveforms: bool = False,
    n_waveform_samples: int = 37,
    n_waveform_channels: int = 4
):
    """
    Convert the contents of the NWBFile to the format of a SPIKES.mat file.

    Parameters
    ----------
    nwbfile_path : PathType
        Path to the input NWBFile.
    matfile_path : PathType
        Path to the output .mat file.
    write_waveforms : bool, optional
        If True, writes the waveform data from the raw acquisition to the .mat file.
        Loads as contiguous array, and therefore requires a large amount of RAM.
        The default is False.
    n_waveform_samples : int, optional
        Number of recording frames to extract following each spike time for each unit. The default is 37.
    n_waveform_channels : int, optional
        Number of nearby channels to extract waveforms from.
        Prioritizes based on order of nearest distance to the maximum channel.
        The default is 4.
    """
    out_dict = dict(spikes=dict(), attributes=dict())

    with NWBHDF5IO(path=nwbfile_path, mode="r") as io:
        nwbfile = io.read()
        units = nwbfile.units
        n_units = len(units)
        trials = nwbfile.trials
        n_trials = len(trials)

        argsort_spiketimes = np.argsort(units.spike_times[()])
        acq_times = [units.spike_times[idx] for idx in argsort_spiketimes]
        out_dict["spikes"].update(acq_times=acq_times)
        out_dict["spikes"].update(nspikes=len(out_dict["spikes"]["acq_times"]))
        spike_trials = np.array([0] * out_dict["spikes"]["nspikes"])
        spike_times_in_trials = np.array([0.] * out_dict["spikes"]["nspikes"])
        trial_index = 0
        incr_trial = True
        trial_start_time = trials.start_time[()]
        last_trial_start_time = 0
        for j, spike_time in enumerate(out_dict["spikes"]["acq_times"]):
            if incr_trial and spike_time > trial_start_time[trial_index]:
                last_trial_start_time = trial_start_time[trial_index]
                trial_index += 1
                if trial_index == n_trials-1:
                    incr_trial = False
            spike_trials[j] = trial_index + 1
            spike_times_in_trials[j] = spike_time - last_trial_start_time

        acquisition_name = list(nwbfile.acquisition)[0]
        sampling_frequency = nwbfile.acquisition[acquisition_name].rate

        # Some None values may be present in the dumped json string for the paramters, causing an error in scipy write
        def dict_clean(items):
            result = dict()
            for key, value in items:
                if value is None:
                    value = "N/A"
                result[key] = value
            return result
        # Assumes this was made by the Feldman processing pipeline
        parameters = json.loads(nwbfile.units.description, object_pairs_hook=dict_clean)
        filter_parameters = parameters["filter_parameters"]
        sorter_parameters = dict(sortparams=parameters["sorter_parameters"], Fs=sampling_frequency)

        assigns = []
        last_range = 0
        unit_map = dict()
        for unit_idx, (unit_id, idx_range) in enumerate(
            zip(nwbfile.units.id[()], nwbfile.units.spike_times_index.data[()])
        ):
            unit_map.update({unit_id: unit_idx})
            assigns.extend([unit_id] * (idx_range - last_range))
            last_range = idx_range

        waveforms = []
        if write_waveforms:
            channel_locations = np.array(
                [[x, y] for x, y in zip(nwbfile.electrodes.rel_x[()], nwbfile.electrodes.rel_y[()])]
            )
            channel_ids = nwbfile.electrodes.id[()]
            for spike_id, spike_time in zip(assigns, out_dict["spikes"]["acq_times"]):
                spike_frame = round(spike_time / sampling_frequency)
                max_channel = nwbfile.units.max_channel.data[unit_map[spike_id]]
                channel_distances = np.linalg.norm(channel_locations - channel_locations[max_channel], axis=1)
                closest_channels = channel_ids[np.argsort(channel_distances)]
                waveforms.append(
                    nwbfile.acquisition[acquisition_name].data[
                        spike_frame:spike_frame+n_waveform_samples,
                        np.sort(closest_channels[:n_waveform_channels])
                    ] * nwbfile.acquisition["ElectricalSeries_raw"].conversion
                )

        out_dict["spikes"].update(
            assigns=np.array(assigns)[argsort_spiketimes],
            att_fname=str(Path(nwbfile_path).absolute()),
            info=dict(),
            labels=[[x, y] for x, y in zip(list(range(1, n_units + 1)), list(range(1, n_units + 1)))],
            offline_filter=filter_parameters,
            params=sorter_parameters,
            spiketimes=spike_times_in_trials,
            trials=spike_trials,
            unwrapped_times=np.array(out_dict["spikes"]["acq_times"]) - trial_start_time[0],
            waveforms=waveforms
        )

        def unique_indexes(ls):
            seen = set()
            res = []
            for i, n in enumerate(ls):
                if n not in seen:
                    res.append(i)
                    seen.add(n)
            return res

        def as_list(in_val):
            try:
                len(in_val)
                return in_val
            except TypeError:
                return [in_val]

        device_name = list(nwbfile.devices)[0]
        unique_idx = unique_indexes(nwbfile.trials.stimulus_elements[()])
        unique_piezo = nwbfile.trials.stimulus_elements[unique_idx]
        piezo_order = np.argsort(unique_piezo)
        piezo_numbers = unique_piezo[piezo_order]
        out_dict["attributes"].update(
            SweepOnsetSorting=[],
            RewardOnset=nwbfile.trials.reward_start_time[()],
            GNG=nwbfile.trials.stimulus_gngs[()],
            TrLaser=nwbfile.trials.laser_is_on[()],
            sweep_table=dict(
                TrNum=nwbfile.trials.id[()],
                StimNum=nwbfile.trials.stimulus_number[()],
                ISS0Time=nwbfile.trials.ISS0Time[()],
                Iss0time=nwbfile.trials.ISS0Time[()],  # duplicate
                Arm0Time=nwbfile.trials.Arm0Time[()],
                TrStartTime=nwbfile.trials.start_time[()],
                TrEndTime=nwbfile.trials.stop_time[()],
                RWStartTime=nwbfile.trials.reward_start_time[()],
                RWEndTime=nwbfile.trials.reward_end_time[()],
                Tone=nwbfile.trials.tone[()],
                TrType=nwbfile.trials.trial_type[()],
                LickInWindow=nwbfile.trials.licks_in_window[()],
                TrOutcome=nwbfile.trials.trial_outcome[()],
                RewardTime=nwbfile.trials.reward_time[()],
                NLicks=nwbfile.trials.number_of_licks[()],
                CumNRewards=nwbfile.trials.cumulative_number_of_rewards[()],
                CumVol=nwbfile.trials.cumulative_volume[()],
                StimOrder=nwbfile.trials.stimulus_order[()],
                Laser=nwbfile.trials.laser_is_on[()],
            ),
            experiment=dict(
                Index=1,
                Animal_ID=nwbfile.subject.subject_id,
                Recording_Date=nwbfile.session_start_time.strftime("%y%m%d"),
                Genotype=nwbfile.subject.genotype,
                Age=nwbfile.subject.age,
                Rec_Num=1,
                Whiskers=nwbfile.trials.stimulus_piezo_labels[unique_idx][piezo_order],
                Piezo_Numbers=piezo_numbers,
                Piezo_distance=2500,
                Weight_Percent=100,
                Lick_Source="",  # igor or adrian
                Tip_Depth=600,
                Electrode=device_name,
                Electrode_Serial_Num=nwbfile.devices[device_name].description,
                Bad_Channels=[],
                Channel_Map=[[nwbfile.electrodes.id.data[()]]]
            ),
            stimuli=dict(
                Trial=[
                    x for x, y in zip(nwbfile.trials.id[()], nwbfile.trials.stimulus_elements[()]) for _ in as_list(y)
                ],
                Posn=[y for x in nwbfile.trials.stimulus_ordinalities[()] for y in as_list(x)],
                StimElem=nwbfile.trials.stimulus_number[()],
                Time_ms=[z - x for x, y in zip(
                    nwbfile.trials.start_time[()],
                    nwbfile.trials.stimulus_elements[()]
                ) for z in as_list(y)],
                Ampl=[1] * len(nwbfile.trials.stimulus_number[()]),
                ElemPiezo=[y for x in nwbfile.trials.stimulus_elements[()] for y in as_list(x)],
                ElemAmp=[y for x in nwbfile.trials.stimulus_amplitudes[()] for y in as_list(x)],
                ElemProb=[y for x in nwbfile.trials.stimulus_probabilities[()] for y in as_list(x)],
                ElemDur=[y for x in nwbfile.trials.stimulus_durations[()] for y in as_list(x)],
                ElemShape=[y for x in nwbfile.trials.stimulus_shapes[()] for y in as_list(x)],
                ElemRise=[y for x in nwbfile.trials.stimulus_rises[()] for y in as_list(x)],
                ElemGNG=[y for x in nwbfile.trials.stimulus_gngs[()] for y in as_list(x)],
                MW_Tone=nwbfile.trials.tone[()],
                MW_Laser=nwbfile.trials.laser_is_on[()],
                WhiskerID=nwbfile.trials.stimulus_piezo_labels[()],
                StimTime_Unwrapped=[y for x in nwbfile.trials.stimulus_times[()] for y in as_list(x)]
            )
        )
        header_info = json.loads(nwbfile.trials.description)  # Assumes this was made by the FeldmanBehaviorInterface
        n_trials = len(nwbfile.trials.id)
        for x, y in header_info.items():
            out_dict["attributes"]["stimuli"].update({x: [y] * n_trials})
        savemat(file_name=str(matfile_path), mdict=out_dict)
