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
    scaled_tr_events = (scaled_tr_events - min(scaled_tr_events)) / np.ptp(scaled_tr_events) * (hex_base - 1)

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
    trial_times[clip_idx:, :]
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
    filter_parameters : dict, optional
        The type and parameters of the filter(s) applied to the recording before spike sorting.
        Must be of the form
            dict(
                type="name of type",
                parameter_1=value,
                ...
            )
        The default is empty.
    sorter_parameters : dict, optional
        The type and parameters of the spike sorter applied to the recording.
        Recommended to be a string conversion of the dictionary passed into the spikesorters.run_sorters() arguments.
        The default is empty.
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

        out_dict["spikes"].update(
            acq_times=units.spike_times[()],
            nspikes=len(units.spike_times)
        )
        spike_trials = np.array([0] * out_dict["spikes"]["nspikes"])
        spike_times_in_trials = np.array([0.] * out_dict["spikes"]["nspikes"])
        trial = 0
        trial_start_time = trials.start_time[()]
        for j, spike_time in enumerate(out_dict["spikes"]["acq_times"]):
            if spike_time > trial_start_time[trial+1]:
                trial += 1
            spike_trials[j] = trial + 1
            spike_times_in_trials[j] = spike_time - trial_start_time[trial]

        acquisition_name = list(nwbfile.acquisition)[0]
        sampling_frequency = nwbfile.acquisition[acquisition_name].rate
        parameters = json.loads(nwbfile.units.description)  # Assumes this was made by the Feldman processing pipeline
        filter_parameters = parameters["filter_parameters"]
        sorter_parameters = dict(sortparams=parameters["sorter_parameters"], Fs=sampling_frequency)

        assigns = []
        for unit, idx_range in zip(nwbfile.units.id[()], nwbfile.units.spike_times_index.data[()]):
            assigns.extend([unit] * idx_range)

        waveforms = []
        channel_locations = np.array(
            [[x, y] for x, y in zip(nwbfile.electrodes.rel_x[()], nwbfile.electrodes.rel_y[()])]
        )
        channel_ids = nwbfile.electrodes.id[()]
        for spike_idx, spike_time in zip(assigns, out_dict["spikes"]["acq_times"]):
            spike_frame = round(spike_time / sampling_frequency)
            max_channel = nwbfile.units.max_channel.data[spike_idx]
            channel_distances = np.linalg.norm(channel_locations - channel_locations[max_channel], axis=1)
            closest_channels = channel_ids[np.argsort(channel_distances)]
            waveforms.append(
                nwbfile.acquisition[acquisition_name].data[
                    spike_frame:spike_frame+n_waveform_samples,
                    np.sort(closest_channels[:n_waveform_channels])
                ] * nwbfile.acquisition["ElectricalSeries_raw"].conversion
            )

        out_dict["spikes"].update(
            assigns=assigns,
            att_fname=str(nwbfile_path.absolute()),
            info=dict(),
            labels=[[x, y] for x, y in zip(list(range(1, n_units + 1)), list(range(1, n_units + 1)))],
            offline_filter=filter_parameters,
            params=sorter_parameters,
            spiketimes=spike_times_in_trials,
            trials=spike_trials,
            unwrapped_times=units.spike_times[()] - trial_start_time[0],
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
                FirstTrialNum=1,
                LastTrialNum=1,
                StimOnsetTime=1,
                StimLayout=1,
                Segment=1,
                SegmentNum=1,
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
