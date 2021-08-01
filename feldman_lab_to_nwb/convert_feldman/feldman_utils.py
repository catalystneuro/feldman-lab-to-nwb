"""Authors: Alessio Buccino and Cody Baker."""
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Union
from scipy.io import savemat
import json

from pynwb import NWBHDF5IO

from spikeextractors import SpikeGLXRecordingExtractor

PathType = Union[str, Path]


def get_trials_info(recording_nidq: SpikeGLXRecordingExtractor, trial_ongoing_channel: int, event_channel: int):
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
    segment_numbers = []
    trial_times = []

    for t in tqdm(range(len(t_start_idxs)), desc="Parsing hex signals"):
        start_idx = t_start_idxs[t]
        stop_idx = t_stop_idxs[t]

        trial_times.append(recording_nidq.frame_to_time(np.array([start_idx, stop_idx])))

        i_start = start_idx
        trial_digits = ""
        stimulus_digits = ""
        segment_digits = ""

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

        for i in range(4):
            digit = np.median(scaled_tr_events[i_start + 10:i_start + tenms_interval - 10])
            digit = int(round(digit))
            segment_digits += hex_dict[digit]
            i_start += tenms_interval
        segment_numbers.append(int(segment_digits, hex_base))
    return np.array(trial_numbers), np.array(stimulus_numbers), np.array(segment_numbers), np.array(trial_times)


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
        spike_times_in_trials = np.array([0] * out_dict["spikes"]["nspikes"])
        trial = 0
        for j, spike_time in enumerate(out_dict["spikes"]["acq_times"]):
            while spike_time > trials["stop_time"][trial] and trial < n_trials:
                trial += 1
            spike_trials[j] = trial + 1
            spike_times_in_trials[j] = spike_time - trials["start_time"][trial]

        sampling_frequency = nwbfile.units.sampling_frequency
        parameters = json.loads(nwbfile.units.description)  # Assumes this was made by the Feldman processing pipeline
        filter_parameters = parameters["filter_parameters"]
        sorter_parameters = parameters["sorter_parameters"]
        sorter_parameters.update(Fs=sampling_frequency)

        waveforms = []
        channel_locations = np.array(
            [[x, y] for x, y in zip(nwbfile.electrodes.rel_x.data, nwbfile.electrodes.rel_y.data)]
        )
        channel_ids = np.array(nwbfile.electrodes.id)
        for spike_index in units.spike_times_indexes:
            spike_frame = round(spike_time/sampling_frequency)
            max_channel = nwbfile.units.max_channel[spike_index]
            channel_distances = np.linalg.norm(channel_locations - channel_locations[max_channel], axis=1)
            closest_channels = channel_ids[np.argsort(channel_distances)]
            waveforms.append(
                nwbfile.acquisition[list(nwbfile.acquisition)[0]].data[
                    spike_frame:spike_frame+n_waveform_samples,
                    closest_channels[:n_waveform_channels]
                ]
            )

        out_dict["spikes"].update(
            assigns=units.spike_times_indexes,
            att_fname=nwbfile_path.absolute(),
            info=dict(),
            labels=[list(range(1, n_units + 1)), list(range(1, n_units + 1))],
            offline_filter=filter_parameters,
            params=sorter_parameters,
            spiketimes=spike_times_in_trials,
            trials=spike_trials,
            unwrapped_times=units.spike_times[()] - trials["start_time"][0],
            waveforms=waveforms
        )

        device_name = list(nwbfile.devices)[0]
        unique_piezo = list(set(nwbfile.trials.stimulus_elements[()]))
        piezo_order = np.argsort(unique_piezo)
        piezo_numbers = unique_piezo[piezo_order]
        out_dict["attributes"].update(
            SweepOnsetSorting=dict(),
            RewardOnset=nwbfile.trials.reward_start_time,
            GNG=nwbfile.trials.stimulus_gngs,
            TrLaser=nwbfile.trials.laser_is_on,
            sweep_table=dict(
                TrNum=1,
                Segment=1,
                ISS0Time=1,
                Arm0Time=1,
                TrStartTime=1,
                TrEndTime=1,
                RWStartTime=1,
                RWEndTime=1,
                StimOnsetTime=1,
                StimNum=1,
                Tone=1,
                TrType=1,
                LickInWindow=1,
                TrOutcome=1,
                RewardTime=1,
                NLicks=1,
                CumNRewards=1,
                CumVol=1,
                StimLayout=1,
                StimOrder=1,
                Laser=1,
                SegmentNum=1,
                Iss0time=1,  # duplicate
                FirstTrialNum=1,
                LastTrialNum=1,
                Nstimuli=1,
                ArduinoMode=1,
                ITImean=1,
                ITIrange=1,
                MaxLicksLimit=1,
                RewardCalib_b=1,
                RewardCalib_m=1,
                UseGlobalGNG=1,
                A=1,
                B=1,
                # ...=1,
                Z=1,
                Nelements=1,
                PiezoLabel0=1,
                # ...=1,
                PieazoLabel18=1,
            ),
            experiment=dict(
                Index=1,
                Animal_ID=nwbfile.subject.subject_id,
                Recording_Date=nwbfile.session_start_time.strftime("%y%m%d"),
                Genotype=nwbfile.subject.genotype,
                Age=nwbfile.subject.age,
                Rec_Num=1,
                Whiskers=list(set(nwbfile.trials.stimulus_piezo_labels))[piezo_order],
                Piezo_Numbers=piezo_numbers,
                Piezo_distance=2500,
                Weight_Percent=100,
                Lick_Source="",  # igor or adrian
                Tip_Depth=600,
                Electrode=device_name,
                Electrode_Serial_Num=nwbfile.devices[device_name].description,
                Bad_Channels=[],
                Channel_Map=[
                    nwbfile.extracellular_ephys.id[nwbfile.extracellular_ephys.electrode.group == group]
                    for group in nwbfile.extracellular_ephys.electrode.group
                ]
            ),
            stimuli=dict(
                Trial=[x for x, y in zip(nwbfile.trials.id, nwbfile.trials.stimulus_elements) for _ in y],
                Posn=[y for x in nwbfile.trials.stimulus_ordinalities for y in x],
                StimElem=nwbfile.trials.stimulus_number,
                Time_ms=[z - x for x, y in zip(nwbfile.trials.start_time, nwbfile.trials.stimulus_elements) for z in y],
                Ampl=[1] * len(nwbfile.trials.stimulus_number),
                ElemPiezo=[y for x in nwbfile.trials.stimulus_elements for y in x],
                ElemAmp=[y for x in nwbfile.trials.stimulus_amplitudes for y in x],
                ElemProb=[y for x in nwbfile.trials.stimulus_probabilities for y in x],
                ElemDur=[y for x in nwbfile.trials.stimulus_durations for y in x],
                ElemShape=[y for x in nwbfile.trials.stimulus_shapes for y in x],
                ElemRise=[y for x in nwbfile.trials.stimulus_rises for y in x],
                ElemGNG=[y for x in nwbfile.trials.stimulus_gngs for y in x],
                MW_Tone=nwbfile.trials.tone,
                MW_Laser=nwbfile.trials.laser_is_on,
                WhiskerID=nwbfile.trials.stimulus_piezo_labels,
                StimTime_Unwrapped=[y for x in nwbfile.trials.stimulus_times for y in x]
            )
        )
        out_dict["stimuli"].update(
            MW_RelAmp=out_dict["stimuli"]["ElemAmp"],
            MW_Prob=out_dict["stimuli"]["ElemProb"],
            MW_GNG=out_dict["stimuli"]["ElemGNG"]
        )
        savemat(file_name=str(matfile_path), mdict=out_dict)
