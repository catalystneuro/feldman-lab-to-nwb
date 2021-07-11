"""Authors: Cody Baker."""
from pandas import DataFrame, read_csv
from pathlib import Path
from warnings import warn
import re
import numpy as np
from typing import Dict, Iterable

from nwb_conversion_tools.basedatainterface import BaseDataInterface
from pynwb import NWBFile
from spikeextractors import SpikeGLXRecordingExtractor

from .feldman_utils import get_trials_info


def add_trial_columns(
    nwbfile: NWBFile,
    trial_column_names: Dict[str, str],
    trial_column_descriptions: Dict[str, str],
    exclude_columns: Iterable[str]
):
    if nwbfile.trials is not None:
        exclude_columns = list(exclude_columns)
        inv_trial_column_names = {v: k for k, v in trial_column_names.items()}
        for column in nwbfile.trials.columns:
            exclude_columns.append(inv_trial_column_names.get(column.name, None))
    valid_columns = set(trial_column_names) - set(exclude_columns)
    for csv_column_name in valid_columns:
        nwbfile.add_trial_column(
            name=trial_column_names[csv_column_name],
            description=trial_column_descriptions.get(csv_column_name, "No description.")
        )
    existing_trial_column_names = [x.name for x in nwbfile.trials.columns]
    for x in [
        "stimulus_times",
        "stimulus_elements",
        "stimulus_amplitudes",
        "stimulus_ordinalities",
        "stimulus_rises",
        "stimulus_gngs",
        "stimulus_shapes",
        "stimulus_durations",
        "stimulus_probabilities",
        "stimulus_piezo_labels"
    ]:
        if x not in existing_trial_column_names:
            nwbfile.add_trial_column(name=x, description="", index=True)

def add_trials(
    nwbfile: NWBFile,
    trial_column_names: Dict[str, str],
    header_data: DataFrame,
    trial_data: DataFrame,
    stimulus_data: DataFrame,
    exclude_columns: Iterable[str] = ()
):
    first_trial = int(header_data["FirstTrialNum"])
    last_trial = int(header_data["LastTrialNum"])
    
    inv_trial_column_names = {v: k for k, v in trial_column_names.items()}
    
    n_elements = int(header_data["Nelements"])
    stimulus_rises = np.array([int(header_data[f"ElemRise{x}"]) for x in range(n_elements)])
    stimulus_gngs = np.array([int(header_data[f"ElemGNG{x}"]) for x in range(n_elements)])
    stimulus_shapes = np.array([int(header_data[f"ElemShape{x}"]) for x in range(n_elements)])
    stimulus_durations = np.array([float(header_data[f"ElemDur{x}"]) / 1e3 for x in range(n_elements)])
    stimulus_probabilities = np.array([int(header_data[f"ElemProb{x}"]) for x in range(n_elements)])
    stimulus_piezo_labels = np.array([str(header_data[f"PiezoLabel{x}"].values[0]) for x in range(n_elements)])
    
    for n, k in enumerate(range(first_trial, last_trial)):
        trial_kwargs = dict()
        for trial_column_name in inv_trial_column_names:
            trial_kwargs.update({trial_column_name: trial_data[inv_trial_column_names[trial_column_name]][n]})
        stimulus_elements = np.array(stimulus_data["StimElem"].loc[stimulus_data["Trial"] == k])
        
        trial_kwargs.update(
            stimulus_times=np.array(trial_data["TrStartTime"][n] + np.array(stimulus_data["Time_ms"].loc[stimulus_data["Trial"] == k])) / 1e3
        )
        trial_kwargs.update(stimulus_elements=stimulus_elements)
        trial_kwargs.update(stimulus_amplitudes=np.array(stimulus_data["Ampl"].loc[stimulus_data["Trial"] == k]))
        trial_kwargs.update(stimulus_ordinalities=np.array(stimulus_data["Posn"].loc[stimulus_data["Trial"] == k]))
        
        trial_kwargs.update(stimulus_rises=[stimulus_rises[stimulus_element] for stimulus_element in stimulus_elements])
        trial_kwargs.update(stimulus_gngs=[stimulus_gngs[stimulus_element] for stimulus_element in stimulus_elements])
        trial_kwargs.update(stimulus_shapes=[stimulus_shapes[stimulus_element] for stimulus_element in stimulus_elements])
        trial_kwargs.update(stimulus_durations=[stimulus_durations[stimulus_element] for stimulus_element in stimulus_elements])
        trial_kwargs.update(stimulus_probabilities=[stimulus_probabilities[stimulus_element] for stimulus_element in stimulus_elements])
        trial_kwargs.update(stimulus_piezo_labels=[stimulus_piezo_labels[stimulus_element] for stimulus_element in stimulus_elements])

        nwbfile.add_trial(**trial_kwargs)


class FeldmanBehaviorDataInterface(BaseDataInterface):
    """Conversion class for the Feldman lab behavioral data."""

    @classmethod
    def get_source_schema(cls):
        return dict(
            required=["folder_path"],
            properties=dict(
                folder_path=dict(type="string")
            )
        )

    def get_metadata(self):
        folder_path = Path(self.source_data["folder_path"])
        header_segments = [x for x in folder_path.iterdir() if "header" in x.name]
        header_data = read_csv(header_segments[0], header=None, sep="\t", index_col=0).T
        metadata = dict(NWBFile=dict(session_id=header_data["ExptName"].values[0]))
        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict, nidq_synch_file: str):
        """
        Primary conversion function for the custom Feldman lab behavioral interface.
        Uses the synch information in the nidq_synch_file to set trial times in NWBFile.
        """
        folder_path = Path(self.source_data["folder_path"])

        (trial_numbers, stimulus_numbers, segment_numbers_from_nidq, trial_times_from_nidq) = get_trials_info(
            recording_nidq=SpikeGLXRecordingExtractor(file_path=nidq_synch_file)
        )
        header_segments = [x for x in folder_path.iterdir() if "header" in x.name]
        # assert len(header_segments) == len(set(segment_numbers_from_nidq)), \
        #     "Mismatch between number of segments extracted from nidq file and number of header.csv files!"

        stim_layout_str = ["Std", "Trains", "IL", "Trains+IL", "RFMap", "2WC", "MWS", "MWD"]
        exclude_columns = set(["TrNum", "Segment", "ISS0Time", "Arm0Time"])
        trial_column_names = dict(
            TrStartTime="start_time",
            TrEndTime="stop_time",
            StimNum="stimulus_number",
            StimLayout="stimulus_layout",
            StimOnsetTime="stimulus_onset_time",
            StimOrder="stimulus_order",
            Tone="tone",
            TrOutcome="trial_outcome",
            TrType="trial_type",
            RewardTime="reward_time",
            RWStartTime="reward_start_time",
            RWEndTime="reward_end_time",
            NLicks="number_of_licks",
            LickInWindow="licks_in_window",
            Laser="laser_is_on",
            CumVol="cumulative_volume",
            CumNRewards="cumulative_number_of_rewards"
        )
        trial_column_descriptions = dict(
            StimNum="The identifier value for stimulus type.",
            StimLayout="",
            StimOnsetTime="The time the stimulus was presented.",
            StimOrder="",
            Tone="",
            TrOutcome="The outcome index for each trial.",
            TrType="The type index for each trial.",
            RewardTime="",
            RWStartTime="",
            RWEndTime="",
            NLicks="",
            LickInWindow="",
            Laser="",
            CumVol="",
            CumNRewards=""
        )
        last_end_time = 0  # shift value for later segments
        for header_segment in header_segments:
            header_data = read_csv(header_segment, header=None, sep="\t", index_col=0).T
            trial_data = read_csv(str(header_segment).replace("header", "trials"), header=0, sep="\t")
            if trial_data["TrStartTime"].iloc[-1] == 0 and trial_data["TrEndTime"].iloc[-1] == 0:
                trial_data = trial_data.iloc[:-1]
            stimulus_data = read_csv(str(header_segment).replace("header", "stimuli"), header=0, sep="\t")
            for x in set(trial_data.keys()) - set(["RewardTime"]):
                if "Time" in x:
                    trial_data[x].loc[trial_data[x] != 0] = trial_data[x].loc[trial_data[x] != 0] / 1e3 + last_end_time
            trial_data["Laser"] = trial_data["Laser"].astype(bool)
            last_end_time = trial_data["TrEndTime"].iloc[-1]

            # segment_number = header_data["SegmentNum"]
            # p = re.compile("_S(\d+)_")
            # res = p.search(header_segment.name)
            # if res is not None and segment_number.values[0] != res.group(1):
            #     warn(
            #         f"Segment number in file name ({header_segment.name}) does not match internal value! "
            #         "Using file name."
            #     )
            # segment_number_from_file_name = int(res.group(1))
            # seg_index = segment_numbers_from_nidq == segment_number_from_file_name
            # segment_trial_start_times = trial_times_from_nidq[seg_index, 0]
            # segment_trial_stop_times = trial_times_from_nidq[seg_index, 1]

            add_trial_columns(
                nwbfile=nwbfile,
                trial_column_names=trial_column_names,
                trial_column_descriptions=trial_column_descriptions,
                exclude_columns=exclude_columns.union(["TrStartTime", "TrEndTime"])
            )
            add_trials(
                nwbfile=nwbfile,
                trial_column_names=trial_column_names,
                trial_data=trial_data,
                stimulus_data=stimulus_data,
                header_data=header_data,
                exclude_columns=exclude_columns
            )
