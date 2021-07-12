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
    trial_csv_column_names: Dict[str, str],
    trial_csv_column_descriptions: Dict[str, str],
    stimulus_column_names: Iterable[str],
    stimulus_column_description: Dict[str, str],
    exclude_columns: Iterable[str]
):
    existing_trial_columns = []
    if nwbfile.trials is not None:
        existing_trial_columns = [x.name for x in nwbfile.trials.columns]
        # exclude_columns = list(exclude_columns)
        # inv_trial_column_names = {v: k for k, v in trial_csv_column_names.items()}
        # for column in nwbfile.trials.columns:
        #     exclude_columns.append(inv_trial_column_names.get(column.name, None))
    valid_trial_csv_columns = set(trial_csv_column_names) - set(existing_trial_columns) - set(exclude_columns)
    for csv_column_name in valid_trial_csv_columns:
        nwbfile.add_trial_column(
            name=trial_csv_column_names[csv_column_name],
            description=trial_csv_column_descriptions.get(csv_column_name, "No description.")
        )
    valid_stimulus_columns = set(stimulus_column_names) - set(existing_trial_columns) - set(exclude_columns)
    for stimulus_column in valid_stimulus_columns:
        nwbfile.add_trial_column(
            name=stimulus_column,
            description=stimulus_column_description[stimulus_column],
            index=True
        )


def add_trials(
    nwbfile: NWBFile,
    trial_starts: Iterable[float],
    trial_stops: Iterable[float],
    header_data: DataFrame,
    trial_data: DataFrame,
    stimulus_data: DataFrame,
    trial_csv_column_names: Dict[str, str],
    stimulus_csv_column_names: Dict[str, str],
    exclude_columns: Iterable[str] = ()
):
    n_elements = int(header_data["Nelements"])
    mapped_header_data = dict(
        stimulus_rises=np.array([int(header_data[f"ElemRise{x}"]) for x in range(n_elements)]),
        stimulus_gngs=np.array([int(header_data[f"ElemGNG{x}"]) for x in range(n_elements)]),
        stimulus_shapes=np.array([int(header_data[f"ElemShape{x}"]) for x in range(n_elements)]),
        stimulus_durations=np.array([float(header_data[f"ElemDur{x}"]) / 1e3 for x in range(n_elements)]),
        stimulus_probabilities=np.array([int(header_data[f"ElemProb{x}"]) for x in range(n_elements)]),
        stimulus_piezo_labels=np.array([str(header_data[f"PiezoLabel{x}"].values[0]) for x in range(n_elements)])
    )

    for n, k in enumerate(range(int(header_data["FirstTrialNum"]), int(header_data["LastTrialNum"]))):
        trial_kwargs = dict(start_time=trial_starts[n], stop_time=trial_stops[n])
        for csv_column_name, trial_column_name in trial_csv_column_names.items():
            trial_kwargs.update({trial_column_name: trial_data[csv_column_name][n]})

        stimulus_elements = np.array(stimulus_data["StimElem"].loc[stimulus_data["Trial"] == k])
        trial_kwargs.update(stimulus_elements=stimulus_elements)
        for csv_column_name, trial_column_name in stimulus_csv_column_names.items():
            trial_kwargs.update(
                {
                    trial_column_name: np.array(stimulus_data[csv_column_name].loc[stimulus_data["Trial"] == k])
                }
            )

        for trial_column_name in mapped_header_data:
            trial_kwargs.update(
                {
                    trial_column_name: [
                        mapped_header_data[trial_column_name][element] for element in stimulus_elements
                    ]
                }
            )

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
        assert len(header_segments) == len(set(segment_numbers_from_nidq)), \
            "Mismatch between number of segments extracted from nidq file and number of header.csv files!"

        exclude_columns = set(["TrNum", "Segment", "ISS0Time", "Arm0Time"])
        trial_csv_column_names = dict(
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
        trial_csv_column_descriptions = dict(
            StimNum="The identifier value for stimulus type.",
            StimLayout=(
                "The index of the simulus layout. 1=Std, 2=Trains, 3=IL, 4=Trains+IL, 5=RFMap, 6=2WC, 7=MWS, 8=MWD"
            ),
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
        stimulus_csv_column_names = dict(
            Time_ms="stimulus_times",
            Ampl="stimulus_amplitudes",
            Posn="stimulus_ordinalities"
        )
        stimulus_column_description = dict(
            stimulus_elements="Type index of each stimulus element.",
            stimulus_times="Time of occurrence of each stimulus element.",
            stimulus_amplitudes="",
            stimulus_ordinalities="Ordinal position of the stimulus element in the train.",
            stimulus_rises="",
            stimulus_gngs="",
            stimulus_shapes="",
            stimulus_durations="Duration of the stimulus element in seconds.",
            stimulus_probabilities="Probability that the stimulus was presented - 0 if deterministic.",
            stimulus_piezo_labels="Manually assigned labels to each stimulus element."
        )
        # last_end_time = 0  # shift value for later segments
        for header_segment in header_segments:
            header_data = read_csv(header_segment, header=None, sep="\t", index_col=0).T
            trial_data = read_csv(str(header_segment).replace("header", "trials"), header=0, sep="\t")
            if trial_data["TrStartTime"].iloc[-1] == 0 and trial_data["TrEndTime"].iloc[-1] == 0:
                trial_data.drop(index=-1)
            stimulus_data = read_csv(str(header_segment).replace("header", "stimuli"), header=0, sep="\t")

            segment_number = header_data["SegmentNum"]
            p = re.compile("_S(\d+)_")
            res = p.search(header_segment.name)
            if res is not None and segment_number.values[0] != res.group(1):
                warn(
                    f"Segment number in file name ({header_segment.name}) does not match internal value! "
                    "Using file name."
                )
            segment_number_from_file_name = int(res.group(1))
            seg_index = segment_numbers_from_nidq == segment_number_from_file_name
            segment_trial_start_times = trial_times_from_nidq[seg_index, 0]
            segment_trial_stop_times = trial_times_from_nidq[seg_index, 1]

            trial_segment_csv_start_times = np.array(trial_data.loc[:, "TrStartTime"])
            for csv_column in trial_data:
                if "Time" in csv_column and not np.all(np.array(trial_data.loc[:, csv_column]) == 0):
                    trial_data.loc[:, csv_column] = (
                        (
                            np.array(trial_data.loc[:, csv_column]) - trial_segment_csv_start_times
                        ) / 1e3 + segment_trial_start_times
                    )
            trial_data.loc[:, "Laser"] = trial_data.loc[:, "Laser"].astype(bool)
            stimulus_data.loc[:, "Time_ms"] = stimulus_data.loc[:, "Time_ms"] / 1e3 + segment_trial_start_times

            add_trial_columns(
                nwbfile=nwbfile,
                trial_csv_column_names=trial_csv_column_names,
                trial_csv_column_descriptions=trial_csv_column_descriptions,
                stimulus_column_names=stimulus_column_description.keys(),
                stimulus_column_description=stimulus_column_description,
                exclude_columns=exclude_columns
            )
            add_trials(
                nwbfile=nwbfile,
                trial_starts=segment_trial_start_times,
                trial_stops=segment_trial_stop_times,
                trial_data=trial_data,
                stimulus_data=stimulus_data,
                header_data=header_data,
                trial_csv_column_names=trial_csv_column_names,
                stimulus_csv_column_names=stimulus_csv_column_names,
                exclude_columns=exclude_columns
            )
