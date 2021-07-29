"""Authors: Cody Baker."""
from pandas import DataFrame, read_csv
from pathlib import Path
import numpy as np
from typing import Dict, Iterable

from nwb_conversion_tools.basedatainterface import BaseDataInterface
from pynwb import NWBFile
from spikeextractors import SpikeGLXRecordingExtractor

from .feldman_utils import get_trials_info, clip_trials


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
    trial_times: Iterable[Iterable[float]],
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
        trial_kwargs = dict(start_time=trial_times[k][0], stop_time=trial_times[k][1])
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
            required=["folder_path", "nidq_synch_file", "trial_ongoing_channel", "event_channel"],
            properties=dict(
                folder_path=dict(type="string"),
                nidq_synch_file=dict(type="string"),
                trial_ongoing_channel=dict(type="number"),
                event_channel=dict(type="number"),
            )
        )

    def get_metadata(self):
        folder_path = Path(self.source_data["folder_path"])
        header_segments = [x for x in folder_path.iterdir() if "header" in x.name]
        header_data = read_csv(header_segments[0], header=None, sep="\t", index_col=0).T
        metadata = dict(NWBFile=dict(session_id=header_data["ExptName"].values[0]))
        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """
        Primary conversion function for the custom Feldman lab behavioral interface.

        Uses the synch information in the nidq_synch_file to set trial times in NWBFile.
        """
        folder_path = Path(self.source_data["folder_path"])

        trial_numbers, stimulus_numbers, trial_times_from_nidq = get_trials_info(
            recording_nidq=SpikeGLXRecordingExtractor(file_path=self.source_data["nidq_synch_file"]),
            trial_ongoing_channel=self.source_data["trial_ongoing_channel"],
            event_channel=self.source_data["event_channel"]
        )
        trial_numbers, stimulus_numbers, trial_times_from_nidq = clip_trials(
            trial_numbers=trial_numbers,
            stimulus_numbers=stimulus_numbers,
            trial_times=trial_times_from_nidq
        )
        header_segments = [x for x in folder_path.iterdir() if "header" in x.name]

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
                "The index of the stimulus layout. 1=Std, 2=Trains, 3=IL, 4=Trains+IL, 5=RFMap, 6=2WC, 7=MWS, 8=MWD"
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
            stimulus_probabilities="Probability that the stimulus was presented; 0 if deterministic.",
            stimulus_piezo_labels="Manually assigned labels to each stimulus element."
        )
        add_trial_columns(
            nwbfile=nwbfile,
            trial_csv_column_names=trial_csv_column_names,
            trial_csv_column_descriptions=trial_csv_column_descriptions,
            stimulus_column_names=stimulus_column_description.keys(),
            stimulus_column_description=stimulus_column_description,
            exclude_columns=exclude_columns
        )
        for header_segment in header_segments:
            header_data = read_csv(header_segment, header=None, sep="\t", index_col=0).T
            trial_data = read_csv(str(header_segment).replace("header", "trials"), header=0, sep="\t")
            if trial_data["TrStartTime"].iloc[-1] == 0 and trial_data["TrEndTime"].iloc[-1] == 0:
                trial_data.drop(trial_data.index[-1], inplace=True)
            stimulus_data = read_csv(str(header_segment).replace("header", "stimuli"), header=0, sep="\t")

            trial_segment_csv_start_times = np.array(trial_data.loc[:, "TrStartTime"])
            for csv_column in trial_data:
                if "Time" in csv_column and not np.all(np.array(trial_data.loc[:, csv_column]) == 0):
                    trial_data.loc[:, csv_column] = (
                        (
                            np.array(trial_data.loc[:, csv_column]) - trial_segment_csv_start_times
                        ) / 1e3 + trial_times_from_nidq[trial_data.loc[:, "TrNum"], 0]
                    )
            trial_data.loc[:, "Laser"] = trial_data.loc[:, "Laser"].astype(bool)
            last_trial = 0
            m = 0
            for j, (trial, offset) in enumerate(zip(stimulus_data.loc[:, "Trial"], stimulus_data.loc[:, "Time_ms"])):
                if trial == last_trial:
                    m += 1
                else:
                    last_trial = trial
                    m = 1
                stimulus_data.loc[j, "Time_ms"] = trial_times_from_nidq[trial, 0] + offset / 1e3 * m

            add_trials(
                nwbfile=nwbfile,
                trial_times=trial_times_from_nidq,
                trial_data=trial_data,
                stimulus_data=stimulus_data,
                header_data=header_data,
                trial_csv_column_names=trial_csv_column_names,
                stimulus_csv_column_names=stimulus_csv_column_names,
                exclude_columns=exclude_columns
            )
