"""Authors: Cody Baker."""
from pandas import DataFrame, read_csv
from pathlib import Path
from warnings import warn
import re
import numpy as np
from typing import Iterable, List

from nwb_conversion_tools.basedatainterface import BaseDataInterface
from pynwb import NWBFile
from spikeextractors import SpikeGLXRecordingExtractor

from .feldman_utils import get_trials_info


def add_trial_columns(
    nwbfile: NWBFile,
    csv_data: DataFrame,
    csv_column_names: Iterable[str],
    trial_column_names: Iterable[str],
    trial_column_descriptions: Iterable[str]
):
    valid_columns = dict()
    for csv_column_name, trial_column_name, trial_column_description in zip(
            csv_column_names,
            trial_column_names,
            trial_column_descriptions
    ):
        if len(set(csv_data[csv_column_name])) > 1:
            valid_columns.update({csv_column_name: trial_column_name})
            nwbfile.add_trial(name=trial_column_name, description=trial_column_description)
    return valid_columns


def add_trials(nwbfile: NWBFile, csv_data: DataFrame, segment_trial_starts: List[str], valid_columns: Iterable[str]):
    trial_kwargs = dict()
    for csv_column_name, trial_column_name in valid_columns.items():
        trial_kwargs.update({trial_column_name: csv_data[csv_column_name]})
    for k in range(len(segment_trial_starts)):
        nwbfile.add_trial(
            start_time=csv_data["TrStartTime"][k],
            stop_time=csv_data["TrEndTime"][k],
        )


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
        (trial_numbers, stimulus_numbers, segment_numbers_from_nidq, trial_times_from_nidq) = get_trials_info(
            recording_nidq=SpikeGLXRecordingExtractor(file_path=nidq_synch_file)
        )

        folder_path = Path(self.source_data["folder_path"])
        header_segments = [x for x in folder_path.iterdir() if "header" in x.name]
        assert len(header_segments) == len(set(segment_numbers_from_nidq)), \
            "Mismatch between number of segments extracted from nidq file and number of header.csv files!"

        stim_layout_str = ["Std", "Trains", "IL", "Trains+IL", "RFMap", "2WC", "MWS", "MWD"]

        column_name_mapping = dict(
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
            Laser="laser",
            CumVol="cumulative_volume",
            CumNRewards="cumulative_number_of_rewards"
        )
        column_description_mapping = dict(
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
        for header_segment in header_segments:
            header_data = read_csv(header_segment, header=None, sep="\t", index_col=0).T
            trial_data = read_csv(str(header_segment).replace("header", "trials"), header=0, sep="\t")
            stimuli_data = read_csv(str(header_segment).replace("header", "stimuli"), header=0, sep="\t")

            segment_number = header_data["SegmentNum"]
            p = re.compile("_S(\d+)_")
            res = p.search(header_segment.name)
            if res is not None and segment_number.values[0] != res.group(1):
                warn(
                    f"Segment number in file name ({header_segment.name}) does not match internal value! "
                    "Using file name."
                )
            segment_number_from_file_name = int(res.group(1))

            stim_layout = int(header_data["StimLayout"].values[0]) - 1  # -1 for zero-indexing
            if stim_layout == 1:
                n_stim = int(header_data["StdStimN"])
                if n_stim != 1:
                    raise NotImplementedError(
                        f"StdStimN ({n_stim}) from header _data in segment file {header_segment} is not yet supported!"
                    )

                seg_index = segment_numbers_from_nidq == segment_number_from_file_name
                segment_trial_starts = trial_times_from_nidq[seg_index, 0]
                segment_trial_ends = trial_times_from_nidq[seg_index, 1]
                segment_stimulus_onset_time = int(header_data["StdStimOnset"]) / 1e3
                stimulus_times = segment_trial_starts + segment_stimulus_onset_time

                amplitude_map = [int(x) for x in header_data[[x for x in header_data if "ElemAmp" in x]].iloc[0]]
                probability_map = [int(x) for x in header_data[[x for x in header_data if "ElemProb" in x]].iloc[0]]
                duration_map = [int(x) / 1e3 for x in header_data[[x for x in header_data if "ElemDur" in x]].iloc[0]]
                shape_map = [int(x) for x in header_data[[x for x in header_data if "ElemShape" in x]].iloc[0]]
                rise_map = [int(x) for x in header_data[[x for x in header_data if "ElemRise" in x]].iloc[0]]
                GNG_map = [int(x) for x in header_data[[x for x in header_data if "ElemGNG" in x]].iloc[0]]

                elem_piezo = {x: header_data[x].values[0] for x in header_data if "ElemPiezo" in x}
                elem_piezo_labels = {x: header_data[x].values[0] for x in header_data if "PiezoLabel" in x}
                assert len(elem_piezo) == len(elem_piezo_labels), "Size mismatch between element piezo and piezo label!"
                element_index_label_pairs = [
                    [elem_piezo[x], elem_piezo_labels[y]] for x, y in zip(elem_piezo, elem_piezo_labels)
                ]
                element_index_label_map = dict(np.unique(element_index_label_pairs, axis=0))

                trial_types = list(trial_data["TrType"])
                trial_outcomes = list(trial_data["TrOutcome"])

                for k in range(len(segment_trial_starts)):
                    nwbfile.add_trial(
                        start_time=segment_trial_starts[k],
                        stop_time=segment_trial_ends[k],
                        stimulus_time=stimulus_times[k],
                        stimulus_number=stimulus_numbers[k],
                        stimulus_name=element_index_label_map[str(stimulus_numbers[k])],
                        trial_type=trial_types[k],
                        trial_outcome=trial_outcomes[k],
                        amplitude=amplitude_map[stimulus_numbers[k]],
                        probability=probability_map[stimulus_numbers[k]],
                        duration=duration_map[stimulus_numbers[k]],
                        shape=shape_map[stimulus_numbers[k]],
                        rise=rise_map[stimulus_numbers[k]],
                        gng=GNG_map[stimulus_numbers[k]]
                    )
            elif stim_layout == 4:
                for header_segment in [1]:
                    header_data = read_csv(header_segment, header=None, sep="\t", index_col=0).T
                    trial_data = read_csv(str(header_segment).replace("header", "trials"), header=0, sep="\t")
                    stimuli_data = read_csv(str(header_segment).replace("header", "stimuli"), header=0, sep="\t")

                    csv_column_names = set(trial_data.keys()) - set(
                            ["TrNum", "Segment", "ISS0Time", "Arm0Time", "TrStartTime", "TrEndTime"]
                    )
                    valid_columns = add_trial_columns(
                        nwbfile=nwbfile,
                        csv_data=trial_data,
                        csv_column_names=csv_column_names,
                        trial_column_names=column_name_mapping,
                        trial_column_descriptions=column_description_mapping
                    )
                    add_trials(
                        nwbfile=nwbfile,
                        csv_data=trial_data,
                        segment_trial_starts=segment_trial_starts,
                        valid_columns=valid_columns
                    )


                    for k in range(len(segment_trial_starts)):
                        nwbfile.add_trial(
                            start_time=segment_trial_starts[k],
                            stop_time=segment_trial_ends[k],
                            stimulus_time=stimulus_times[k],
                            stimulus_number=stimulus_numbers[k],
                            stimulus_name=element_index_label_map[str(stimulus_numbers[k])],
                            trial_type=trial_types[k],
                            trial_outcome=trial_outcomes[k],
                            duration=duration_map[stimulus_numbers[k]],
                            shape=shape_map[stimulus_numbers[k]],
                            rise=rise_map[stimulus_numbers[k]],
                            gng=GNG_map[stimulus_numbers[k]]
                        )
            elif stim_layout == 5:
                n_stim = int(header_data["StdStimN"])
                seg_index = segment_numbers_from_nidq == segment_number_from_file_name
                segment_trial_starts = trial_times_from_nidq[seg_index, 0]
                segment_trial_ends = trial_times_from_nidq[seg_index, 1]
                segment_stimulus_onset_time = int(header_data["StdStimOnset"]) / 1e3
                stimulus_times = segment_trial_starts + segment_stimulus_onset_time

                amplitude_map = [int(x) for x in header_data[[x for x in header_data if "ElemAmp" in x]].iloc[0]]
                probability_map = [int(x) for x in header_data[[x for x in header_data if "ElemProb" in x]].iloc[0]]
                duration_map = [int(x) / 1e3 for x in header_data[[x for x in header_data if "ElemDur" in x]].iloc[0]]
                shape_map = [int(x) for x in header_data[[x for x in header_data if "ElemShape" in x]].iloc[0]]
                rise_map = [int(x) for x in header_data[[x for x in header_data if "ElemRise" in x]].iloc[0]]
                GNG_map = [int(x) for x in header_data[[x for x in header_data if "ElemGNG" in x]].iloc[0]]

                elem_piezo = {x: header_data[x].values[0] for x in header_data if "ElemPiezo" in x}
                elem_piezo_labels = {x: header_data[x].values[0] for x in header_data if "PiezoLabel" in x}
                assert len(elem_piezo) == len(elem_piezo_labels), "Size mismatch between element piezo and piezo label!"
                element_index_label_pairs = [
                    [elem_piezo[x], elem_piezo_labels[y]] for x, y in zip(elem_piezo, elem_piezo_labels)
                ]
                element_index_label_map = dict(np.unique(element_index_label_pairs, axis=0))

                trial_types = list(trial_data["TrType"])
                trial_outcomes = list(trial_data["TrOutcome"])
                reward_start_times = trial_data["RWStartTime"] - trial_data["TrStartTime"] + segment_trial_starts
                reward_end_times = trial_data["RWEndTime"] - trial_data["TrStartTime"] + segment_trial_starts

                nwbfile.add_trial_column(name="reward_start_time", description="Start time of reward.")  # check name
                nwbfile.add_trial_column(name="reward_stop_time", description="Stop time of reward.")  # check name
                for k in range(len(segment_trial_starts)):
                    nwbfile.add_trial(
                        start_time=segment_trial_starts[k],
                        stop_time=segment_trial_ends[k],
                        stimulus_time=stimulus_times[k],
                        stimulus_number=stimulus_numbers[k],
                        stimulus_name=element_index_label_map[str(stimulus_numbers[k])],
                        trial_type=trial_types[k],
                        trial_outcome=trial_outcomes[k],
                        amplitude=amplitude_map[stimulus_numbers[k]],
                        probability=probability_map[stimulus_numbers[k]],
                        duration=duration_map[stimulus_numbers[k]],
                        shape=shape_map[stimulus_numbers[k]],
                        rise=rise_map[stimulus_numbers[k]],
                        gng=GNG_map[stimulus_numbers[k]],
                        reward_start_time=reward_start_times[k],
                        reward_stop_time=reward_end_times[k]
                    )
            else:
                raise NotImplementedError("StimLayouts other than 1 and 5 have not yet been implemented!")