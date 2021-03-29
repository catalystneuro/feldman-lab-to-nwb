"""Authors: Cody Baker."""
import pandas as pd
from pathlib import Path
from warnings import warn
import re
import numpy as np

from ndx_events import AnnotatedEventsTable
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.conversion_tools import check_module
from pynwb import NWBFile
from spikeextractors import SpikeGLXRecordingExtractor

from .feldman_utils import get_trials_info


class FeldmanBehaviorDataInterface(BaseDataInterface):
    """Conversion class for the Feldman lab behavioral data."""

    @classmethod
    def get_source_schema(cls):
        return dict(
            required=['folder_path'],
            properties=dict(
                folder_path=dict(type='string')
            )
        )

    def get_metadata(self):
        folder_path = Path(self.source_data["folder_path"])
        header_segments = [x for x in folder_path.iterdir() if "header" in x.name]
        header_data = pd.read_csv(header_segments[0], header=None, sep="\t", index_col=0).T
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

        elements = []
        element_times = []
        amplitude = []
        ordinality = []
        element_index_label_pairs = []
        trial_starts = []
        trial_ends = []
        for header_segment in header_segments:
            header_data = pd.read_csv(header_segment, header=None, sep="\t", index_col=0).T

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
            if stim_layout != 0:
                raise NotImplementedError(
                    f"StimLayouts other than '{stim_layout_str[stim_layout]}' type have not yet been implemented!"
                )

            trial_data = pd.read_csv(str(header_segment).replace("header", "trials"), header=0, sep="\t")
            trial_starts.extend(trial_data['TrStartTime'] / 1e3)
            trial_ends.extend(trial_data['TrEndTime'] / 1e3)
            # may have to correct these by ISS0 shifts? And/or TTL pulses?

            stimuli_data = pd.read_csv(str(header_segment).replace("header", "stimuli"), header=0, sep="\t")
            elements.extend(trial_data["StimNum"].tolist())

            # use the nidq time instead of csv time
            segment_trial_starts = trial_times_from_nidq[segment_numbers_from_nidq == segment_number_from_file_name, 0]
            element_times.extend(segment_trial_starts + stimuli_data["Time_ms"] / 1e3)

            amplitude.extend(stimuli_data["Ampl"])
            ordinality.extend(stimuli_data["Posn"])

            elem_piezo = {x: header_data[x].values[0] for x in header_data if "ElemPiezo" in x}
            elem_piezo_labels = {x: header_data[x].values[0] for x in header_data if "PiezoLabel" in x}
            assert len(elem_piezo) == len(elem_piezo_labels), "Size mismatch between element piezo and piezo labels!"
            element_index_label_pairs.extend(
                [[elem_piezo[x], elem_piezo_labels[y]] for x, y in zip(elem_piezo, elem_piezo_labels)]
            )
        assert len(trial_numbers) == len(trial_starts), \
            "Mismatch between number of trials extracted from nidq file and number reported in trials.csv file!"
        assert (stimulus_numbers == elements).all(), \
            "Mismatch between stimulus numbers extracted from nidq file and those reported in trials.csv file!"

        element_index_label_map = dict(np.unique(element_index_label_pairs, axis=0))
        annotated_events = AnnotatedEventsTable(
            name="StimulusEvents",
            description="Stimulus elements, events, and properties.",
            resolution=np.nan
        )
        annotated_events.add_column(
            name="amplitude",
            description="Amplitude (microns) of element.",
            index=True
        )
        annotated_events.add_column(
            name="ordinality",
            description="Ordinal position of this element within the trial. E.g., '1' is the first stimulus in trial.",
            index=True
        )
        annotated_events.add_column(
            name="element_id",
            description="The assigned element ID.",
            index=True
        )
        annotated_events.add_column(
            name="element_name",
            description="The custom label for the element ID.",
            index=True
        )
        # event_times can't be compressed, returned error
        # Tried adding compression to other columns, and it took a H5DataIO type, but didn't actually perform it
        annotated_events.add_event_type(
            label="Element",
            event_description="Time (in seconds) when an element was presented.",
            event_times=element_times,
            amplitude=amplitude,
            ordinality=ordinality,
            element_id=elements,
            element_name=[element_index_label_map[str(x)] for x in elements]
        )
        check_module(nwbfile=nwbfile, name="events", description="Processed event data.").add(annotated_events)

        # use trial times from nidq file
        for k in range(len(trial_times_from_nidq)):
            nwbfile.add_trial(start_time=trial_times_from_nidq[k, 0], stop_time=trial_times_from_nidq[k, 1])
