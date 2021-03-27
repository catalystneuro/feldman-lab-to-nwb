"""Authors: Cody Baker."""
import pandas as pd
from pathlib import Path
from warnings import warn
import re
import numpy as np

from ndx_events import AnnotatedEventsTable, LabeledEvents
from hdmf.backends.hdf5.h5_utils import H5DataIO
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.conversion_tools import check_module
from pynwb import NWBFile


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

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Feldman lab behavioral interface."""
        folder_path = Path(self.source_data["folder_path"])
        header_segments = [x for x in folder_path.iterdir() if "header" in x.name]

        stim_layout_str = ["Std", "Trains", "IL", "Trains+IL", "RFMap", "2WC", "MWS", "MWD"]
        # stimuli_header_vals = [
        #     f"{x}stimuli" for x in ["N", "TrStimN", "ILStimN", "TrStimN", "RFStimN", "2WCStimN", "MWStimN", "MWStimN"]
        # ]

        stim_layout_per_seg = []
        n_trials_per_seg = []
        elements = []
        element_times = []
        amplitude = []
        ordinality = []
        element_index_label_pairs = []
        trial_starts = []
        trial_ends = []
        for header_segment in header_segments:
            header_data = pd.read_csv(header_segment, header=None, sep="\t", index_col=0).T

            # TODO: figure out how to get this upstream in metadata
            session_name = header_data["ExptName"]

            segment_number = header_data["SegmentNum"]
            p = re.compile("_S(\d+)_")
            res = p.search(header_segment.name)
            if res is not None and segment_number.values[0] != res.group(1):
                warn(
                    f"Segment number in file name ({header_segment.name}) does not match internal value!"
                    "Using file name."
                )

            n_trials_per_seg.append(int(header_data["LastTrialNum"]) - int(header_data["FirstTrialNum"]))

            stim_layout = int(header_data["StimLayout"].values[0]) - 1  # -1 for zero-indexing
            if stim_layout != 0:
                raise NotImplementedError("StimLayouts other than 'Std' type have not yet been implemented!")
            stim_layout_per_seg.append(stim_layout_str[stim_layout])

            trial_data = pd.read_csv(str(header_segment).replace("header", "trials"), header=0, sep="\t")
            trial_starts.extend(trial_data['TrStartTime'] / 1e3)
            trial_ends.extend(trial_data['TrEndTime'] / 1e3)
            # may have to correct these by ISS0 shifts? And/or TTL pulses?

            stimuli_data = pd.read_csv(str(header_segment).replace("header", "stimuli"), header=0, sep="\t")
            elements.extend(trial_data["StimNum"].tolist())
            element_times.extend(trial_data["StimOnsetTime"] / 1e3)
            amplitude.extend(stimuli_data["Ampl"])
            ordinality.extend(stimuli_data["Posn"])

            elem_piezo = {x: header_data[x].values[0] for x in header_data if "ElemPiezo" in x}
            elem_piezo_labels = {x: header_data[x].values[0] for x in header_data if "PiezoLabel" in x}
            assert len(elem_piezo) == len(elem_piezo_labels), "Size mismatch between element piezo and piezo labels!"
            element_index_label_pairs.extend(
                [[elem_piezo[x], elem_piezo_labels[y]] for x, y in zip(elem_piezo, elem_piezo_labels)]
            )

        # The LabeledEvents series can more efficiently compress/represent the stimulus series data
        # but cannot include custom metadata such as amplitude/ordinality. Also fully supported compression.
        #
        # unique_events = np.unique(elements)
        # unique_element_index_label_pairs = np.unique(element_index_label_pairs, axis=0)
        # unique_labels = [x[1] for x in unique_element_index_label_pairs if int(x[0]) in unique_events]
        # events_map = {event: n for n, event in enumerate(unique_events)}
        # event_data = [events_map[event] for event in elements]
        # events = LabeledEvents(
        #     name="StimulusEvents",
        #     description="Stimulus events from the experiment.",
        #     timestamps=H5DataIO(element_times, compression="gzip"),
        #     resolution=np.nan,
        #     data=H5DataIO(event_data, compression="gzip"),
        #     labels=unique_labels
        # )
        # nwbfile.add_acquisition(events)

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

        for k in range(len(trial_starts)):
            nwbfile.add_trial(start_time=trial_starts[k], stop_time=trial_ends[k])
