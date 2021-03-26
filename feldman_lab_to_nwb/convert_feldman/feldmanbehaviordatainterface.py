"""Authors: Cody Baker."""
import pandas as pd
from pathlib import Path
from warnings import warn
import re
import numpy as np

# from ndx_events import LabeledEvents
from hdmf.backends.hdf5.h5_utils import H5DataIO
from nwb_conversion_tools.basedatainterface import BaseDataInterface
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
        segments = [x for x in folder_path.iterdir() if "header" in x.name]

        stim_layout_str = ["Std", "Trains", "IL", "Trains+IL", "RFMap", "2WC", "MWS", "MWD"]
        stimuli_header_vals = [
            f"{x}stimuli" for x in ["N", "TrStimN", "ILStimN", "TrStimN", "RFStimN", "2WCStimN", "MWStimN", "MWStimN"]
        ]

        stim_layout_per_seg = []
        n_trials_per_seg = []
        n_stimuli_per_seg = []
        n_elements_per_seg = []
        trial_starts = []
        trial_ends = []
        for segment in segments:
            header_data = pd.read_csv(segment, header=None, sep="\t", index_col=0).T
            session_name = header_data["ExptName"]

            segment_number = header_data["SegmentNum"]
            p = re.compile("_S(\d+)_")
            res = p.search(segment.name)
            if res is not None and segment_number.values[0] != res.group(1):
                warn(f"Segment number in file name ({segment.name}) does not match internal value! Using file name.")

            n_trials_per_seg.append(int(header_data["LastTrialNum"]) - int(header_data["FirstTrialNum"]))

            stim_layout = int(header_data["StimLayout"].values[0]) - 1  # -1 for zero-indexing
            if stim_layout != 0:
                raise NotImplementedError("StimLayouts other than 'Std' type have not yet been implemented!")
            stim_layout_per_seg.append(stim_layout_str[stim_layout])

            n_stimuli_per_seg.append(int(header_data[stimuli_header_vals[stim_layout]].values[0]))
            n_elements_per_seg.append(int(header_data["Nelements"].values[0]))

            trial_data = pd.read_csv(str(segment).replace("header", "trials"), header=0, sep="\t")
            trial_starts.extend(trial_data['TrStartTime'] / 1e3)
            trial_ends.extend(trial_data['TrEndTime'] / 1e3)
            # may have to correct these by ISS0 shifts? And/or TTL pulses?

        # n_trials = sum(n_trials_per_seg)
        # n_stimuli = sum(n_stimuli_per_seg)
        # n_elements = sum(n_elements_per_seg)

        # event_timestamps = events_data['Time'].to_numpy() / 1E3
        # event_labels = events_data['Tag'].to_numpy()
        # unique_events = set(event_labels)
        # events_map = {event: n for n, event in enumerate(unique_events)}
        # event_data = [events_map[event] for event in event_labels]

        # events = LabeledEvents(
        #     name='StimulusEvents',
        #     description="Stimulus events from the experiment.",
        #     timestamps=H5DataIO(event_timestamps, compression="gzip"),
        #     resolution=np.nan,
        #     data=H5DataIO(event_data, compression="gzip"),
        #     labels=list(unique_events)
        # )
        # nwbfile.add_acquisition(events)

        for k in range(len(trial_starts)):
            nwbfile.add_trial(start_time=trial_starts[k], stop_time=trial_ends[k])
