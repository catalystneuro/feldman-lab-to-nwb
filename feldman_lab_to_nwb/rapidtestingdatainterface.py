"""Authors: Cody Baker."""
from pathlib import Path

import nwb_conversion_tools
from nwb_conversion_tools import SpikeGLXRecordingInterface
from pynwb import NWBFile
from spikeextractors import NwbRecordingExtractor, SpikeGLXRecordingExtractor

from .utils import get_trials_info


def infer_ap_filepath(nidq_file_path: str):
    session_name = Path(nidq_file_path).parent.stem
    ap_filepath = nidq_file_path.replace("_t0.nidq.bin", f"_imec0/{session_name}_t0.imec0.ap.bin")
    return ap_filepath


class RapidTestingDataInterface(SpikeGLXRecordingInterface):
    """Conversion class for the Feldman lab behavioral data."""

    def get_metadata(self):
        electrode_interface = SpikeGLXRecordingInterface(
            file_path=infer_ap_filepath(nidq_file_path=self.source_data["file_path"])
        )
        metadata = electrode_interface.get_metadata()
        metadata["Ecephys"]["Electrodes"] = []
        return electrode_interface.get_metadata()

    def get_conversion_options(self):
        return dict()

    def run_conversion(self, nwbfile: NWBFile, metadata: dict, trial_ongoing_channel: int, event_channel: int):
        """
        Rapid conversion of trial information recovered from the nidq file.

        Does not require the more detailed behavioral csv files.
        """
        electrode_extractor = SpikeGLXRecordingExtractor(
            file_path=infer_ap_filepath(nidq_file_path=self.source_data["file_path"])
        )
        nwb_conversion_tools.utils.spike_interface.add_devices(recording=electrode_extractor, nwbfile=nwbfile, metadata=metadata)
        nwb_conversion_tools.utils.spike_interface.add_electrode_groups(recording=electrode_extractor, nwbfile=nwbfile, metadata=metadata)
        nwb_conversion_tools.utils.spike_interface.add_electrodes(recording=electrode_extractor, nwbfile=nwbfile, metadata=metadata)

        trial_numbers, stimulus_numbers, trial_times_from_nidq = get_trials_info(
            recording_nidq=self.recording_extractor,
            trial_ongoing_channel=trial_ongoing_channel,
            event_channel=event_channel,
        )

        nwbfile.add_trial_column(name="stim_number", description="The identifier value for stimulus type.")
        for k in range(len(trial_times_from_nidq)):
            nwbfile.add_trial(
                start_time=trial_times_from_nidq[k, 0],
                stop_time=trial_times_from_nidq[k, 1],
                stim_number=stimulus_numbers[k]
            )
