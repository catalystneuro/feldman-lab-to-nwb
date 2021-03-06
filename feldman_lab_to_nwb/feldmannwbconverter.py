"""Authors: Cody Baker."""
from pathlib import Path
from pandas import read_csv

from spikeextractors import SpikeGLXRecordingExtractor
from nwb_conversion_tools import NWBConverter, SpikeGLXRecordingInterface, SpikeGLXLFPInterface

from .feldmanbehaviordatainterface import FeldmanBehaviorDataInterface
from .utils import get_trials_info, clip_recording


class FeldmanNWBConverter(NWBConverter):
    """Primary conversion class for the Feldman lab processing pipeline."""

    data_interface_classes = dict(
        SpikeGLXRecording=SpikeGLXRecordingInterface,
        SpikeGLXLFP=SpikeGLXLFPInterface,
        Behavior=FeldmanBehaviorDataInterface
    )

    def __init__(self, source_data: dict, nidq_synch_file: str, trial_ongoing_channel: int, event_channel: int):
        if "Behavior" in source_data:
            source_data["Behavior"].update(
                nidq_synch_file=nidq_synch_file,
                trial_ongoing_channel=trial_ongoing_channel,
                event_channel=event_channel
            )
        super().__init__(source_data=source_data)
        trial_numbers, _, trial_times = get_trials_info(
            recording_nidq=SpikeGLXRecordingExtractor(nidq_synch_file),
            trial_ongoing_channel=trial_ongoing_channel,
            event_channel=event_channel
        )
        if trial_numbers[0] != 0:
            for interface in set(["SpikeGLXRecording", "SpikeGLXLFP"]).intersection(self.data_interface_objects):
                self.data_interface_objects[interface].recording_extractor = clip_recording(
                    trial_numbers=trial_numbers,
                    trial_times=trial_times,
                    recording=self.data_interface_objects[interface].recording_extractor
                )
                self.data_interface_objects[interface].recording_extractor.set_channel_gains = (
                    self.data_interface_objects[interface].recording_extractor._parent_recording.get_channel_gains()
                )

    def get_metadata(self):
        behavior_folder_path = Path(self.data_interface_objects["Behavior"].source_data["folder_path"])
        metadata = super().get_metadata()
        metadata["NWBFile"].update(institution="UC Berkeley", lab="Feldman")
        if "session_id" not in metadata["NWBFile"]:
            metadata["NWBFile"].update(session_id="_".join(next(behavior_folder_path.iterdir()).stem.split("_")[:3]))
        return metadata
