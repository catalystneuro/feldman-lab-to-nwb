"""Authors: Cody Baker."""
from pathlib import Path

from nwb_conversion_tools import NWBConverter, SpikeGLXRecordingInterface, SpikeGLXLFPInterface

from .feldmanbehaviordatainterface import FeldmanBehaviorDataInterface


class FeldmanNWBConverter(NWBConverter):
    """Primary conversion class for the Feldman lab processing pipeline."""

    data_interface_classes = dict(
        SpikeGLXRecording=SpikeGLXRecordingInterface,
        SpikeGLXLFP=SpikeGLXLFPInterface,
        Behavior=FeldmanBehaviorDataInterface
    )

    def get_metadata(self):
        behavior_folder_path = Path(self.data_interface_objects["Behavior"].source_data["folder_path"])
        metadata = super().get_metadata()
        metadata["NWBFile"].update(institution="UC Berkeley", lab="Feldman")
        if "session_id" not in metadata["NWBFile"]:
            metadata["NWBFile"].update(session_id="_".join(next(behavior_folder_path.iterdir()).stem.split("_")[:3]))
        return metadata
