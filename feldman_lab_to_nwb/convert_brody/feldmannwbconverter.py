"""Authors: Cody Baker."""
from nwb_conversion_tools import NWBConverter, SpikeGLXRecordingInterface

from .feldmanbehaviordatainterface import FeldmanBehaviorDataInterface


class FeldmanNWBConverter(NWBConverter):
    """Primary conversion class for the Feldman lab processing pipeline."""

    data_interface_classes = dict(
        SpikeGLXRecording=SpikeGLXRecordingInterface,
        Behavior=FeldmanBehaviorDataInterface
    )

    def get_metadata(self):
        # TODO
        raise NotImplementedError("Not built yet!")
