"""Authors: Cody Baker."""
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from pynwb import NWBFile


class FeldmanBehaviorDataInterface(BaseDataInterface):
    """Conversion class for the Brody lab behavioral data."""

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        # TODO
        raise NotImplementedError("Not built yet!")

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Brody lab behavioral interface."""
        # TODO
        raise NotImplementedError("Not built yet!")
