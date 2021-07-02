"""Authors: Cody Baker."""
from datetime import datetime

from nwb_conversion_tools import NWBConverter

from .rapidtestingdatainterface import RapidTestingDataInterface


class RapidTestingNWBConverter(NWBConverter):
    """Primary conversion class for the rapid testing pipeline for the Feldman lab."""

    data_interface_classes = dict(RapidTesting=RapidTestingDataInterface)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["NWBFile"].update(institution="UC Berkeley", lab="Feldman")
        if "session_id" not in metadata["NWBFile"]:
            metadata["NWBFile"].update(
                session_id=f"Rapid_Testing_{str(datetime.now())}",
                session_start_time=datetime.now()
            )
        return metadata
