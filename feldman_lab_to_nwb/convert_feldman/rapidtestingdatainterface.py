"""Authors: Cody Baker."""
from nwb_conversion_tools import SpikeGLXRecordingInterface
from pynwb import NWBFile

from .feldman_utils import get_trials_info


class RapidTestingDataInterface(SpikeGLXRecordingInterface):
    """Conversion class for the Feldman lab behavioral data."""

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """
        Rapid conversion of trial information recovered from the nidq file.

        Does not require the more detailed behavioral csv files.
        """
        (trial_numbers, stimulus_numbers, segment_numbers_from_nidq, trial_times_from_nidq) = get_trials_info(
            recording_nidq=self.recording_extractor
        )

        nwbfile.add_trial_column(name="stim_number", description="The identifier value for stimulus type.")
        for k in range(len(trial_times_from_nidq)):
            nwbfile.add_trial(
                start_time=trial_times_from_nidq[k, 0],
                stop_time=trial_times_from_nidq[k, 1],
                stim_number=stimulus_numbers[k]
            )
