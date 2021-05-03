"""Authors: Cody Baker."""
from pathlib import Path
from datetime import timedelta, datetime

# from isodate import duration_isoformat

from feldman_lab_to_nwb import FeldmanNWBConverter

# Point to the base folder path for both recording data and Virmen
base_path = Path("E:/Feldman")

# Name the NWBFile and point to the desired save path
nwbfile_path = base_path / "EarlyTesting.nwb"

# Point to the various files for the conversion
raw_data_path = base_path / "Neuropixels_Feldman" / "210209" / "SpikeGLX"
raw_session_name = "LR_210209_g0"
# raw_session_name = "LR_210209_g1"
# raw_session_name = "LR_210209_2_g0"
# raw_session_name = "LR_210209_2_g1"
raw_data_file = (
    raw_data_path / raw_session_name / f"{raw_session_name}_imec0" / f"{raw_session_name}_t0.imec0.ap.bin"
)
lfp_data_file = raw_data_file.parent / raw_data_file.name.replace("ap", "lf")
behavior_folder_path = base_path / "Neuropixels_Feldman" / "210209"  / "ADRIAN"
nidq_synch_file = str(raw_data_path / raw_session_name / f"{raw_session_name}_t0.nidq.bin")

# Enter Session and Subject information here - uncomment any fields you want to include
session_description = "Enter session description here."
# session_start_time = datetime(1970, 1, 1)  # not necessary if writing raw recording data (SpikeGLX sets it)

subject_info = dict(
    subject_id="Name of experimental subject",  # Required for upload to DANDDI
    # description="Enter optional subject description here",
    # weight="Enter subject weight here",
    # age=duration_isoformat(timedelta(days=0)),  # Enter the age of the subject in days
    # species="Mus musculus",
    # genotype="Enter subject genotype here",
    # sex="Enter subject sex here"
)

# Set some global conversion options here
stub_test = True


# Run the conversion
source_data = dict(
#    SpikeGLXRecording=dict(file_path=str(raw_data_file)),
#    SpikeGLXLFP=dict(file_path=str(lfp_data_file)),
    Behavior=dict(folder_path=str(behavior_folder_path))
)
conversion_options = dict(
#    SpikeGLXRecording=dict(stub_test=stub_test),
#    SpikeGLXLFP=dict(stub_test=stub_test),
    Behavior=dict(nidq_synch_file=nidq_synch_file)
)
converter = FeldmanNWBConverter(source_data=source_data)
metadata = converter.get_metadata()
metadata["NWBFile"].update(session_description=session_description)
metadata.update(Subject=subject_info)
converter.run_conversion(
    nwbfile_path=str(nwbfile_path),
    metadata=metadata,
    conversion_options=conversion_options,
    overwrite=True
)
