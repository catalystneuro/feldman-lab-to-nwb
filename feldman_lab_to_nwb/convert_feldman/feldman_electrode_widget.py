import numpy as np

from ipywidgets import widgets
from pynwb import NWBFile
from pynwb.misc import Units
from nwbwidgets import default_neurodata_vis_spec
from nwbwidgets.misc import PSTHWidget
from nwbwidgets import base
import plotly.graph_objects as go

ELECTRODE_COLOR = "#000000"

ELECTRODE_SIZE = 3
DETECTED_SIZE = 10
SELECTED_SIZE = 20


def calculate_response(
        nwbfile: NWBFile,
        pre_trial_window: float = 10.,
        evoked_window: float = 1.,
        stim_offset: float = .1,
        std_threshold: float = 1e-6
):
    """
    Calculate the reponse of units from the table of the NWBFile.

    Evaluates as the difference between pre-trial and post-presentation
    spiking rates scaled by the pre-stimulus noise levels.

    Parameters
    ----------
    nwbfile : NWBFile
        Source of the units table.
    pre_trial_window : float, optional
        Length of time to evaluate pre-trial spiking over, in seconds.
    evoked_window : float, optional
        Length of time to evaluate evoked spiking over, in seconds.
    stim_offset : float, optional
        The amount of time in seconds after the start of trial that evokes spiking activity.
    std_threshold : float, optional
        Sets all values with a standard deviation in spiking rate below this threshold to NaN.
    """
    units_spk_time = nwbfile.units.spike_times.data[:]
    units_spk_time_index = nwbfile.units.spike_times_index.data[:]
    trial_starts = nwbfile.trials.start_time.data[:]

    unit_avg_evoked_spk_rate = []
    unit_avg_prestim_spk_rate = []
    unit_std_prestim_spk_rate = []
    response = []
    for j, _ in enumerate(units_spk_time_index):
        spk_idx_lb = 0 if j == 0 else units_spk_time_index[j-1]
        spks = units_spk_time[spk_idx_lb:units_spk_time_index[j]]
        evoked_spk_rate = []
        prestim_spk_rate = []
        for trial_start in trial_starts:
            evoked_spk_rate.append(
                sum(
                    (spks >= trial_start + stim_offset) & (spks < trial_start + stim_offset + evoked_window)
                ) / evoked_window
            )
            prestim_spk_rate.append(
                sum((spks >= trial_start - pre_trial_window) & (spks < trial_start)) / pre_trial_window
            )

        valid_trials = np.array(prestim_spk_rate) != 0.0
        unit_avg_evoked_spk_rate.append(np.mean(evoked_spk_rate, where=valid_trials))
        unit_avg_prestim_spk_rate.append(np.mean(prestim_spk_rate, where=valid_trials))
        unit_std_prestim_spk_rate.append(np.std(prestim_spk_rate, where=valid_trials))

    for j, _ in enumerate(units_spk_time_index):
        if unit_std_prestim_spk_rate[j] <= std_threshold:
            response.append(np.nan)
        else:
            response.append((unit_avg_evoked_spk_rate[j] - unit_avg_prestim_spk_rate[j]) / unit_std_prestim_spk_rate[j])
    return np.array(response)


class ElectrodePositionSelector(widgets.VBox):

    def update_point(self, trace, points, selector):
        pass

    def __init__(self, electrodes):
        super().__init__()
        x = electrodes["rel_x"].data[:]
        y = electrodes["rel_y"].data[:]
        n_channels = len(x)

        unit_ids = electrodes.get_ancestor("NWBFile").units.id.data[:]

        response = calculate_response(nwbfile=electrodes.get_ancestor("NWBFile"))
        all_response = np.array([np.nan] * n_channels)
        all_response[unit_ids] = response

        self.fig = go.FigureWidget(
            [
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    text=[f"Channel ID: {j}" for j in electrodes.id.data[:]],
                    marker=dict(
                        colorbar=dict(title="Responsitivity"),
                        cmax=max(response[~np.isnan(response)]),
                        cmin=min(response[~np.isnan(response)]),
                        color=all_response,
                        colorscale="Viridis"
                    ),
                )
            ]
        )
        self.scatter = self.fig.data[0]
        s = np.array([ELECTRODE_SIZE] * n_channels)
        s[unit_ids] = DETECTED_SIZE
        s[unit_ids[0]] = SELECTED_SIZE
        self.scatter.marker.size = s

        self.scatter.on_click(self.update_point)
        self.fig.layout.hovermode = "closest"
        self.fig.update_layout(
            title=dict(
                text="electrode grid",
                xanchor="center",
                yanchor="top",
                y=0.9,
                x=0.5,
                font=dict(family="monospace", size=14)
            ),
            autosize=False,
            width=420,
            height=640,
            xaxis=dict(showline=False, showticklabels=False, ticks=""),
            yaxis=dict(showline=False, showticklabels=False, ticks="")
        )

        self.children = [self.fig]

    def update(self, electrodes, index: int = 0):
        n_channels = len(self.scatter.marker.size)
        units_idx = np.where(np.array(self.scatter.marker.size) >= 10)[0]

        s = np.array([ELECTRODE_SIZE] * n_channels)
        s[units_idx] = DETECTED_SIZE
        s[units_idx[index]] = SELECTED_SIZE

        with self.fig.batch_update():
            self.scatter.marker.size = s


class PSTHWithElectrodeSelector(widgets.HBox):

    def update_point(self, trace, points, selector):
        n_channels = len(self.electrode_position_selector.scatter.marker.size)
        is_unit = np.array(self.electrode_position_selector.scatter.marker.size) >= 10
        index = points.point_inds[0]
        if is_unit[index]:
            s = np.array([3] * n_channels)
            s[is_unit] = DETECTED_SIZE
            s[index] = SELECTED_SIZE

            with self.electrode_position_selector.fig.batch_update():
                self.electrode_position_selector.scatter.marker.size = s

        self.psth_widget.unit_controller.value = np.where(
            np.array(self.electrode_position_selector.scatter.marker.size)[is_unit] == 15
        )[0][0]


    def __init__(self, units):
        super().__init__()
        self.electrodes = units.get_ancestor("NWBFile").electrodes

        self.psth_widget = PSTHWidget(units)
        self.electrode_position_selector = ElectrodePositionSelector(self.electrodes)
        self.electrode_position_selector.scatter.on_click(self.update_point)

        self.children = [self.psth_widget, self.electrode_position_selector]
        self.psth_widget.unit_controller.observe(self.handle_unit_controller, "value")

    def handle_unit_controller(self, change):
        self.electrode_position_selector.update(electrodes=self.electrodes, index=change["owner"].index)


default_neurodata_vis_spec[Units]["Grouped PSTH"] = PSTHWithElectrodeSelector


def nwb2widget(node, neurodata_vis_spec=default_neurodata_vis_spec):
    return base.nwb2widget(node, neurodata_vis_spec)
