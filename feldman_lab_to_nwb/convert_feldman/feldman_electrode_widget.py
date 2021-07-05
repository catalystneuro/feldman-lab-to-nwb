import numpy as np
from typing import Optional

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
    pre_trial_window: float,
    evoked_window: float,
    event_name: str,
    cat: Optional[str] = None,
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
    pre_trial_window : float
        Length of time to evaluate pre-trial spiking over, in seconds.
    evoked_window : float
        Length of time to evaluate evoked spiking over, in seconds.
    event_name : str
        Name of event from trials table to align to.
    cat : str, optional
        Categorial column of the trials table to take condition responsitivity over.
        If not None, will return maximum response over categories instead of unconditional.
    std_threshold : float, optional
        Sets all values with a standard deviation in spiking rate below this threshold to NaN.
    """
    units_spk_time = nwbfile.units.spike_times.data[:]
    units_spk_time_index = nwbfile.units.spike_times_index.data[:]
    alignment_times = getattr(nwbfile.trials, event_name).data[:]
    unique_cats = set()
    if cat is not None:
        cat_data = getattr(nwbfile.trials, cat).data[:]
        unique_cats.update(cat_data)

    unit_avg_evoked_spk_rate = []
    unit_avg_prestim_spk_rate = []
    unit_std_prestim_spk_rate = []
    unit_avg_evoked_spk_rate_per_cat = []
    unit_avg_prestim_spk_rate_per_cat = []
    unit_std_prestim_spk_rate_per_cat = []
    avg_response = []
    response_per_cat = []
    max_response = []
    for j, _ in enumerate(units_spk_time_index):
        spk_idx_lb = 0 if j == 0 else units_spk_time_index[j-1]
        spks = units_spk_time[spk_idx_lb:units_spk_time_index[j]]
        evoked_spk_rate = []
        prestim_spk_rate = []
        for alignment_time in alignment_times:
            evoked_spk_rate.append(
                sum((spks >= alignment_time) & (spks < alignment_time + evoked_window)) / evoked_window
            )
            prestim_spk_rate.append(
                sum((spks >= alignment_time - pre_trial_window) & (spks < alignment_time)) / pre_trial_window
            )

        valid_trials = np.array(prestim_spk_rate) != 0.0
        unit_avg_evoked_spk_rate_per_cat.append({x: [] for x in unique_cats})
        unit_avg_prestim_spk_rate_per_cat.append({x: [] for x in unique_cats})
        unit_std_prestim_spk_rate_per_cat.append({x: [] for x in unique_cats})
        if any(valid_trials):
            unit_avg_evoked_spk_rate.append(np.mean(evoked_spk_rate, where=valid_trials))
            unit_avg_prestim_spk_rate.append(np.mean(prestim_spk_rate, where=valid_trials))
            unit_std_prestim_spk_rate.append(np.std(prestim_spk_rate, where=valid_trials))
            for unique_cat in unique_cats:
                valid_cat = valid_trials & cat_data == unique_cat
                if any(valid_cat):
                    unit_avg_evoked_spk_rate_per_cat[-1][unique_cat] = np.mean(evoked_spk_rate, where=valid_cat)
                    unit_avg_prestim_spk_rate_per_cat[-1][unique_cat] = np.mean(prestim_spk_rate, where=valid_cat)
                    unit_std_prestim_spk_rate_per_cat[-1][unique_cat] = np.std(prestim_spk_rate, where=valid_cat)
                else:
                    unit_avg_evoked_spk_rate_per_cat[-1][unique_cat] = np.nan
                    unit_avg_prestim_spk_rate_per_cat[-1][unique_cat] = np.nan
                    unit_std_prestim_spk_rate_per_cat[-1][unique_cat] = np.nan
        else:
            unit_avg_evoked_spk_rate.append(np.nan)
            unit_avg_prestim_spk_rate.append(np.nan)
            unit_std_prestim_spk_rate.append(np.nan)
            for unique_cat in unique_cats:
                unit_avg_evoked_spk_rate_per_cat[-1][unique_cat] = np.nan
                unit_avg_prestim_spk_rate_per_cat[-1][unique_cat] = np.nan
                unit_std_prestim_spk_rate_per_cat[-1][unique_cat] = np.nan

    for j, _ in enumerate(units_spk_time_index):
        if unit_std_prestim_spk_rate[j] <= std_threshold:
            avg_response.append(np.nan)
            if cat is not None:
                response_per_cat.append(np.nan)
                max_response.append(np.nan)
        else:
            avg_response.append(
                (unit_avg_evoked_spk_rate[j] - unit_avg_prestim_spk_rate[j]) / unit_std_prestim_spk_rate[j]
            )
            if cat is not None:
                response_per_cat.append({x: [] for x in unique_cats})
                for unique_cat in unique_cats:
                    response_per_cat[-1][unique_cat] = (
                        unit_avg_evoked_spk_rate_per_cat[j][unique_cat]
                        - unit_avg_prestim_spk_rate_per_cat[j][unique_cat]
                        ) / unit_std_prestim_spk_rate_per_cat[j][unique_cat]
                max_response.append(max(response_per_cat[-1].values()))

    internals = dict(
        evoked_spk_rate=evoked_spk_rate,
        prestim_spk_rate=prestim_spk_rate,
        unit_avg_evoked_spk_rate=unit_avg_evoked_spk_rate,
        unit_avg_prestim_spk_rate=unit_avg_prestim_spk_rate,
        unit_std_prestim_spk_rate=unit_std_prestim_spk_rate,
        response_per_cat=response_per_cat
    )
    if cat is None:
        internals.update(max_response=max_response)
        return np.array(avg_response), internals
    else:
        internals.update(avg_response=avg_response)
        return np.array(max_response), internals


class ElectrodePositionSelector(widgets.VBox):

    def update_point(self, trace, points, selector):
        pass

    def __init__(self, electrodes, psth_info):
        super().__init__()
        x = electrodes["rel_x"].data[:]
        y = electrodes["rel_y"].data[:]
        n_channels = len(x)

        unit_ids = electrodes.get_ancestor("NWBFile").units.id.data[:]

        unit_response, _ = calculate_response(nwbfile=electrodes.get_ancestor("NWBFile"), **psth_info)
        valid_unit_response = unit_response[~np.isnan(unit_response)]
        channel_response = np.array([np.nan] * n_channels)
        channel_response[unit_ids] = unit_response

        self.fig = go.FigureWidget(
            [
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    text=[f"Channel ID: {j}" for j in electrodes.id.data[:]],
                    marker=dict(
                        colorbar=dict(title="Responsitivity"),
                        cmax=max(valid_unit_response),
                        cmin=min(valid_unit_response),
                        color=channel_response,
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

    def update_selected(self, electrodes, index: int = 0):
        n_channels = len(self.scatter.marker.size)
        units_idx = np.where(np.array(self.scatter.marker.size) >= 10)[0]

        s = np.array([ELECTRODE_SIZE] * n_channels)
        s[units_idx] = DETECTED_SIZE
        s[units_idx[index]] = SELECTED_SIZE

        with self.fig.batch_update():
            self.scatter.marker.size = s

    def update_response(self, electrodes, psth_info):
        unit_ids = electrodes.get_ancestor("NWBFile").units.id.data[:]
        n_channels = len(self.scatter.marker.size)
        response, _ = calculate_response(nwbfile=electrodes.get_ancestor("NWBFile"), **psth_info)
        all_response = np.array([np.nan] * n_channels)
        all_response[unit_ids] = response

        self.scatter.marker.cmax = max(response[~np.isnan(response)])
        self.scatter.marker.cmin = min(response[~np.isnan(response)])
        self.scatter.marker.color = all_response


class PSTHWithElectrodeSelector(widgets.HBox):

    def update_point(self, trace, points, selector):
        n_channels = len(self.electrode_position_selector.scatter.marker.size)
        is_unit = np.array(self.electrode_position_selector.scatter.marker.size) >= DETECTED_SIZE
        index = points.point_inds[0]
        if is_unit[index]:
            s = np.array([ELECTRODE_SIZE] * n_channels)
            s[is_unit] = DETECTED_SIZE
            s[index] = SELECTED_SIZE

            with self.electrode_position_selector.fig.batch_update():
                self.electrode_position_selector.scatter.marker.size = s

        self.psth_widget.unit_controller.value = np.where(
            np.array(self.electrode_position_selector.scatter.marker.size)[is_unit] == SELECTED_SIZE
        )[0][0]

    def __init__(self, units):
        super().__init__()
        self.electrodes = units.get_ancestor("NWBFile").electrodes

        self.psth_widget = PSTHWidget(units)
        self.electrode_position_selector = ElectrodePositionSelector(
            self.electrodes,
            psth_info=dict(
                pre_trial_window=self.psth_widget.before_ft.value,
                evoked_window=self.psth_widget.after_ft.value,
                event_name=self.psth_widget.trial_event_controller.value,
                cat=self.psth_widget.gas.children[0].value
            ),
        )
        self.electrode_position_selector.scatter.on_click(self.update_point)

        self.children = [self.psth_widget, self.electrode_position_selector]
        self.psth_widget.unit_controller.observe(self.handle_unit_controller, "value")
        self.psth_widget.before_ft.observe(self.handle_response, "value")
        self.psth_widget.after_ft.observe(self.handle_response, "value")
        self.psth_widget.trial_event_controller.observe(self.handle_response, "value")
        self.psth_widget.gas.children[0].observe(self.handle_response, "value")

    def handle_unit_controller(self, change):
        self.electrode_position_selector.update_selected(electrodes=self.electrodes, index=change["owner"].index)

    def handle_response(self, change):
        self.electrode_position_selector.update_response(
            electrodes=self.electrodes,
            psth_info=dict(
                pre_trial_window=self.psth_widget.after_ft.value,
                evoked_window=self.psth_widget.after_ft.value,
                event_name=self.psth_widget.trial_event_controller.value,
                cat=self.psth_widget.gas.children[0].value
            )
        )


default_neurodata_vis_spec[Units]["Grouped PSTH"] = PSTHWithElectrodeSelector


def nwb2widget(node, neurodata_vis_spec=default_neurodata_vis_spec):
    return base.nwb2widget(node, neurodata_vis_spec)
