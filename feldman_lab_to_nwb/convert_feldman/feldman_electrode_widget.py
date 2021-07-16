"""Authors: Cody Baker."""
import numpy as np
from typing import Iterable

from ipywidgets import widgets
from pynwb.file import ElectrodeTable
from pynwb.misc import Units
from nwbwidgets import default_neurodata_vis_spec
from nwbwidgets.misc import PSTHWidget
from nwbwidgets import base
import plotly.graph_objects as go

from .feldman_electrode_widget_utils import calculate_all_responses

ELECTRODE_COLOR = "#000000"

ELECTRODE_SIZE = 3
DETECTED_SIZE = 10
SELECTED_SIZE = 20


class ElectrodePositionSelector(widgets.VBox):
    """Custom widget object for selecting channel activity from a visual grid."""

    def update_point(self, trace, points, selector):
        pass

    def __init__(
        self,
        electrodes: ElectrodeTable,
        pre_alignment_window: Iterable[float],
        post_alignment_window: Iterable[float],
        event_name: str
    ):
        """
        Visualize the electrode grid with on-click selection of channels in one-to-one relationship to units.

        Parameters
        ----------
        electrodes: ElectrodeTable
            Electrodes Table of an NWBFile.
        pre_alignment_window : Iterable[float]
            Array-like of the form [start, end] flipped with respect to the value of the alignment time,
            in units seconds. E.g., pre_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
            will calculate spiking activity over the windows [[0.7, 1.1], [1.8, 2.2], [2.9, 3.3]] respectively.
        post_alignment_window : Iterable[float]
            Array-like of the form [start, end] with respect to the value of the alignment time, in units seconds.
            E.g., post_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
            will calculate spiking activity over the windows [[1.3, 1.7], [2.4, 2.8], [3.5, 3.9]] respectively.
        event_name : str
            Name of event from trials table to align to.
        """
        super().__init__()
        x = electrodes["rel_x"].data[:]
        y = electrodes["rel_y"].data[:]
        n_channels = len(x)

        nwbfile = electrodes.get_ancestor("NWBFile")
        unit_ids = nwbfile.units.id.data[:]

        unit_response = calculate_all_responses(
            units=nwbfile.units,
            trials=nwbfile.trials,
            pre_alignment_window=pre_alignment_window,
            post_alignment_window=post_alignment_window,
            event_name=event_name
        )
        valid_unit_response = unit_response[~np.isnan(unit_response)]
        channel_response = np.array([np.nan] * n_channels)
        channel_response[unit_ids] = unit_response

        self.fig = go.FigureWidget(
            [
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    text=[
                        f"Channel ID: {channel_id} <br> Responsitivity: {round(response, 2)}"
                        for channel_id, response in zip(electrodes.id.data[:], channel_response)
                    ],
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

    def update_response(
        self,
        electrodes: ElectrodeTable,
        pre_alignment_window: Iterable[float],
        post_alignment_window: Iterable[float],
        event_name: str
    ):
        """
        Visualize the electrode grid with on-click selection of channels in one-to-one relationship to units.

        Parameters
        ----------
        electrodes: ElectrodeTable
            Electrodes Table of an NWBFile.
        pre_alignment_window : Iterable[float]
            Array-like of the form [start, end] flipped with respect to the value of the alignment time,
            in units seconds. E.g., pre_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
            will calculate spiking activity over the windows [[0.7, 1.1], [1.8, 2.2], [2.9, 3.3]] respectively.
        post_alignment_window : Iterable[float]
            Array-like of the form [start, end] with respect to the value of the alignment time, in units seconds.
            E.g., post_alignment_window = [0.1, 0.5] with alignment_times = [1.2, 2.3, 3.4]
            will calculate spiking activity over the windows [[1.3, 1.7], [2.4, 2.8], [3.5, 3.9]] respectively.
        event_name : str
            Name of event from trials table to align to.
        """
        unit_ids = electrodes.get_ancestor("NWBFile").units.id.data[:]
        n_channels = len(self.scatter.marker.size)

        nwbfile = electrodes.get_ancestor("NWBFile")
        response, _ = calculate_all_responses(
            units=nwbfile.units,
            trials=nwbfile.trials,
            pre_alignment_window=pre_alignment_window,
            post_alignment_window=post_alignment_window,
            event_name=event_name
        )
        channel_response = np.array([np.nan] * n_channels)
        channel_response[unit_ids] = response

        self.scatter.marker.cmax = max(response[~np.isnan(response)])
        self.scatter.marker.cmin = min(response[~np.isnan(response)])
        self.scatter.marker.color = channel_response

        self.scatter.text = [
            f"Channel ID: {channel_id}, Responsitivity: {round(response, 2)}"
            for channel_id, response in zip(electrodes.id.data[:], channel_response)
        ]


class PSTHWithElectrodeSelector(widgets.HBox):
    """Custom widget for combining PSTH tab with the electrode grid selector, allowing communication between the two."""

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

    def __init__(self, units: Units):
        """
        Visualize PSTH with the ability to select electrodes from clicking within the grid.

        Parameters
        ----------
        units : Units
            Units Table of an NWBFile.
        """
        super().__init__()
        self.electrodes = units.get_ancestor("NWBFile").electrodes

        self.psth_widget = PSTHWidget(units)
        self.electrode_position_selector = ElectrodePositionSelector(
            self.electrodes,
            pre_alignment_window=[0, self.psth_widget.before_ft.value],
            post_alignment_window=[0, self.psth_widget.after_ft.value],
            event_name=self.psth_widget.trial_event_controller.value
        )
        self.electrode_position_selector.scatter.on_click(self.update_point)

        self.children = [
            self.psth_widget,
            widgets.VBox([
                self.electrode_position_selector,
                self.responsitivity_evoked_offset
            ])
        ]
        self.psth_widget.unit_controller.observe(self.handle_unit_controller, "value")
        self.psth_widget.before_ft.observe(self.handle_response, "value")
        self.psth_widget.after_ft.observe(self.handle_response, "value")
        self.psth_widget.trial_event_controller.observe(self.handle_response, "value")

    def handle_unit_controller(self, change):
        self.electrode_position_selector.update_selected(electrodes=self.electrodes, index=change["owner"].index)

    def handle_response(self, change):
        self.electrode_position_selector.update_response(
            electrodes=self.electrodes,
            responsitivity_params=dict(
                pre_alignment_window=[0, self.psth_widget.before_ft.value],
                evoked_window=[0, self.psth_widget.after_ft.value],
                event_name=self.psth_widget.trial_event_controller.value,
                cat=self.psth_widget.gas.children[0].value
            )
        )


default_neurodata_vis_spec[Units]["Grouped PSTH"] = PSTHWithElectrodeSelector


def nwb2widget(node, neurodata_vis_spec=default_neurodata_vis_spec):
    return base.nwb2widget(node, neurodata_vis_spec)
