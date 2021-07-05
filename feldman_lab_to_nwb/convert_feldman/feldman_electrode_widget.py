import numpy as np

from ipywidgets import widgets, Layout
from pynwb.misc import Units
from nwbwidgets import default_neurodata_vis_spec
from nwbwidgets.misc import PSTHWidget
from nwbwidgets import base
import plotly.graph_objects as go

from .feldman_electrode_widget_utils import calculate_response

ELECTRODE_COLOR = "#000000"

ELECTRODE_SIZE = 3
DETECTED_SIZE = 10
SELECTED_SIZE = 20


class ElectrodePositionSelector(widgets.VBox):

    def update_point(self, trace, points, selector):
        pass

    def __init__(self, electrodes, psth_info):
        super().__init__()
        x = electrodes["rel_x"].data[:]
        y = electrodes["rel_y"].data[:]
        n_channels = len(x)

        nwbfile = electrodes.get_ancestor("NWBFile")
        unit_ids = nwbfile.units.id.data[:]

        unit_response, _ = calculate_response(units=nwbfile.units, trials=nwbfile.trials, **psth_info)
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
                        f"Channel ID: {channel_id}, Responsitivity: {round(response, 2)}"
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

    def update_response(self, electrodes, psth_info):
        unit_ids = electrodes.get_ancestor("NWBFile").units.id.data[:]
        n_channels = len(self.scatter.marker.size)

        nwbfile = electrodes.get_ancestor("NWBFile")
        response, _ = calculate_response(units=nwbfile.units, trials=nwbfile.trials, **psth_info)
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
                pre_alignment_window=[0, self.psth_widget.before_ft.value],
                evoked_window=[0.1, 0.1 + self.psth_widget.after_ft.value],
                event_name=self.psth_widget.trial_event_controller.value,
                cat=self.psth_widget.gas.children[0].value
            ),
        )
        self.electrode_position_selector.scatter.on_click(self.update_point)

        self.responsitivity_evoked_offset = widgets.FloatText(
            0.0,
            min=0,
            description="offset",
            layout=Layout(width="200px")
        )

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
        self.psth_widget.gas.children[0].observe(self.handle_response, "value")

    def handle_unit_controller(self, change):
        self.electrode_position_selector.update_selected(electrodes=self.electrodes, index=change["owner"].index)

    def handle_response(self, change):
        self.electrode_position_selector.update_response(
            electrodes=self.electrodes,
            psth_info=dict(
                pre_alignment_window=[0, self.psth_widget.before_ft.value],
                evoked_window=[
                    self.responsitivity_evoked_offset.value,
                    self.psth_widget.after_ft.value
                ],
                event_name=self.psth_widget.trial_event_controller.value,
                cat=self.psth_widget.gas.children[0].value
            )
        )


default_neurodata_vis_spec[Units]["Grouped PSTH"] = PSTHWithElectrodeSelector


def nwb2widget(node, neurodata_vis_spec=default_neurodata_vis_spec):
    return base.nwb2widget(node, neurodata_vis_spec)
