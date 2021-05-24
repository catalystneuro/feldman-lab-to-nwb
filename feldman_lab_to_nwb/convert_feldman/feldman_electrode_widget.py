import numpy as np

from ipywidgets import widgets
from pynwb.misc import Units
from nwbwidgets import default_neurodata_vis_spec
from nwbwidgets.misc import PSTHWidget
from nwbwidgets import base
import plotly.graph_objects as go


BACKGROUND_COLOR = "#9FE19D"
ELECTRODE_COLOR = "#d81700"
DETECTED_COLOR = "#002bd8"
SELECTED_COLOR = "#fccb00"


class ElectrodePositionSelector(widgets.VBox):

    def update_point(self, trace, points, selector):
        pass

    def __init__(self, electrodes):
        super().__init__()
        x = electrodes["rel_x"].data[:]
        y = electrodes["rel_y"].data[:]

        units_data = electrodes.get_ancestor("NWBFile").units.id.data[:]
        n_units = len(x)
        self.fig = go.FigureWidget(
            [
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    text=[f"Channel ID: {j}" for j in electrodes.id.data[:]]
                )
            ]
        )
        self.scatter = self.fig.data[0]

        colors = np.array([ELECTRODE_COLOR] * n_units)
        colors[units_data] = DETECTED_COLOR
        colors[units_data[0]] = SELECTED_COLOR
        self.scatter.marker.color = colors
        size = np.array([3] * n_units)
        size[units_data] = 10
        size[units_data[0]] = 15
        self.scatter.marker.size = size

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
            plot_bgcolor=BACKGROUND_COLOR,
            xaxis=dict(showline=False, showticklabels=False, ticks=""),
            yaxis=dict(showline=False, showticklabels=False, ticks="")
        )

        self.children = [self.fig]

    def update(self, electrodes, index: int = 0):
        n_electrodes = len(self.scatter.marker.size)
        units_idx = np.where(np.array(self.scatter.marker.size) >= 10)[0]
        units_data = np.array(self.scatter.marker.size)[units_idx]
        n_units = len(units_data)
        c = np.array([ELECTRODE_COLOR] * n_electrodes)
        c[units_idx] = DETECTED_COLOR
        c[units_idx[index]] = SELECTED_COLOR

        s = np.array([3] * n_electrodes)
        s[units_idx] = 10
        s[units_idx[index]] = 15

        with self.fig.batch_update():
            self.scatter.marker.color = c
            self.scatter.marker.size = s


class PSTHWithElectrodeSelector(widgets.HBox):

    def update_point(self, trace, points, selector):
        units_data = np.array(self.electrode_position_selector.scatter.marker.size) >= 10
        my_point = points.point_inds[0]
        if units_data[my_point]:
            n_units = len(units_data)
            c = np.array([ELECTRODE_COLOR] * n_units)
            c[units_data] = DETECTED_COLOR
            c[my_point] = SELECTED_COLOR

            s = np.array([3] * n_units)
            s[units_data] = 10
            s[my_point] = 15

            with self.electrode_position_selector.fig.batch_update():
                self.electrode_position_selector.scatter.marker.color = c
                self.electrode_position_selector.scatter.marker.size = s

        self.psth_widget.unit_controller.value = np.where(
            np.array(self.electrode_position_selector.scatter.marker.size)[units_data] == 15
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
