import numpy as np

from ipywidgets import widgets
from pynwb.misc import Units

import nwbwidgets
from nwbwidgets import default_neurodata_vis_spec
from nwbwidgets.ecephys import ElectrodePositionSelector
from nwbwidgets.misc import PSTHWidget
from nwbwidgets import base


class PSTHWithElectrodeSelector(widgets.HBox):

    def update_point(self, trace, points, selector):
        n_points = len(self.electrode_position_selector.scatter.marker.color)
        c = ["#a3a7e4"] * n_points
        s = [10] * n_points
        my_point = points.point_inds[0]
        c[my_point] = "#bae2be"
        s[my_point] = 15
        with self.electrode_position_selector.fig.batch_update():
            self.electrode_position_selector.scatter.marker.color = c
            self.electrode_position_selector.scatter.marker.size = s
        self.psth_widget.unit_controller.value = np.where(
            np.array(self.electrode_position_selector.scatter.marker.size) == 15
        )[0][0]
        
        
    # def __init__(self, electrodes):
    #     super().__init__()
    #     x = electrodes["rel_x"].data[:]
    #     y = electrodes["rel_y"].data[:]

    #     scatter_kwargs = dict(x=x, y=y, mode="markers")
    #     self.filter_by_units = widgets.Checkbox(
    #         value=True,
    #         description="hide non-spiking electrodes",
    #         disabled=False,
    #         indent=False
    #     )
    #     if electrodes.get_ancestor("NWBFile").units is None:
    #         self.filter_by_units.disabled = True

    #     if self.filter_by_units.value and not self.filter_by_units.disabled:
    #         units_data = electrodes.get_ancestor("NWBFile").units.id.data[:]
    #         n_units = len(units_data)
    #         scatter_kwargs.update(x=x[units_data], y=y[units_data], text=[f"Unit ID: {x}" for x in units_data])
    #     else:
    #         n_units = len(x)
    #         scatter_kwargs.update(x=x, y=y)

    #     self.fig = go.FigureWidget([go.Scatter(**scatter_kwargs)])
    #     self.scatter = self.fig.data[0]

    #     colors = np.array(["#a3a7e4"] * n_units)
    #     colors[0] = "#bae2be"
    #     self.scatter.marker.color = colors
    #     size = np.array([10] * n_units)
    #     size[0] = 15
    #     self.scatter.marker.size = size

    #     self.scatter.on_click(self.update_point)
    #     self.fig.layout.hovermode = "closest"

    #     self.children = [
    #         widgets.VBox(
    #             [
    #                 self.filter_by_units,
    #                 self.fig
    #             ]
    #         )
    #     ]

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
