"""
Hyperspectral tissue simulation visualization toolkit
===================================================================
The MIT License (MIT)

Copyright (c) 2018 Computer Assisted Medical Interventions Group, DKFZ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated simpa_documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

REQUIREMENTS:
===================================================================
You will need to install the following packages in order to use this script. It is recommended to create a virtualenv
to install this packages:

`pip install dash dash-table dash-daq dash_colorscales plotly plotly-express dash-bootstrap-components pandas numpy
dash-slicer`

USAGE:
===================================================================

"""
import warnings

from dash import Dash, html, Input, Output, State, dcc, callback_context, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_colorscales as dcs
import plotly_express as px
import plotly.graph_objects as go
import argparse
import base64
import numpy as np
import os
import pandas as pd
from skimage import draw
from scipy import ndimage
from typing import *

from simpa.io_handling import load_hdf5
from simpa.utils import get_data_field_from_simpa_output, SegmentationClasses, Tags

EXTERNAL_STYLESHEETS = [dbc.themes.JOURNAL,
                        # 'assets/dcc.css'
                        ]
app = Dash(__name__, external_stylesheets=EXTERNAL_STYLESHEETS, title="SIMPA")

SIMPA_LOGO = './assets/simpa_logo.png'
CAMI_LOGO = './assets/CAMIC_logo-wo_DKFZ.png'
GITHUB_LOGO = './assets/GitHub-Mark-64px.png'

ENCODED_GITHUB_LOGO = (base64.b64encode(open(GITHUB_LOGO, 'rb').read())).decode()
ENCODED_SIMPA_LOGO = (base64.b64encode(open(SIMPA_LOGO, 'rb').read())).decode()
ENCODED_CAMI_LOGO = (base64.b64encode(open(CAMI_LOGO, 'rb').read())).decode()

DEFAULT_COLORSCALE = ['rgb(5,48,97)', 'rgb(33,102,172)', 'rgb(67,147,195)', 'rgb(146,197,222)', 'rgb(209,229,240)',
                      'rgb(247,247,247)', 'rgb(253,219,199)', 'rgb(244,165,130)', 'rgb(214,96,77)',
                      'rgb(178,24,43)', 'rgb(103,0,31)']
GITHUB_LINK = 'https://github.com/CAMI-DKFZ/simpa'
BG_COLOR = "#506784"
FONT_COLOR = "#F3F6FA"
APP_TITLE = "SIMPA"
PREVENT_3D_UPDATES = ['volume_axis', 'colorscale_picker', 'volume_slider']


class DataContainer:
    shape: Union[None, List, Tuple] = None
    simpa_output: Union[None, Dict] = None
    simpa_data_fields: Union[None, Dict] = None
    wavelengths: Union[np.ndarray, None] = None
    segmentation_labels: Union[Dict, None] = None
    plot_types: List = ['imshow', 'hist-3D', 'hist-2D', 'box', 'violin', 'contour']
    click_points: Dict = {'x': [], 'y': [], 'z': []}


DATA = DataContainer()

# plotly figure controls configuration
GRAPH_CONFIG = {
    "modeBarButtonsToAdd": [
        "drawclosedpath",
        "drawrect",
        "eraseshape",
    ],
}

app.layout = html.Div([
    html.Div(id='dummy', style={'display': None}),
    html.Div([
        dbc.Toast(
            [html.P("Found N infinite values in array", id='toast_content')],
            id="alert_toast",
            header="Not finite values",
            is_open=False,
            dismissable=True,
            icon="danger",
            style={"position": "fixed", "top": 66, "right": 10, "width": 250},
            duration=4000,
        ),
    ], style=dict(zIndex=1, position="relative")),
    dbc.Row([
        dbc.Col([
            html.H4("SIMPA Visualization Tool"),
            html.H6("CAMI, Computer Assisted Medical Interventions"),
        ], width=8),
        dbc.Col([
            html.Img(src='data:image/png;base64,{}'.format(ENCODED_SIMPA_LOGO), width='100%')
        ], width=1),
        dbc.Col([
            html.Img(src='data:image/png;base64,{}'.format(ENCODED_CAMI_LOGO), width='100%')
        ], width=2),
        dbc.Col([
            html.A(
                href=GITHUB_LINK,
                children=[
                    html.Img(src='data:image/png;base64,{}'.format(ENCODED_GITHUB_LOGO), width='75%')
                ]
            )
        ], width=1),
    ], style=dict(zIndex=0)),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Tabs(children=[
                dcc.Tab(label="About", id="about-tab", children=[
                    html.Br(),
                    html.P("This is a dash app designed by the SIMPA developer team. For more information on "
                           "the SIMPA toolkit visit:", style={'text-align': 'justify'}),
                    html.A("SIMPA", href=GITHUB_LINK),
                    html.P("This app was developed based on the Dash framework from Plotly. It will allow you"
                           "to interactively visualize the simulated results that the SIMPA toolkit outputs. "
                           "If you encounter any problems please reach out to the developer team through the"
                           "GitHub page of SIMPA.",
                           style={'text-align': 'justify'}),
                    dcc.Markdown("In the **Handlers** tab you will find a set of controllers that will help "
                                 "you select the dataset you want to visualize and different visualization "
                                 "controllers to select the correct displaying format.",
                                 style={'text-align': 'justify'}),
                    dcc.Markdown("In the *Data selection* subcategory you can choose which axis of the data "
                                 "you want to visualize. In the *Visual settings* subcategory you can choose "
                                 "the global color scale used for each subplot",
                                 style={'text-align': 'justify'})
                ]),
                dcc.Tab(label="Handlers", id="handler-tab", children=[
                    html.Hr(),
                    html.H6("Plotting / Handling settings"),
                    dbc.Input(
                        id="data_path",
                        type="text",
                        pattern=None,
                        placeholder="Path to simulation folder or file",
                        persistence=True,
                        persistence_type="session",
                        invalid=True,
                        autofocus=True
                    ),
                    dcc.Dropdown(
                        multi=False,
                        id="file_selection",
                        placeholder="Simulation files",
                        persistence=True,
                        persistence_type="session",
                    ),
                    html.Hr(),
                    html.H6("Data selection"),
                    html.P("Axis"),
                    dbc.RadioItems(
                        id="volume_axis",
                        persistence_type="session",
                        options=[{'label': 'x', 'value': 0},
                                 {'label': 'y', 'value': 1},
                                 {'label': 'z', 'value': 2}],
                        value=2,
                        inline=True
                    ),
                    html.Hr(),
                    html.H6("Visual settings"),
                    html.P("Color scale"),
                    dcs.DashColorscales(
                        id="colorscale_picker",
                        nSwatches=7,
                        fixSwatches=True,
                        colorscale=DEFAULT_COLORSCALE
                    ),
                    html.Hr(),
                    html.H6("General Information"),
                    html.Div([

                    ], id="general_info")
                ])
            ]),

        ], width=2),
        dbc.Col([
            dcc.Tabs([
                dcc.Tab(label="Visualization", id="tab-1", children=[
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                dbc.Switch(
                                    id="annotate_switch",
                                    value=False,
                                    label="Annotate"
                                ),
                                html.Div(
                                    id="plot_div_1",
                                    children=[
                                        html.Div([
                                            dcc.RangeSlider(
                                                id="plot_scaler1",
                                                min=0,
                                                max=1,
                                                step=0.01,
                                                value=[0., 1.],
                                                marks={i: {'label': f'{i:1.1f}'} for i in np.arange(0, 1, 0.1)},
                                                allowCross=False,
                                                pushable=0.05,
                                                tooltip=dict(always_visible=True, placement="left"),
                                                persistence_type="session",
                                                disabled=False,
                                                vertical=True,
                                            ),
                                        ], style={"width": "5%", "display": 'inline-block'}
                                        ),

                                        dcc.Graph(id="plot_11",
                                                  hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                                  style={"width": "45%", "display": 'inline-block'},
                                                  config=GRAPH_CONFIG),
                                        dcc.Graph(id="plot_12",
                                                  hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                                  style={"width": "45%", "display": 'inline-block'},
                                                  ),
                                        html.Div([
                                            dcc.RangeSlider(
                                                id="plot_scaler2",
                                                min=0,
                                                max=1,
                                                step=0.01,
                                                value=[0., 1.],
                                                marks={i: {'label': f'{i:1.1f}'} for i in np.arange(0, 1, 0.1)},
                                                allowCross=False,
                                                pushable=0.05,
                                                tooltip=dict(always_visible=True, placement="left"),
                                                persistence_type="session",
                                                disabled=False,
                                                vertical=True,
                                            ),
                                        ], style={"width": "5%", "display": 'inline-block'}
                                        ),
                                        html.Div([
                                            dcc.Dropdown(
                                                id="plot_type1",
                                                multi=False,
                                                placeholder="Plot type",
                                                persistence_type="session",
                                                disabled=False,
                                                style=dict(width='150px', display='inline-block')
                                            ),
                                            dcc.Dropdown(
                                                id="param1",
                                                multi=False,
                                                placeholder="Parameter to plot",
                                                persistence_type="session",
                                                disabled=True,
                                                style={'width': '150px', 'display': 'inline-block',
                                                       'margin-left': '5px'}
                                            ),
                                            html.H6("Volume slice selector"),
                                            dcc.Slider(
                                                min=0,
                                                max=100,
                                                value=50,
                                                step=1,
                                                tooltip=dict(always_visible=True, placement="top"),
                                                updatemode="mouseup",
                                                persistence_type="session",
                                                id="volume_slider",
                                                marks={
                                                    0: {'label': '0', 'style': {'color': '#77b0b1'}},
                                                    26: {'label': '26'},
                                                    37: {'label': '37'},
                                                    100: {'label': '100', 'style': {'color': '#f50'}}
                                                }
                                            )
                                        ], style={"width": "50%", "display": 'inline-block'}),
                                        html.Div([
                                            dcc.Dropdown(
                                                id="plot_type2",
                                                multi=False,
                                                placeholder="Plot type",
                                                persistence_type="session",
                                                disabled=True,
                                                style=dict(width='150px', display='inline-block')
                                            ),
                                            dcc.Dropdown(
                                                id="param2",
                                                multi=False,
                                                placeholder="Parameter to plot",
                                                persistence_type="session",
                                                disabled=True,
                                                style={'width': '150px', 'display': 'inline-block',
                                                       'margin-left': '5px'}
                                            ),
                                            html.H6("Wavelength selector"),
                                            dcc.Slider(
                                                min=0,
                                                max=100,
                                                value=50,
                                                step=1,
                                                tooltip=dict(always_visible=True, placement="top"),
                                                updatemode="mouseup",
                                                persistence_type="session",
                                                id="channel_slider",
                                                marks={
                                                    0: {'label': '0', 'style': {'color': '#77b0b1'}},
                                                    26: {'label': '26'},
                                                    37: {'label': '37'},
                                                    100: {'label': '100', 'style': {'color': '#f50'}}
                                                }
                                            )
                                        ], style={"width": "50%", "display": 'inline-block'})
                                    ]
                                )
                            ])
                        ]),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Spectra visualization"),
                                html.Div(
                                    id="plot_div_2",
                                    children=[
                                        dcc.Dropdown(
                                            id="param3",
                                            multi=True,
                                            placeholder="Parameter to plot over wavelengths",
                                            persistence_type="session",
                                            disabled=True,
                                            style=dict(width='400px', display='inline-block')
                                        ),
                                        dbc.Button(
                                            "Reset graph",
                                            id="reset_points",
                                            style={'display': 'inline-block', 'verticalAlign': 'top',
                                                   'margin-left': '5px'},
                                            outline=True,
                                            color="primary",
                                        ),
                                        dbc.Input(
                                            id="n_bins",
                                            type="number",
                                            min=10,
                                            max=100,
                                            step=1,
                                            value=10,
                                            placeholder="N bins",
                                            style={'display': 'inline-block',
                                                   'verticalAlign': 'top',
                                                   'margin-left': '8px',
                                                   'width': '5%'
                                                   },
                                        ),
                                        dcc.Graph(id="plot_21",
                                                  hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                                  style={"width": "100%", "display": 'inline-block'})
                                    ]
                                )
                            ])
                        ], id="spectra-row")
                    ], width=12),
                ]),
            ])
        ])
    ])
], style={'padding': 40, 'zIndex': 0})


@app.callback(
    Output("file_selection", "options"),
    [Input("data_path", "n_submit")],
    [State("data_path", "value")]
)
def populate_file_selection(_, data_path: Union[None, str]) -> List[Dict]:
    """
    queries all files in directory and filters them according to ending: ``.hdf5``. Creates a list of dictionaries
    based on such files: ``[{'label': ..., 'value': ...}, ...]``. If data path is invalid then ``PreventUpdate`` is
    raised

    :param _: Dummy parameter used to trigger callback
    :param data_path: path to file or folder containing ``.hdf5`` simulations
    :return: List of dictionaries ``[{'label': ..., 'value': ...}, ...]``
    """
    if isinstance(data_path, str):
        if os.path.isdir(data_path):
            file_list = os.listdir(data_path)
            file_list = [f for f in file_list if f.endswith('.hdf5')]
            options_list = [{'label': i, 'value': os.path.join(data_path, i)} for i in file_list]
            return options_list
        elif os.path.isfile(data_path):
            file_list = [os.path.basename(data_path)]
            options_list = [{'label': i, 'value': data_path} for i in file_list]
            return options_list
        else:
            raise PreventUpdate("Please select a file or folder!")
    else:
        raise PreventUpdate()


@app.callback(Output('data_path', 'invalid'),
              Output('data_path', 'valid'),
              Input('data_path', 'value'),
              )
def update_data_path_validity(data_path: Union[str, None]) -> Tuple[bool, bool]:
    """
    updates appearance of input component in ``app.layout`` to represent the validity of path to data

    :param data_path: path to file or folder containing ``.hdf5`` files containing simulations generated with ``SIMPA``
    :return: tuple of boolean values to update ``invalid`` and ``valid`` properties of ``data_path`` component
    """
    if isinstance(data_path, str):
        valid = True if (os.path.isfile(data_path) and data_path.endswith('.hdf5')) or \
                        os.path.isdir(data_path) else False
    else:
        valid = False
    return 1 - valid, valid


@app.callback(Output("n_bins", "disabled"),
              Input("annotate_switch", "value"))
def deactivate_n_bins(annotate: bool) -> bool:
    """
    enables ``n_bins`` input component when annotation process is started

    :param annotate: indicates if annotations process is started
    :return: bool that activates ``n_bins`` input component
    """
    return 1 - annotate


@app.callback(
    Output("param1", "options"),
    Output("param1", "disabled"),
    Output("channel_slider", "min"),
    Output("channel_slider", "max"),
    Output("channel_slider", "value"),
    Output("channel_slider", "step"),
    Output("channel_slider", "marks"),
    Output("volume_slider", "min"),
    Output("volume_slider", "max"),
    Output("volume_slider", "value"),
    Output("volume_slider", "step"),
    Output("volume_slider", "marks"),
    Output("volume_slider", "disabled"),
    Output("param2", "options"),
    Output("param2", "disabled"),
    Output("param3", "options"),
    Output("param3", "disabled"),
    Output("plot_type1", "value"),
    Output("plot_type1", "options"),
    Output("plot_type2", "value"),
    Output("plot_type2", "options"),
    Output("plot_type1", "disabled"),
    Output("plot_type2", "disabled"),
    Input("file_selection", "value"),
    Input("volume_axis", "value")
)
def populate_file_params(file_path: Union[None, str],
                         axis: int) -> \
        Tuple[List[Dict[str, Any]], bool, int, int, Any, Union[int, Any], Dict, int, Union[int, Any], int, int, Dict,
              bool, List[Dict[str, Any]], bool, List[Dict[str, Any]], bool, str, List[Dict[str, Any]], str,
              List[Dict[str, Any]], bool, bool]:
    """
    loads data selected from ``file_selection``input component if this was the triggering component. The data is sliced
    along new axis when the triggering component is ``volume_axis``.

    :param file_path: path to ``.hdf5`` file containing simulations generated with SIMPA
    :param axis: axis along which a view of the data is desired
    :return:
    """
    global DATA
    ctx = callback_context
    if not ctx.triggered:
        button_id = None
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == "volume_axis" and DATA.shape is not None:
        if len(DATA.shape) == 3:
            n_slices = DATA.shape[axis]
            disable_vol_slider = False
            vol_slider_marks = {i: {'label': str(i)} for i in range(n_slices)[::int(n_slices / 10)]}
            vol_slider_min = 0
            vol_slider_max = n_slices - 1
            vol_slider_value = 0
        elif len(DATA.shape) == 2:
            disable_vol_slider = True
            vol_slider_marks = {i: {'label': str(i)} for i in range(0, 100, 10)}
            vol_slider_min = 0
            vol_slider_max = 100
            vol_slider_value = 0
        else:
            raise ValueError(f"Number of dimensions of mua is not supported: {DATA.shape}")
        return no_update, no_update, no_update, no_update, no_update, \
               no_update, no_update, \
               vol_slider_min, vol_slider_max, vol_slider_value, 1, vol_slider_marks, disable_vol_slider, \
               no_update, no_update, no_update, no_update, no_update, no_update, \
               no_update, no_update, no_update, no_update
    if file_path is None:
        raise PreventUpdate()
    if os.path.isfile(file_path):
        DATA.simpa_output = load_hdf5(file_path)
        _load_data_fields()
        mua = DATA.simpa_data_fields['mua'][DATA.wavelengths[0]]
        DATA.shape = mua.shape
        if len(mua.shape) == 3:
            n_slices = DATA.simpa_data_fields['mua'][DATA.wavelengths[0]].shape[-1]
            disable_vol_slider = False
            vol_slider_marks = {i: {'label': str(i)} for i in range(n_slices)[::int(n_slices / 10)]}
            vol_slider_min = 0
            vol_slider_max = n_slices - 1
            vol_slider_value = 0
        elif len(mua.shape) == 2:
            disable_vol_slider = True
            vol_slider_marks = {i: {'label': str(i)} for i in range(0, 100, 10)}
            vol_slider_min = 0
            vol_slider_max = 100
            vol_slider_value = 0
        else:
            raise ValueError(f"Number of dimensions of mua is not supported: {mua.shape}")
        plot_options = [{'label': t, 'value': t} for t in DATA.plot_types if "3d" not in t]
        options_list = [{'label': key, 'value': key} for key in DATA.simpa_data_fields.keys() if key != 'units']
        if len(DATA.wavelengths) > 20:
            marks = {int(wv): str(wv) for wv in DATA.wavelengths[::int(len(DATA.wavelengths) / 10)]}
        else:
            marks = {int(wv): str(wv) for wv in DATA.wavelengths}
        if len(DATA.wavelengths) == 1:
            wv_step = 1
        else:
            wv_step = DATA.wavelengths[1] - DATA.wavelengths[0]
        return options_list, False, min(DATA.wavelengths), max(DATA.wavelengths), DATA.wavelengths[0], \
               wv_step, marks, \
               vol_slider_min, vol_slider_max, vol_slider_value, 1, vol_slider_marks, disable_vol_slider, \
               options_list, False, options_list, False, "imshow", plot_options, "imshow", plot_options, \
               False, False


def _get_data_properties(data_dict: Dict) -> List:
    """
    extracts all keywords from ``data_dict`` that match properties from ``Tags`` which start with ``DATA_FIELD``. This
    function iterates also in nested dictionaries of ``data_dict``

    :param data_dict: dictionary containing SIMPA output
    :return: list of keywords in ``data_dict`` also in nested dictionaries
    """
    expected_fields = [getattr(Tags, t) for t in Tags().__dir__() if t.startswith('DATA_FIELD')]
    data_fields = []
    for k, item in data_dict.items():
        if k in expected_fields:
            data_fields.append(k)
        if isinstance(item, Dict):
            data_fields += _get_data_properties(data_dict=item)
    return data_fields


def _load_data_fields() -> None:
    """
    extracts all data fields from ``DATA.simpa_output`` and stores them in ``DATA.simpa_data_fields. It iterates over
    all tags in ``Tags`` and stores the ones that start with ``DATA_FIELD`` and exist in ``simpa_output``

    :return: None
    """
    DATA.wavelengths = DATA.simpa_output["settings"]["wavelengths"]
    segment_class = SegmentationClasses()
    DATA.segmentation_labels = [k for k in segment_class.__dir__() if '__' not in k]
    DATA.segmentation_labels = {getattr(segment_class, k): k for k in DATA.segmentation_labels}
    data_fields = dict()
    sim_props = _get_data_properties(data_dict=DATA.simpa_output)

    for data_field in sim_props:
        data_fields[data_field] = dict()
        for wavelength in DATA.wavelengths:
            try:
                data_wv = get_data_field_from_simpa_output(DATA.simpa_output, data_field, wavelength)
            except ValueError as err:
                warnings.warn(f"failed to query {data_field}, got error: \n{err}")
                continue
            if isinstance(data_wv, np.ndarray):
                data_wv = np.rot90(data_wv, 1)
                data_fields[data_field][wavelength] = data_wv
                data_fields[data_field]["shape"] = len(data_wv.shape)
        if not data_fields[data_field]:
            del data_fields[data_field]

    DATA.simpa_data_fields = data_fields


@app.callback(
    Output("plot_11", "figure"),
    Output("plot_scaler1", "disabled"),
    Input("param1", "value"),
    Input("colorscale_picker", "colorscale"),
    Input("channel_slider", "value"),
    Input("plot_scaler1", "value"),
    Input("plot_type1", "value"),
    Input("volume_axis", "value"),
    Input("volume_slider", "value"),
    Input("plot_11", "clickData"),
    Input("plot_12", "clickData")
)
def update_plot_11(data_field, colorscale, wavelength, z_range, plot_type, axis, axis_ind, _, __):
    return plot_data_field(data_field, colorscale, wavelength, z_range, plot_type, axis, axis_ind)


@app.callback(
    Output("plot_12", "figure"),
    Output("plot_scaler2", "disabled"),
    Input("param2", "value"),
    Input("colorscale_picker", "colorscale"),
    Input("channel_slider", "value"),
    Input("plot_scaler2", "value"),
    Input("plot_type2", "value"),
    Input("volume_axis", "value"),
    Input("volume_slider", "value"),
    Input("plot_11", "clickData"),
    Input("plot_12", "clickData")
)
def update_plot_12(data_field, colorscale, wavelength, z_range, plot_type, axis, axis_ind, _, __):
    return plot_data_field(data_field, colorscale, wavelength, z_range, plot_type, axis, axis_ind)


def get_current_frame(data_field: Union[List, str],
                      wavelength: Union[int, float],
                      axis_ind: int,
                      axis: int) -> Union[np.ndarray, Dict]:
    """
    Extracts view of volume data. If ``data_field`` is a list, a dictionary is created where each item is an
    ``np.ndarray`` and each key the name of the parameter. All data is extracted from ``DATA``

    :param data_field: name of parameter from which the volume is extracted
    :param wavelength: wavelength that is desired to be extracted
    :param axis_ind: index along ``axis`` that is desired
    :param axis: axis that is desired to be extracted
    :return: array of dictionary of arrays with extracted data
    """
    plot_data = None
    if isinstance(data_field, str):
        if len(DATA.simpa_data_fields[data_field][wavelength].shape) == 3:
            plot_data = np.take(DATA.simpa_data_fields[data_field][wavelength], indices=axis_ind, axis=axis)
        elif len(DATA.simpa_data_fields[data_field][wavelength].shape) == 2:
            plot_data = DATA.simpa_data_fields[data_field][wavelength]
    elif isinstance(data_field, list):
        plot_data = dict()
        for k in data_field:
            if len(DATA.simpa_data_fields[k][wavelength].shape) == 3:
                plot_data[k] = np.take(DATA.simpa_data_fields[k][wavelength], indices=axis_ind, axis=axis)
            elif len(DATA.simpa_data_fields[k][wavelength].shape) == 2:
                plot_data[k] = DATA.simpa_data_fields[k][wavelength]
    return plot_data


def to_pandas(data_dict: Dict) -> pd.DataFrame:
    """
    transforms dictionary to pandas dataframe and metls it according to variable ``Parameter`` and value name ``Value``

    :param data_dict: dictionary containing data to be transformed
    :return: pandas DataFrame
    """
    df = pd.DataFrame(data_dict)
    df = df.melt(var_name="Parameter", value_name="Value")
    return df


def get_structure_names(array: np.ndarray) -> np.ndarray:
    """
    transforms int values in ``array`` to their corresponding name of structure based on ``DATA.segmentation_labels``

    :param array: array of `ìnt values``
    :return: array of strings representing the names of the structures
    """
    names = array.copy().astype('str')
    unique_values = np.unique(names)
    for v in unique_values:
        names[names == v] = DATA.segmentation_labels.get(float(v))
    return names


def plot_data_field(data_field: Union[None, str],
                    colorscale: str,
                    wavelength: Union[int, float],
                    z_range: Union[List, Tuple],
                    plot_type: Union[None, str],
                    axis: int,
                    axis_ind: int) -> Tuple[go.Figure, bool]:
    """
    generates plot to update ``plot_11`` and ``plot_12`` in ``app.layout``. In addition, it determines if the scaler
    sliders are activated or deactivated. All data used for plotting is extracted from ``DATA``

    :param data_field: parameter from simpa output to be plotted
    :param colorscale: colorscale used for plot
    :param wavelength: wavelength to be extracted
    :param z_range: range used to scale colorscale of plot
    :param plot_type: type of plot to be generated
    :param axis: axis along which data is extracted
    :param axis_ind: index along ``axis`` from which data is extracted
    :return: go.Figure and bool indicating if sliders for scaling are deactivated or activated
    """
    if data_field is None or wavelength is None:
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        component_id = None
    else:
        component_id = ctx.triggered[0]['prop_id'].split('.')[0]

    plot_data = get_current_frame(data_field=data_field, wavelength=wavelength, axis_ind=axis_ind, axis=axis)
    if plot_data is None:
        raise PreventUpdate
    z_min = np.nanmin(plot_data) + (np.nanmax(plot_data) - np.nanmin(plot_data)) * z_range[0]
    z_max = np.nanmax(plot_data) * z_range[1]
    if plot_type == "imshow":
        kwargs = dict()
        if data_field == "seg":
            custom_data = get_structure_names(plot_data)
            kwargs["hovertext"] = custom_data
        plot = [go.Heatmap(z=plot_data, colorscale=colorscale, zmin=z_min, zmax=z_max, **kwargs)]
        figure = go.Figure(data=plot)
        for i, x in enumerate(DATA.click_points["x"]):
            if x:
                figure.add_annotation(x=x,
                                      y=DATA.click_points["y"][i],
                                      text=str(i),
                                      showarrow=True,
                                      arrowhead=6,
                                      bgcolor="#ffffff",
                                      opacity=0.8
                                      )
        disable_scaler = False
    elif plot_type == "hist-2D":
        df = pd.DataFrame()
        df[data_field] = list(plot_data.flatten())
        figure = px.histogram(data_frame=df, y=data_field, marginal="box")
        disable_scaler = True
    elif plot_type == "box":
        df = pd.DataFrame()
        df[data_field] = list(plot_data.flatten())
        figure = px.box(data_frame=df, y=data_field, notched=True)
        disable_scaler = True
    elif plot_type == 'violin':
        df = pd.DataFrame()
        df[data_field] = list(plot_data.flatten())
        figure = px.violin(data_frame=df, y=data_field, box=True)
        disable_scaler = True
    elif plot_type == "contour":
        figure = go.Figure(data=go.Contour(z=plot_data, contours=dict(showlabels=True,
                                                                      labelfont=dict(size=14, color="white")),
                                           colorscale=colorscale))
        for i, x in enumerate(DATA.click_points["x"]):
            figure.add_annotation(x=x, y=DATA.click_points["y"], text=str(i), showarrow=True, arrowhead=6)
        disable_scaler = True
    elif plot_type == "hist-3D":
        if component_id in PREVENT_3D_UPDATES:
            raise PreventUpdate
        plot_data = DATA.simpa_data_fields[data_field][wavelength].flatten()
        if plot_data.size > 10000:
            plot_data = np.random.choice(plot_data, 10000)
        df = pd.DataFrame()
        df[data_field] = list(plot_data)
        figure = px.histogram(data_frame=df, y=data_field, marginal="box")
        disable_scaler = True
    else:
        raise PreventUpdate
    return figure, disable_scaler


@app.callback(Output("dummy", "children"),
              Output("plot_11", "clickData"),
              Output("plot_12", "clickData"),
              Input("reset_points", "n_clicks"))
def rest_points(_):
    """
    Resets clicked data in ``DATA.click_points``, ``plot_11`` and ``plot_12``

    :param _: dummy param to trigger callback
    :return: Nulls to update clicked data
    """
    global DATA
    DATA.click_points = {'x': [], 'y': [], 'z': []}
    return [], {}, {}


def get_data_from_last_shape(relayout_data: Dict,
                             plot_data: Dict) -> Union[None, Dict]:
    """
    It extracts the data from ``plot_data`` given ``relayout_data`` incoming from updates in a plotly Figure. It extracts
    the data based on the last shape in ``relayout_data``. Only supports shapes of type ``path`` and ``rect``.

    :param relayout_data: dictionary containing all relayout data from a plotly Figure
    :param plot_data: dictionary containing all the data to be plotted
    :return: None or dictionary containing the extracted data as flattened numpy arrays.
    """
    shapes = relayout_data.get('shapes')
    results = dict()
    for k in plot_data:
        path = None
        array = plot_data[k]
        if shapes:
            last_shape = relayout_data["shapes"][-1]
            shape_type = last_shape.get("type")
            if shape_type == "path":
                path = last_shape.get("path")
            elif shape_type == "rect":
                x0, y0 = int(last_shape["x0"]), int(last_shape["y0"])
                x1, y1 = int(last_shape["x1"]), int(last_shape["y1"])
                x0, x1 = min([x0, x1]), max(x0, x1)
                y0, y1 = min([y0, y1]), max(y0, y1)
                roi_array = array[y0:y1, x0:x1]
                return roi_array.ravel()
        elif any(["shapes" in k and "path" in k for k in relayout_data]):
            path = [relayout_data[k] for k in relayout_data if "shapes" in k][0]
        elif any(["shapes" in k and "x0" in k for k in relayout_data]):
            keys = {"x0": [k for k in relayout_data if "x0" in k][0],
                    "x1": [k for k in relayout_data if "x1" in k][0],
                    "y0": [k for k in relayout_data if "y0" in k][0],
                    "y1": [k for k in relayout_data if "y1" in k][0]}
            x0, y0 = int(relayout_data[keys["x0"]]), int(relayout_data[keys["y0"]])
            x1, y1 = int(relayout_data[keys["x1"]]), int(relayout_data[keys["y1"]])
            x0, x1 = min([x0, x1]), max(x0, x1)
            y0, y1 = min([y0, y1]), max(y0, y1)
            roi_array = array[y0:y1, x0:x1]
            if not roi_array.size:
                return None
            else:
                return roi_array.ravel()
        if path:
            mask = path_to_mask(path, array.shape)
            results[k] = array[mask]
    return results if results else None


@app.callback(Output("param3", "value"),
              Input("param1", "value"),
              State("param3", "value"),
              prevent_initial_update=True
              )
def sync_data_fields(param1: Union[None, str, List], param3: Union[None, str, List]) -> List:
    """
    synchronizes data fields from ``param1`` and ``param3`` by appending ``param1``to all existing params in ``param3``

    :param param1: parameter to be appended to ``param3``
    :param param3: target parameter list to be extended
    :return:
    """
    if param1 is None:
        param1 = []
    if param3 is None:
        param3 = []
    if isinstance(param1, str):
        param1 = [param1]
    if isinstance(param3, str):
        param3 = [param3]
    return param1 + param3


@app.callback(
    Output("plot_21", "figure"),
    Input("param3", "value"),
    Input("plot_11", "clickData"),
    Input("plot_12", "clickData"),
    Input("volume_slider", "value"),
    Input("volume_axis", "value"),
    Input("reset_points", "n_clicks"),
    Input("annotate_switch", "value"),
    Input("plot_11", "relayoutData"),
    Input("channel_slider", "value"),
    Input("n_bins", "value")
)
def plot_spectrum(data_field: Union[None, str],
                  click_data1: Dict,
                  click_data2: Dict,
                  axis_ind: int,
                  axis: int,
                  n_clicks: Union[None, int],
                  annotate: bool,
                  relayout_data: Union[None, Dict],
                  wavelength: Union[float, int],
                  n_bins: int) -> go.Figure:
    """
    generates figure to update spectral visualization or histogram visualization of annotated regions extracted from
    ``plot_11`` or ``plot_12``. Always last annotated shape is used. The type of plot is defined by ``annotate``.
    It is either a line plot or a histogram with rugged plot.

    :param data_field: parameter form simpa output to be plotted
    :param click_data1: clicked data from ``plot_11``
    :param click_data2: clicked data from ``plot_12``
    :param axis_ind: index along axis from which to extract data
    :param axis: axis from which to extract data
    :param n_clicks: dummy parameter used to trigger plot reset functionality
    :param annotate: indicates if the data extracted from last annotated shape should be plotted as histogram
    :param relayout_data: relayout data extracted from plotly Figure
    :param wavelength: wavelength to be extracted
    :param n_bins: number of binds used to plot histogram if ``annotate==True``
    :return: ``go.Figure``
    """
    global DATA
    layout = {}
    ctx = callback_context
    if not ctx.triggered:
        component_id = None
    else:
        component_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if component_id == 'reset_points':
        return go.Figure(data={})
    if annotate:
        if data_field is None:
            raise PreventUpdate
        if "shapes" in relayout_data or any(["shapes" in k for k in relayout_data]):
            plot_data = get_current_frame(data_field=data_field, wavelength=wavelength, axis_ind=axis_ind,
                                          axis=axis)
            if plot_data is None:
                raise PreventUpdate
            hist_data = get_data_from_last_shape(relayout_data, plot_data)
            if hist_data is None:
                raise PreventUpdate
            hist_df = to_pandas(hist_data)
            return px.histogram(data_frame=hist_df,
                                color="Parameter",
                                opacity=0.7,
                                marginal="rug",
                                nbins=n_bins)

    if data_field is None or (click_data1 is None and click_data2 is None):
        raise PreventUpdate
    else:
        if isinstance(data_field, str):
            data_field = [data_field]
        for click_data in [click_data1, click_data2]:
            if not click_data:
                continue
            x_c = click_data["points"][0]["x"]
            y_c = click_data["points"][0]["y"]
            if not isinstance(x_c, int) or not isinstance(y_c, int):
                continue
            if point_already_in_data(x_c, y_c, DATA.click_points):
                continue
            DATA.click_points["x"].append(x_c)
            DATA.click_points["y"].append(y_c)
        if not DATA.click_points["x"] or not DATA.click_points["y"]:
            raise PreventUpdate
        plot_data = list()
        for i, x in enumerate(DATA.click_points["x"]):
            y = DATA.click_points["y"][i]
            for param in data_field:
                spectral_values = list()
                for wavelength in DATA.wavelengths:
                    if DATA.simpa_data_fields[param]["shape"] == 3:
                        array = np.take(DATA.simpa_data_fields[param][wavelength], indices=axis_ind,
                                        axis=axis)
                    else:
                        array = DATA.simpa_data_fields[param][wavelength]
                    spectral_values.append(array[y, x])
                layout = go.Layout(xaxis={'autorange': True})
                plot_data += [go.Scatter(x=DATA.wavelengths,
                                         y=spectral_values,
                                         mode="lines+markers",
                                         name=param + f" x={x}, y={y}",
                                         )
                              ]
        return go.Figure(data=plot_data, layout=layout)


def point_already_in_data(x: int,
                          y: int,
                          points: dict) -> bool:
    """
    checks if a point ``(x,y)`` already exists in a dictionary of ``points``

    :param x: x value
    :param y: y value
    :param points: dictionary of points stored as ``{'x': [...], 'y': [...]}``
    :return:
    """
    tuples = [(a, b) for a, b in zip(points["x"], points["y"])]
    return (x, y) in tuples


@app.callback(
    Output("general_info", "children"),
    Output("toast_content", "children"),
    Output("alert_toast", "is_open"),
    Input("param1", "value"),
    State("channel_slider", "value"),
)
def update_general_info(param1: Union[None, str],
                        wv: Union[int, float]) -> Tuple[List, List, bool]:
    """
    updates general info markdown in ``app.layout`` given `param_1` and a wavelength. Checks if there are invalid values
    in the volume, such as ``np.NaN``, not finite values. It also outputs data shape along each dimension.

    :param param1: parameter to be analyzed
    :param wv: wavelength to be extracted
    :return: list of toast children and if the toast is open or not
    """
    if not param1:
        raise PreventUpdate
    global DATA
    array = DATA.simpa_data_fields[param1][wv]
    shape = array.shape
    size = array.size * len(DATA.wavelengths)
    n_finite = [np.isfinite(DATA.simpa_data_fields[param1][w]).sum() for w in DATA.wavelengths]
    n_finite = np.sum(n_finite)
    n_not_finite = size - n_finite
    children = [
        dcc.Markdown(f'''
        Data shape per channel is `{shape}`\n
        Data size per parameter is `{size}`\n
        Not finite points: `{n_not_finite} -> {100 * n_not_finite / size}%`\n
        ''')
    ]
    if n_not_finite:
        open_toast = True
        toast_child = [
            dcc.Markdown(f"Found not finite values in array: `{n_not_finite} -> {100 * n_not_finite / size}%`")]
    else:
        open_toast = False
        toast_child = []
    return children, toast_child, open_toast


def path_to_indices(path: str) -> np.ndarray:
    """
    transforms SVG path to numpy array of coordinates, each row being a (row, col) point

    :param path: string representing the SVG path to be transformed
    :return: numpy array with coordinates
    """
    indices_str = [
        el.replace("M", "").replace("Z", "").split(",") for el in path.split("L")
    ]
    return np.rint(np.array(indices_str, dtype=float)).astype(int)


def path_to_mask(path: str, shape: Tuple) -> np.ndarray:
    """
    transforms SVG path to a boolean array where all pixels enclosed by the path
    are True, and the other pixels are False.

    :param path: string representing the SVG path to be transformed
    :param shape: shape of mask to be generated
    :return:
    """
    cols, rows = path_to_indices(path).T
    rr, cc = draw.polygon(rows, cols)
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    mask = ndimage.binary_fill_holes(mask)
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Run app in debug mode. Changes in code will update the app automatically.")
    parser.add_argument("-p", "--port", type=int,
                        help="Specify the port where the app should run (localhost:port), default is 8050")
    parser.add_argument("-host", "--host_name", type=str,
                        help="Name of the host on which the app is run, by default it is localhost")
    arguments = parser.parse_args()
    port = 8050
    host_name = "localhost"
    if arguments.port:
        port = arguments.port
    if arguments.host_name:
        host_name = arguments.host_name
    app.run_server(debug=arguments.debug, host=host_name, port=port)
