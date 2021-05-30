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

`pip install dash dash-table dash-daq dash_colorscales plotly plotly-express dash-bootstrap-components pandas numpy`

USAGE:
===================================================================

"""

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_table
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_colorscales as dcs
import plotly_express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import argparse
import base64
import numpy as np
import os
import pandas as pd

from simpa.io_handling import load_hdf5
from simpa.utils.tags import Tags

external_stylesheets = [dbc.themes.JOURNAL, '.assets/dcc.css']
app = dash.Dash(external_stylesheets=external_stylesheets, title="SIMPA")

simpa_logo = './assets/simpa_logo.png'
cami_logo = './assets/CAMIC_logo-wo_DKFZ.png'
encoded_simpa_logo = (base64.b64encode(open(simpa_logo, 'rb').read())).decode()
encoded_cami_logo = (base64.b64encode(open(cami_logo, 'rb').read())).decode()

DEFAULT_COLORSCALE = ['rgb(5,48,97)', 'rgb(33,102,172)', 'rgb(67,147,195)', 'rgb(146,197,222)', 'rgb(209,229,240)',
                      'rgb(247,247,247)', 'rgb(253,219,199)', 'rgb(244,165,130)', 'rgb(214,96,77)',
                      'rgb(178,24,43)', 'rgb(103,0,31)']


class DataContainer:
    simpa_output = None
    simpa_data_fields = None
    wavelengths = None
    plot_types = ['contour-3D', 'imshow', 'hist-3D', 'hist-2D', 'box', 'violin', 'contour']
    click_points = {'x': [], 'y': [], 'z': []}


data = DataContainer()

# plotly figure controls configuration
config = {
    "modeBarButtonsToAdd": [
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ]
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
        ], width=9),
        dbc.Col([
            html.Img(src='data:image/png;base64,{}'.format(encoded_simpa_logo), width='100%')
        ], width=1),
        dbc.Col([
            html.Img(src='data:image/png;base64,{}'.format(encoded_cami_logo), width='100%')
        ], width=2)
    ], style=dict(zIndex=0)),
    html.Br(),
    dcc.Tabs([
        dcc.Tab(label="Visualization", id="tab-1", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.H6("Plotting / Handling settings"),
                    dbc.Input(
                        id="data_path",
                        type="text",
                        pattern=None,
                        placeholder="Path to simulation folder or file",
                        persistence_type="session",
                    ),
                    dcc.Dropdown(
                        id="file_selection",
                        placeholder="Simulation files",
                        persistence_type="session",
                    ),
                    html.Br(),
                    html.Hr(),
                    html.H6("Data selection"),
                    dcc.Dropdown(
                        id="volume_axis",
                        multi=False,
                        placeholder="Volume axis",
                        persistence_type="session",
                        options=[{'label': 'x', 'value': 0},
                                 {'label': 'y', 'value': 1},
                                 {'label': 'z', 'value': 2}],
                        value=2
                    ),
                    html.Br(),
                    html.Hr(),
                    html.H6("Visual settings"),
                    html.P("Color scale"),
                    dcs.DashColorscales(
                        id="colorscale_picker",
                        nSwatches=7,
                        fixSwatches=True,
                        colorscale=DEFAULT_COLORSCALE
                    ),
                    html.Br(),
                    html.Hr(),
                    html.H6("General Information"),
                    html.Div([

                    ], id="general_info")
                ], width=2),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Parameter visualization"),
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
                                              config=config),
                                    dcc.Graph(id="plot_12",
                                              hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                              style={"width": "45%", "display": 'inline-block'},
                                              config=config),
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
                                            style={'width': '150px', 'display': 'inline-block', 'margin-left': '5px'}
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
                                            style={'width': '150px', 'display': 'inline-block', 'margin-left': '5px'}
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
                                        "Reset points",
                                        id="reset_points",
                                        style={'display': 'inline-block', 'verticalAlign': 'top', 'margin-left': '5px'},
                                        outline=True,
                                        color="primary",
                                    ),
                                    dcc.Graph(id="plot_21",
                                              hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                              style={"width": "100%", "display": 'inline-block'})
                                ]
                            )
                        ])
                    ])
                ], width=10),
            ])
        ]),
        dcc.Tab(label="Properties", children=[
            html.Br(),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H6("Data selection"),
                        html.Hr(),
                        html.Br(),
                        dash_table.DataTable(
                            id="data_table",
                            columns=[{"name": 'Col 1', "id": 'Col 1'}, {"name": 'Col 2', "id": 'Col 2'}],
                            data=[{'Col 1': 1, 'Col 2': 0.5}, {'Col 1': 2, 'Col 2': 1.5}],
                            style_as_list_view=True,
                            style_table={'overflowX': 'scroll', 'maxHeight': '300px',
                                         'overflowY': 'scroll', 'maxWidth': '800px'},
                            fixed_rows={'headers': True, 'data': 0},
                            style_cell={
                                'minWidth': '50px', 'maxWidth': '90px',
                                'overflow': 'hidden',
                                'textOverflow': 'ellipsis',
                                'padding': '5px',
                                'textAlign': 'center'
                            },
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ],
                            style_header={
                                'backgroundColor': '#e6b600ff',
                                'fontWeight': 'bold'
                            },
                            filter_action='native',
                            sort_action='native',
                            sort_mode='multi',
                            row_selectable="multi",
                            row_deletable=True,
                            persistence=False,
                        )
                    ]), width=6
                )
            ])
        ])
    ])
], style={'padding': 40, 'zIndex': 0})


@app.callback(
    Output("file_selection", "options"),
    [Input("data_path", "n_submit")],
    [State("data_path", "value")]
)
def populate_file_selection(_, data_path):
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
    Output("param2", "options"),
    Output("param2", "disabled"),
    Output("param3", "options"),
    Output("param3", "disabled"),
    Output("volume_slider", "disabled"),
    Output("plot_type1", "value"),
    Output("plot_type1", "options"),
    Output("plot_type2", "value"),
    Output("plot_type2", "options"),
    Output("plot_type1", "disabled"),
    Output("plot_type2", "disabled"),
    Input("file_selection", "value"),
)
def populate_file_params(file_path):
    if file_path is None:
        raise PreventUpdate()
    if os.path.isfile(file_path):
        data.simpa_output = load_hdf5(file_path)
        get_data_fields()
        if data.simpa_output["settings"][Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW[0]]:
            is_2d = True
            vol_slider_marks = {i: {'label': str(i)} for i in range(10)}
            vol_slider_min = 0
            vol_slider_max = 9
            vol_slider_value = 0
            plot_options = [{'label': t, 'value': t} for t in data.plot_types if "3D" not in t]
        else:
            is_2d = False
            n_slices = data.simpa_data_fields['mua'][data.wavelengths[0]].shape[-1]
            vol_slider_marks = {i: {'label': str(i)} for i in range(n_slices)[::int(n_slices / 10)]}
            vol_slider_min = 0
            vol_slider_max = n_slices - 1
            vol_slider_value = 0
            plot_options = [{'label': t, 'value': t} for t in data.plot_types if "3d" not in t]
        options_list = [{'label': key, 'value': key} for key in data.simpa_data_fields.keys() if key != 'units']
        marks = {int(wv): str(wv) for wv in data.wavelengths[::int(len(data.wavelengths) / 10)]}
        return options_list, False, min(data.wavelengths), max(data.wavelengths), data.wavelengths[0], \
                data.wavelengths[1] - data.wavelengths[0], marks, \
                vol_slider_min, vol_slider_max, vol_slider_value, 1, vol_slider_marks, \
                options_list, False, options_list, False, is_2d, "imshow", plot_options, "imshow", plot_options, \
                False, False


def get_data_fields():
    data.wavelengths = data.simpa_output["settings"]["wavelengths"]
    data_fields = dict()
    sim_props = list(data.simpa_output["simulations"]["simulation_properties"].keys())
    simulations = list(data.simpa_output["simulations"]["optical_forward_model_output"].keys())

    for data_field in sim_props:
        data_fields[data_field] = dict()
        for wavelength in data.wavelengths:
            data_fields[data_field][wavelength] = data.simpa_output["simulations"]\
                ["simulation_properties"][data_field][f"{wavelength}"]

    for data_field in simulations:
        data_fields[data_field] = dict()
        for wavelength in data.wavelengths:
            data_fields[data_field][wavelength] = data.simpa_output["simulations"]\
                ["optical_forward_model_output"][data_field][f"{wavelength}"]

    data.simpa_data_fields = data_fields


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
)
def plot_data_field(data_field, colorscale, wavelength, z_range, plot_type, axis, axis_ind):
    if data_field is None or wavelength is None:
        raise PreventUpdate
    else:
        if len(data.simpa_data_fields[data_field][wavelength].shape) == 3:
            plot_data = np.take(np.rot90(data.simpa_data_fields[data_field][wavelength], 1),
                                indices=axis_ind, axis=axis)
        elif len(data.simpa_data_fields[data_field][wavelength].shape) == 2:
            plot_data = np.rot90(data.simpa_data_fields[data_field][wavelength], 1)
        else:
            raise PreventUpdate
        z_min = np.nanmin(plot_data) * z_range[0]
        z_max = np.nanmax(plot_data) * z_range[1]
        if plot_type == "imshow":
            plot = [go.Heatmap(z=plot_data, colorscale=colorscale, zmin=z_min, zmax=z_max)]
            figure = go.Figure(data=plot)
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
                                                                          labelfont=dict(size=14, color="white"))))
            disable_scaler = True
        elif plot_type == "hist-3D":
            plot_data = np.rot90(data.simpa_data_fields[data_field][wavelength], 1).flatten()
            if plot_data.size > 10000:
                plot_data = np.random.choice(plot_data, 10000)
            df = pd.DataFrame()
            df[data_field] = list(plot_data)
            figure = px.histogram(data_frame=df, y=data_field, marginal="box")
            disable_scaler = True
        elif plot_type == "contour-3D":
            plot_data = np.rot90(data.simpa_data_fields[data_field][wavelength], 1)
            x, y, z = np.where(plot_data)
            v = plot_data[(x, y, z)]
            figure = go.Figure(go.Volume(x=x, y=y, z=z, value=v, opacity=0.1, isomin=z_min,
                                         isomax=z_max,
                                         caps=dict(x_show=False, y_show=False, z_show=False),
                                         surface=dict(fill=0.5, pattern='odd', count=10)))
            for i, x in enumerate(data.click_points["x"]):
                figure.add_annotation(x, data.click_points["y"], text=str(i), showarrow=True, arrowhead=6)
            disable_scaler = True
        else:
            raise PreventUpdate
        return figure, disable_scaler


@app.callback(
    Output("plot_12", "figure"),
    Output("plot_scaler2", "disabled"),
    Input("param2", "value"),
    Input("colorscale_picker", "colorscale"),
    Input("channel_slider", "value"),
    Input("plot_scaler2", "value"),
    Input("plot_type2", "value"),
    Input("volume_axis", "value"),
    Input("volume_slider", "value")
)
def plot_data_field(data_field, colorscale, wavelength, z_range, plot_type, axis, axis_ind):
    if data_field is None or wavelength is None:
        raise PreventUpdate
    else:
        if len(data.simpa_data_fields[data_field][wavelength].shape) == 3:
            plot_data = np.take(np.rot90(data.simpa_data_fields[data_field][wavelength], 1),
                                indices=axis_ind, axis=axis)
        elif len(data.simpa_data_fields[data_field][wavelength].shape) == 2:
            plot_data = np.rot90(data.simpa_data_fields[data_field][wavelength], 1)
        else:
            raise PreventUpdate
        z_min = np.nanmin(plot_data) * z_range[0]
        z_max = np.nanmax(plot_data) * z_range[1]
        if plot_type == "imshow":
            plot = [go.Heatmap(z=plot_data, colorscale=colorscale, zmin=z_min, zmax=z_max)]
            figure = go.Figure(data=plot)
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
                                                                          labelfont=dict(size=14, color="white"))))
            disable_scaler = True
        elif plot_type == "hist-3D":
            plot_data = np.rot90(data.simpa_data_fields[data_field][wavelength], 1).flatten()
            if plot_data.size > 10000:
                plot_data = np.random.choice(plot_data, 10000)
            df = pd.DataFrame()
            df[data_field] = list(plot_data)
            figure = px.histogram(data_frame=df, y=data_field, marginal="box")
            disable_scaler = True
        elif plot_type == "contour-3D":
            plot_data = np.rot90(data.simpa_data_fields[data_field][wavelength], 1)
            x, y, z = np.where(plot_data)
            v = plot_data[(x, y, z)]
            figure = go.Figure(go.Volume(x=x, y=y, z=z, value=v, opacity=0.1, isomin=z_min,
                                         isomax=z_max,
                                         caps=dict(x_show=False, y_show=False, z_show=False),
                                         surface=dict(fill=0.5, pattern='odd', count=10)))
            disable_scaler = True
        else:
            raise PreventUpdate
        return figure, disable_scaler


@app.callback(Output("dummy", "children"),
              Output("plot_11", "clickData"),
              Output("plot_12", "clickData"),
              Input("reset_points", "n_clicks"))
def rest_points(_):
    global data
    data.click_points = {'x': [], 'y': [], 'z': []}
    return [], {}, {}


@app.callback(
    Output("plot_21", "figure"),
    Input("param3", "value"),
    Input("plot_11", "clickData"),
    Input("plot_12", "clickData"),
    Input("volume_slider", "value"),
    Input("volume_axis", "value"),
)
def plot_spectrum(data_field, click_data1, click_data2, axis_ind, axis):
    global data
    if data_field is None or (click_data1 is None and click_data2 is None):
        raise PreventUpdate
    else:
        if isinstance(data_field, str):
            data_field = [data_field]
        if data.simpa_output["settings"][Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW[0]]:
            is_2d = True
        else:
            is_2d = False
        for click_data in [click_data1, click_data2]:
            if not click_data:
                continue
            x_c = click_data["points"][0]["x"]
            y_c = click_data["points"][0]["y"]
            if "z" in click_data["points"][0]:
                z_c = click_data["points"][0]["z"]
            else:
                z_c = None
            if not isinstance(z_c, int):
                z_c = None
            if not isinstance(x_c, int) or not isinstance(y_c, int):
                continue
            if (x_c, y_c, z_c) not in zip(data.click_points["x"], data.click_points["y"],
                                          data.click_points["z"]):
                data.click_points["x"].append(x_c)
                data.click_points["y"].append(y_c)
                data.click_points["z"].append(z_c)
        if not data.click_points["x"] or not data.click_points["y"]:
            raise PreventUpdate
        plot_data = list()
        for i, x in enumerate(data.click_points["x"]):
            y = data.click_points["y"][i]
            z = data.click_points["z"][i]
            for param in data_field:
                spectral_values = list()
                for wavelength in data.wavelengths:
                    if z and isinstance(z, int):
                        spectral_values.append(np.rot90(data.simpa_data_fields[param][wavelength], 1)[x, y, z])
                    elif is_2d:
                        spectral_values.append(np.rot90(data.simpa_data_fields[param][wavelength], 1)[y, x])
                    else:
                        # 3D data but no "z" in clickData
                        array = np.take(np.rot90(data.simpa_data_fields[param][wavelength], 1), indices=axis_ind,
                                        axis=axis)
                        spectral_values.append(array[y, x])
                plot_data += [go.Scatter(x=data.wavelengths, y=spectral_values, mode="lines+markers",
                                         name=param + f" x={x}, y={y}, z={z}")]
        return go.Figure(data=plot_data)


@app.callback(
    Output("general_info", "children"),
    Output("toast_content", "children"),
    Output("alert_toast", "is_open"),
    Input("param1", "value"),
    State("channel_slider", "value"),
)
def update_general_info(param1, wv):
    if not param1:
        raise PreventUpdate
    global data
    if data.simpa_output["settings"][Tags.SIMULATION_EXTRACT_FIELD_OF_VIEW[0]]:
        is_2d = True
    else:
        is_2d = False
    array = data.simpa_data_fields[param1][wv]
    shape = array.shape
    size = array.size * len(data.wavelengths)
    n_finite = [np.isfinite(data.simpa_data_fields[param1][w]).sum() for w in data.wavelengths]
    n_finite = np.sum(n_finite)
    n_not_finite = size - n_finite
    children = [
        dcc.Markdown(f'''
        Data is `{'2D' if is_2d else '3D'}`\n
        Data shape per channel is `{shape}`\n
        Data size per parameter is `{size}`\n
        Not finite points: `{n_not_finite} -> {100*n_not_finite/size}%`\n
        ''')
    ]
    if n_not_finite:
        open_toast = True
        toast_child = [dcc.Markdown(f"Found not finite values in array: `{n_not_finite} -> {100*n_not_finite/size}%`")]
    else:
        open_toast = False
        toast_child = []
    return children, toast_child, open_toast


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
    app.run_server(debug=True, host=host_name, port=port)
