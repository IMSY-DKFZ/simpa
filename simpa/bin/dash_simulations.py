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

`pip install dash dash-table dash-daq plotly plotly-express dash-bootstrap-components pandas numpy`

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
from simpa.io_handling import load_hdf5


external_stylesheets = [dbc.themes.JOURNAL, '.assets/dcc.css']
app = dash.Dash(external_stylesheets=external_stylesheets, title="SIMPA")

simpa_logo = './.assets/simpa_logo.png'
cami_logo = './.assets/CAMIC_logo-wo_DKFZ.png'
encoded_simpa_logo = (base64.b64encode(open(simpa_logo, 'rb').read())).decode()
encoded_cami_logo = (base64.b64encode(open(cami_logo, 'rb').read())).decode()


class DataContainer:

    simpa_output = None
    simpa_data_fields = None
    wavelengths = None
    plot_types = ['scatter3d', 'imshow', 'histogram-volume', 'histograme-channel', 'box', 'violin', 'contour']
DEFAULT_COLORSCALE = ['rgb(5,48,97)', 'rgb(33,102,172)', 'rgb(67,147,195)', 'rgb(146,197,222)', 'rgb(209,229,240)',
                      'rgb(247,247,247)', 'rgb(253,219,199)', 'rgb(244,165,130)', 'rgb(214,96,77)',
                      'rgb(178,24,43)', 'rgb(103,0,31)']

data = DataContainer()

app.layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H4("SIMPA Visualization Tool"),
            html.H6("CAMI, Computer Assisted Medical Interventions"),
        ], width=6),
        dbc.Col([
            html.Img(src='data:image/png;base64,{}'.format(encoded_simpa_logo), width='50%')
        ], width=3),
        dbc.Col([
            html.Img(src='data:image/png;base64,{}'.format(encoded_cami_logo), width='50%')
        ], width=3)
    ]),
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
                        # multi=True,
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
                        options=[{'label': 'x', 'value': 'x'},
                                 {'label': 'y', 'value': 'y'},
                                 {'label': 'z', 'value': 'z'}],
                        value='z'
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
                    html.P("Limit plotted values as %"),
                    dcc.RangeSlider(
                        id="plot_scaler",
                        min=0,
                        max=100,
                        value=[10, 90],
                        marks={i: {'label': str(i)} for i in list(range(101))[::10]},
                        allowCross=False,
                        pushable=5,
                        tooltip=dict(always_visible=True, placement="top"),
                        persistence_type="session",
                    )
                ], width=2),
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Parameter visualization"),
                            html.Div(
                                id="plot_div_1",
                                children=[
                                    dcc.Graph(id="plot_11",
                                              hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                              style={"width": "50%", "display": 'inline-block'}),
                                    dcc.Graph(id="plot_12",
                                              hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                              style={"width": "50%", "display": 'inline-block'}),
                                    html.Div([
                                        dcc.Dropdown(
                                            id="plot_type1",
                                            multi=False,
                                            placeholder="Plot type",
                                            persistence_type="session",
                                            disabled=True,
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
                                            # min=0,
                                            # max=100,
                                            # value=50,
                                            # step=1,
                                            tooltip=dict(always_visible=True, placement="top"),
                                            updatemode="mouseup",
                                            persistence_type="session",
                                            id="channel_slider",
                                            # marks={
                                            #     0: {'label': '0', 'style': {'color': '#77b0b1'}},
                                            #     26: {'label': '26'},
                                            #     37: {'label': '37'},
                                            #     100: {'label': '100', 'style': {'color': '#f50'}}
                                            # }
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
                                        multi=False,
                                        placeholder="Parameter to plot over wavelengths",
                                        persistence_type="session",
                                        disabled=True,
                                    ),
                                    dcc.Graph(id="plot_21",
                                              hoverData={'points': [{'x': 0, 'y': 0, 'customdata': None}]},
                                              style={"width": "100%", "display": 'inline-block'})
                                ]
                            )
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    html.H6("Plot information"),
                    html.Hr()
                ], width=2)
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
], style={'padding': 40})


# class DataContainer(object):
#     plot_types = ['scatter3d', 'imshow', 'histogram-volume', 'histograme-channel', 'box', 'violin', 'contour']
#     wavelengths = np.arange(500, 1000, 5)
#     vol_shape = (60, 60, 60)
#     X, Y, Z = np.mgrid[:1:20j, :1:20j, :1:20j]
#     vol = (X - 1) ** 2 + (Y - 1) ** 2 + Z ** 2
#     data = (X, Y, Z, vol)
#
#
# data_container = DataContainer()


# @app.callback(Output("plot_11", "figure"),
#               [Input("plot_type1", "value"),
#                Input("param1", "value"),
#                Input("colorscale_picker", "colorscale")])
# def update_plot11(plot_type, param, colorscale):
#     if plot_type == 'scatter3d':
#         if len(data_container.vol_shape) == 3:
#             x, y, z, v = data_container.data[param]
#             fig = go.Volume(
#                 x=x,
#                 y=y,
#                 z=z,
#                 v=v,
#                 opacity=0.2,
#                 colorscale=colorscale
#             )
#             return fig
#         else:
#             return {"data": []}
#     elif plot_type == "histogram-volume":
#         x, y, z, v = data_container.data
#         if isinstance(v, np.ndarray):
#             fig = px.histogram(v)
#             return fig
#         else:
#             return {"data": []}
#     else:
#         return {"data": []}


@app.callback(
    Output("file_selection", "options"),
    Input("data_path", "value")
)
def populate_file_selection(data_path):
    if os.path.isdir(data_path):
        file_list = os.listdir(data_path)
        options_list = [{'label': i, 'value': os.path.join(data_path, i)} for i in file_list]
        return options_list
    elif os.path.isfile(data_path):
        file_list = [os.path.basename(data_path)]
        options_list = [{'label': i, 'value': data_path} for i in file_list]
        return options_list
    else:
        raise PreventUpdate("Please select a file or folder!")


@app.callback(
    Output("file_selection", "value"),
    Input("file_selection", "options")
)
def set_default_file(options):
    if len(options) == 1:
        return options[0]["value"]


@app.callback(
    Output("param1", "options"),
    Output("param1", "disabled"),
    Output("channel_slider", "min"),
    Output("channel_slider", "max"),
    Output("channel_slider", "value"),
    Output("channel_slider", "step"),
    Output("param2", "options"),
    Output("param2", "disabled"),
    Input("file_selection", "value"),
)
def populate_file_params(file_path):
    if file_path is None:
        raise PreventUpdate
    print("hallo")
    data.simpa_output = load_hdf5(file_path)
    get_data_fields()
    options_list = [{'label': key, 'value': key} for key in data.simpa_data_fields.keys()]

    return options_list, False, min(data.wavelengths), max(data.wavelengths), data.wavelengths[0], data.wavelengths[1] - data.wavelengths[0], options_list, False


def get_data_fields():
    data.wavelengths = data.simpa_output["settings"]["wavelengths"]
    datafields = dict()
    sim_props = list(data.simpa_output["simulations"]["original_data"]["simulation_properties"]["{}".format(data.wavelengths[0])].keys())
    simulations = list(data.simpa_output["simulations"]["original_data"]["optical_forward_model_output"]["{}".format(data.wavelengths[0])].keys())

    for datafield in sim_props:
        datafields[datafield] = dict()
        for wavelength in data.wavelengths:
            datafields[datafield][wavelength] = data.simpa_output["simulations"]["original_data"] \
                ["simulation_properties"]["{}".format(wavelength)][datafield]

    for datafield in simulations:
        datafields[datafield] = dict()
        for wavelength in data.wavelengths:
            datafields[datafield][wavelength] = data.simpa_output["simulations"]["original_data"] \
                ["optical_forward_model_output"]["{}".format(wavelength)][datafield]

    data.simpa_data_fields = datafields


@app.callback(
    Output("plot_11", "figure"),
    Input("param1", "value"),
    Input("colorscale_picker", "colorscale"),
    Input("channel_slider", "value")
)
def plot_data_field(datafield, colorscale, wavelength):
    if datafield is None or wavelength is None:
        raise PreventUpdate
    else:
        datafield = np.rot90(data.simpa_data_fields[datafield][wavelength], 1)
        plotdata = [go.Heatmap(z=datafield, colorscale=colorscale)]
        return go.Figure(data=plotdata)


@app.callback(
    Output("plot_12", "figure"),
    Input("param2", "value"),
    Input("colorscale_picker", "colorscale"),
    Input("channel_slider", "value")
)
def plot_data_field(datafield, colorscale, wavelength):
    if datafield is None or wavelength is None:
        raise PreventUpdate
    else:
        datafield = np.rot90(data.simpa_data_fields[datafield][wavelength], 1)
        plotdata = [go.Heatmap(z=datafield, colorscale=colorscale)]
        return go.Figure(data=plotdata)


@app.callback(
    Output("plot_21", "figure"),
    Input("param1", "value"),
    Input("plot_11", "clickData")
)
def plot_spectrum(datafield, clickdata):
    if datafield is None or clickdata is None:
        raise PreventUpdate
    else:
        x = clickdata["points"][0]["x"]
        y = clickdata["points"][0]["y"]
        spectral_values = list()
        for wavelength in data.wavelengths:
            spectral_values.append(np.rot90(data.simpa_data_fields[datafield][wavelength], 1)[y, x])
        plotdata = [go.Scatter(x=data.wavelengths, y=spectral_values, mode="lines+markers")]
        return go.Figure(data=plotdata)


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
