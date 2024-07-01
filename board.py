from dash import Dash, html, dcc, callback, Output, Input
import dash_daq as daq
import plotly.express as px
import pandas as pd
from collections import defaultdict
from plot import create_graph_access_pattern, create_graph_req_rate, create_graph_size_dist, create_graph_popularity, create_graph_reuse_dist,\
    create_graph_reuse_heatmap, create_graph_size_heatmap, create_graph_popularity_decay
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
height = '50vh'

# --------------------- read in all generated result ---------------------
result_path = "."
import glob, os
result_files = glob.glob(os.path.join(result_path, '*'))
result_dict = defaultdict(list)
for file in result_files:
    base_name_without_suffix = os.path.basename(file[:file.rfind(".")])
    result_dict[base_name_without_suffix].append(file)

trace_list = list(result_dict.keys())
try:
    trace_list.remove('sta')
    trace_list.remove('traceSta')
except ValueError:
    logger.info("[Info] No sta and traceSta")
    
traces = pd.Series(trace_list)

# -------------------------------------------------------------------------

app = Dash()
server = app.server

app.layout = html.Div([
    html.H1(children='TraceBoard', style={'textAlign':'center'}),
    dcc.Dropdown(traces.unique(), placeholder='select a trace', id='trace-selection'),
    html.Hr(),
    html.Div(children='''
        the number of objects to plot
    '''),
        dcc.Input(
        placeholder='Enter a value...',
        type='number',
        value=500,
        id='n-obj-to-plot'
    ),
    daq.BooleanSwitch(id="pb", on=False),
    # ------------------------------------------------------------------------- done
    html.H2(children='Access Patern', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-access-pattern-real', style={'flex': '1', 'height': height}),
        dcc.Graph(id='graph-access-pattern-virtual', style={'flex': '1', 'height': height})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    # ------------------------------------------------------------------------- done
    html.H2(children='Request Rate', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-request-rate', style={'flex': '1', 'height': height}),
        dcc.Graph(id='graph-request-rate-byte', style={'flex': '1', 'height': height}),
        dcc.Graph(id='graph-request-rate-object', style={'flex': '1', 'height': height})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    # -------------------------------------------------------------------------
    html.H2(children='Size Distribution & Popularity', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-size-dist', style={'flex': '1', 'height': height}),
        dcc.Graph(id='graph-popularity', style={'flex': '1', 'height': height})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    # -------------------------------------------------------------------------
    html.H2(children='Reuse Distribution', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-reuse-dist-real', style={'flex': '1', 'height': height}),
        dcc.Graph(id='graph-reuse-dist-virtual', style={'flex': '1', 'height': height})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    
    # -------------------------------------------------------------------------
    html.Hr(),
    html.H2(children='Expensive Figures', style={'textAlign':'center'}),
    # -------------------------------------------------------------------------
    html.H2(children='Size Distribution Heatmap', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-size-heat-req', style={'flex': '1', 'height': height}),
        dcc.Graph(id='graph-size-heat-obj', style={'flex': '1', 'height': height})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    # -------------------------------------------------------------------------
    html.H2(children='Reuse Distribution Heatmap', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-reuse-heat-real', style={'flex': '1', 'height': height}),
        dcc.Graph(id='graph-reuse-heat-virtual', style={'flex': '1', 'height': height})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    # -------------------------------------------------------------------------
    html.H2(children='Popularity Decay', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-pd', style={'flex': '1', 'height': height}),
        dcc.Graph(id='graph-pd-heat', style={'flex': '1', 'height': height})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    
])

@callback(
    [Output('graph-access-pattern-real', 'figure'),
     Output('graph-access-pattern-virtual', 'figure')
    ],
    [Input('trace-selection', 'value'),
    Input('n-obj-to-plot', 'value')]
)
def update_graph_access_pattern(trace, n_obj_to_plot):
    if trace is None:
        logger.debug("No datapath provided.")
        return {}, {}
    if result_path + '/' + trace + '.accessRtime' in result_dict[trace]:
        fig_access_pattern_r = create_graph_access_pattern(result_path + '/' + trace + '.accessRtime', n_obj_to_plot, logger)
    else:
        logger.debug("No .accessRtime file.")
    if result_path + '/' + trace + '.accessVtime' in result_dict[trace]:
        fig_access_pattern_v = create_graph_access_pattern(result_path + '/' + trace + '.accessVtime', n_obj_to_plot, logger)
    else:
        logger.debug("No .accessVtime file.")
    return fig_access_pattern_r, fig_access_pattern_v
        

@callback(
    [Output('graph-request-rate', 'figure'),
     Output('graph-request-rate-byte', 'figure'),
     Output('graph-request-rate-object', 'figure'),
    ],
    Input('trace-selection', 'value'),
)
def update_graph_req_rate(trace):
    if trace is None:
        logger.debug("No datapath provided.")
        return {}, {}, {}
    
    if result_path + '/' + trace + '.reqRate_w300' in result_dict[trace]:
        fig_rr, fig_rr_b, fig_rr_o = create_graph_req_rate(result_path + '/' + trace + '.reqRate_w300', logger)
    else:
        logger.debug("No .reqRate_w300 file.")
    return fig_rr, fig_rr_b, fig_rr_o


@callback(
    Output('graph-size-dist', 'figure'),
    Input('trace-selection', 'value'),
)
def update_graph_size_dist(trace):
    if trace is None:
        logger.debug("No datapath provided.")
        return {}
    
    if result_path + '/' + trace + '.size' in result_dict[trace]:
        fig = create_graph_size_dist(result_path + '/' + trace + '.size', logger)
    else:
        logger.debug("No .size file.")
    return fig

@callback(
    Output('graph-popularity', 'figure'),
    Input('trace-selection', 'value'),
)
def update_graph_popularity(trace):
    if trace is None:
        logger.debug("No datapath provided.")
        return {}
    
    if result_path + '/' + trace + '.popularity' in result_dict[trace]:
        fig = create_graph_popularity(result_path + '/' + trace + '.popularity', logger)
    else:
        logger.debug("No .size file.")
    return fig

@callback(
    [Output('graph-reuse-dist-real', 'figure'),
     Output('graph-reuse-dist-virtual', 'figure')
    ],
    Input('trace-selection', 'value'),
)
def update_graph_reuse_dist(trace):
    if trace is None:
        logger.debug("No datapath provided.")
        return {}, {}
    if result_path + '/' + trace + '.reuse' in result_dict[trace]:
        fig_reuse_dist_r, fig_reuse_dist_v = create_graph_reuse_dist(result_path + '/' + trace + '.reuse', logger)
    else:
        logger.debug("No .reuse file.")
    return fig_reuse_dist_r, fig_reuse_dist_v

# -------------------------------------------------------
@callback(
    [Output('graph-size-heat-req', 'figure'),
     Output('graph-size-heat-obj', 'figure')
    ],
    [Input('trace-selection', 'value'),
     Input('pb', 'on')]
)
def update_graph_size_heat(trace, on):
    if on:
        if trace is None:
            logger.debug("No datapath provided.")
            return {}, {}
        if result_path + '/' + trace + '.sizeWindow_w300_obj' in result_dict[trace]\
            and result_path + '/' + trace + '.sizeWindow_w300_req' in result_dict[trace]:
            fig_size_heat_1, fig_size_heat_2 = create_graph_size_heatmap(result_path + '/' + trace + '.sizeWindow_w300', logger)
        else:
            logger.debug("No .sizeWindow_w300_obj/req file.")
        return fig_size_heat_1, fig_size_heat_2
    else:
        return {}, {}

@callback(
    [Output('graph-reuse-heat-real', 'figure'),
     Output('graph-reuse-heat-virtual', 'figure')
    ],
    [Input('trace-selection', 'value'),
     Input('pb', 'on')]
)
def update_graph_reuse_heat(trace, on):
    if on:
        if trace is None:
            logger.debug("No datapath provided.")
            return {}, {}
        if result_path + '/' + trace + '.reuseWindow_w300_rt' in result_dict[trace]\
            and result_path + '/' + trace + '.reuseWindow_w300_vt' in result_dict[trace]:
            fig_reuse_heat_1, fig_reuse_heat_2 = create_graph_reuse_heatmap(result_path + '/' + trace + '.reuseWindow_w300', logger)
        else:
            logger.debug("No .reuseWindow_w300_rt/vt file.")
        return fig_reuse_heat_1, fig_reuse_heat_2
    else:
        return {}, {}

@callback(
    [Output('graph-pd', 'figure'),
     Output('graph-pd-heat', 'figure')
    ],
    [Input('trace-selection', 'value'),
     Input('pb', 'on')]
)
def update_popularity_decay(trace, on):
    if on:
        if trace is None:
            logger.debug("No datapath provided.")
            return {}, {}
        if result_path + '/' + trace + '.popularityDecay_w300_obj' in result_dict[trace]:
            fig_pd, fig_pd_heat = create_graph_popularity_decay([result_path + '/' + trace + '.popularityDecay_w300_obj'], logger)
        else:
            logger.debug("No .popularityDecay_w300_obj file.")
        return fig_pd, fig_pd_heat
    else:
        return {}, {}

if __name__ == '__main__':
    app.run(debug=True, dev_tools_ui=False, port=8888)

