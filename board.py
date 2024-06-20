from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from collections import defaultdict
from plot import create_graph_access_pattern, create_graph_req_rate, create_graph_size_dist, create_graph_popularity

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --------------------- read in all generated result ---------------------
result_path = "/home/haocheng/libCacheSim/scripts/traceAnalysis/results"
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

app.layout = [
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
    # ------------------------------------------------------------------------- done
    html.H2(children='Access Patern', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-access-pattern-real', style={'flex': '1', 'height': '30vh'}),
        dcc.Graph(id='graph-access-pattern-virtual', style={'flex': '1', 'height': '30vh'})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    # ------------------------------------------------------------------------- done
    html.H2(children='Request Rate', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-request-rate', style={'flex': '1', 'height': '30vh'}),
        dcc.Graph(id='graph-request-rate-byte', style={'flex': '1', 'height': '30vh'}),
        dcc.Graph(id='graph-request-rate-object', style={'flex': '1', 'height': '30vh'})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    # -------------------------------------------------------------------------
    html.H2(children='Size Distribution & Popularity', style={'textAlign':'left'}),
    html.Div([
        dcc.Graph(id='graph-size-dist', style={'flex': '1', 'height': '30vh'}),
        dcc.Graph(id='graph-popularity', style={'flex': '1', 'height': '30vh'})
    ], style={'display': 'flex', 'flex-direction': 'row'}),
]

@callback(
    [Output('graph-access-pattern-real', 'figure'),
     Output('graph-access-pattern-virtual', 'figure')
    ],
    [Input('trace-selection', 'value'),
    Input('n-obj-to-plot', 'value')]
)
def update_graph_access_pattern_r(trace, n_obj_to_plot):
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
    Output('graph-graph-size-dist', 'figure'),
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
    Output('graph-graph-polularity', 'figure'),
    Input('trace-selection', 'value'),
)
def update_graph_polularity(trace):
    if trace is None:
        logger.debug("No datapath provided.")
        return {}
    
    if result_path + '/' + trace + '.polularity' in result_dict[trace]:
        fig = create_graph_popularity(result_path + '/' + trace + '.polularity', logger)
    else:
        logger.debug("No .size file.")
    return fig

if __name__ == '__main__':
    app.run(debug=True)
