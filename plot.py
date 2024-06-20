import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR = ["red", "orange"]

def create_graph_access_pattern(path, n_obj_to_plot, logger):
    logger.debug(f"file path: {path}")
    access_time_list = read_access_pattern(path, n_obj_to_plot)
    is_real_time = "Rtime" in path  # else is "Vtime"
    data = []
    for idx, ts_list in enumerate(access_time_list):
        for ts in ts_list:
            time = ts / 3600 if is_real_time else ts / 1e6
            data.append({"Time": time, "Object": idx})

    df = pd.DataFrame(data)
    if df.size == 0:
        return {}
    xlabel = "Time (hour)" if is_real_time else "Time (# million requests)"
    fig = px.scatter(df, x="Time", y="Object", title="Real Time" if is_real_time else "Virtual Time", labels={"Time": xlabel, "Object": "Sampled object"})
    return fig


def read_access_pattern(path, n_obj_to_plot):
    access_time_list = list()
    with open(path) as ifile:
        content = ifile.readlines()
        num_of_lines = len(content)
        n_total_obj = num_of_lines - 2  # exclude data line and desc line
        sample_ratio = max(1, n_total_obj // n_obj_to_plot)
        
        for i, line in enumerate(content[2:]):
            if len(line.strip()) == 0:
                continue
            elif i % sample_ratio == 0:
                access_time_list.append([float(i) for i in line.split(",")[:-1]])
    access_time_list.sort(key=lambda x: x[0])
    return access_time_list

def create_graph_req_rate(path, logger):
    req_rate_list, byte_rate_list, obj_rate_list, new_obj_rate_list, time_window = read_req_rate(path)
    x = [i * time_window / 3600 for i in range(len(req_rate_list))]

    # Request Rate Graph
    fig_req_rate = go.Figure()
    fig_req_rate.add_trace(go.Scatter(x=x, y=[r / 1000 for r in req_rate_list], mode='lines', name='Request Rate', line=dict(color=COLOR[0])))
    fig_req_rate.update_layout(title='Request Rate Over Time', xaxis_title='Time (Hour)', yaxis_title='Request Rate (KQPS)')

    # Byte Rate Graph
    fig_byte_rate = go.Figure()
    fig_byte_rate.add_trace(go.Scatter(x=x, y=[n_byte / 1024 / 1024 for n_byte in byte_rate_list], mode='lines', name='Byte Rate', line=dict(color=COLOR[0])))
    fig_byte_rate.update_layout(title='Byte Rate Over Time', xaxis_title='Time (Hour)', yaxis_title='Byte Rate (Mbps)')

    # Object Rate Graph
    fig_obj_rate = go.Figure()
    fig_obj_rate.add_trace(go.Scatter(x=x, y=obj_rate_list, mode='lines', name='Object Rate', line=dict(color=COLOR[0])))
    fig_obj_rate.add_trace(go.Scatter(x=x, y=new_obj_rate_list, mode='lines', name='New Object Rate', line=dict(color=COLOR[1], dash='dash')))
    fig_obj_rate.update_layout(title='Object Rate Over Time', xaxis_title='Time (Hour)', yaxis_title='Object Rate (QPS)', legend=dict(x=0.01, y=0.99))

    return fig_req_rate, fig_byte_rate, fig_obj_rate

def read_req_rate(path):
    req_rate_list = None
    time_window = None
    with open(path) as ifile:
        _ = ifile.readline()  # data line
        line = ifile.readline()
        assert "# req rate - time window" in line, "the input file might not be reqRate data file"
        time_window = int(line.split()[6].strip("()s "))
        req_rate_list = [float(i) for i in ifile.readline().split(",")[:-1]]
        assert "byte rate" in ifile.readline()
        byte_rate_list = [float(i) for i in ifile.readline().split(",")[:-1]]
        assert "obj rate" in ifile.readline()
        obj_rate_list = [float(i) for i in ifile.readline().split(",")[:-1]]
        assert "first seen obj (cold miss) rate" in ifile.readline()
        new_obj_rate_list = [float(i) for i in ifile.readline().split(",")[:-1]]
    return req_rate_list, byte_rate_list, obj_rate_list, new_obj_rate_list, time_window


def update_graph_size_dist(path):
    pass

def create_graph_popularity(path):
    pass
