import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import regex as re
from collections import Counter
from utils.data_utils import conv_to_cdf
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit
    
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
    with open(path, 'r') as ifile:
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
    with open(path, 'r') as ifile:
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


def create_graph_size_dist(path, logger):
    obj_size_req_cnt, obj_size_obj_cnt = read_size_dist(path)

    if len(obj_size_obj_cnt) == 0 or len(obj_size_req_cnt) == 0:
        logger.debug("empty size dist")
    x_req, y_req = conv_to_cdf(None, data_dict=obj_size_req_cnt)
    x_obj, y_obj = conv_to_cdf(None, data_dict=obj_size_obj_cnt)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_req, y=y_req, mode='lines', name='Request'))
    fig.add_trace(go.Scatter(x=x_obj, y=y_obj, mode='lines', name='Object'))

    fig.update_layout(
        title="Size Distribution",
        xaxis_title="Object size (Byte)",
        yaxis_title="Fraction of requests (CDF)",
        legend_title="Legend",
        xaxis_type="log",
        template="plotly_white"
    )

    return fig


def read_size_dist(path):
    with open(path, 'r') as ifile:
        _ = ifile.readline()  # read data line
        desc_line = ifile.readline()
        m = re.match(r"# object_size: req_cnt", desc_line)
        assert "# object_size: req_cnt" in desc_line or "# object_size: freq" in desc_line, (
            "the input file might not be size data file, desc line "
            + desc_line
            + " data "
            + path
        )

        obj_size_req_cnt, obj_size_obj_cnt = {}, {}
        for line in ifile:
            if line[0] == "#" and "object_size: obj_cnt" in line:
                break
            else:
                size, count = [int(i) for i in line.split(":")]
                obj_size_req_cnt[size] = count

        for line in ifile:
            size, count = [int(i) for i in line.split(":")]
            obj_size_obj_cnt[size] = count

    return obj_size_req_cnt, obj_size_obj_cnt

    
def create_graph_popularity(path, logger):
    sorted_freq, _ = read_popularity(path)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=sorted_freq,
        mode='lines',
        name='Frequency'
    ))

    fig.update_layout(
        title="Popularity Zipf Curve",
        xaxis_title="Object rank",
        yaxis_title="Frequency",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )

    # pdf_path = "{}/{}_pop_rank.pdf".format(FIG_DIR, figname_prefix)
    # pio.write_image(fig, pdf_path, format="pdf")
    # logger.info(f"Plot saved to {pdf_path}")

    x = np.log(np.arange(1, 1 + len(sorted_freq)))
    y = np.log(np.array(sorted_freq))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    if sorted_freq[0] < 100:
        s = "{:48} {:12} obj alpha 0, r^2 0 (the most popular object has less than 100 requests)".format(
            path,
            len(sorted_freq),
        )
    else:
        s = "{:48} {:12} obj alpha {:.4f}, r^2 {:.4f}".format(
            path, len(sorted_freq), -slope, r_value * r_value
        )

    logger.info(s)
    return fig


def read_popularity(path):
    
    with open(path) as ifile:
        data_line = ifile.readline()
        desc_line = ifile.readline()
        assert "# freq (sorted):cnt" in desc_line, (
            "the input file might not be popularity freq data file " + "data " + path
        )

        sorted_freq = []
        freq_cnt = Counter()
        for line in ifile:
            freq, cnt = [int(i) for i in line.strip("\n").split(":")]
            for i in range(cnt):
                sorted_freq.append(freq)
            freq_cnt[freq] += 1

    return sorted_freq, freq_cnt

def create_graph_reuse_dist(path, logger):
    reuse_rtime_count, reuse_vtime_count = read_graph_reuse_dist(path)

    x, y = conv_to_cdf(None, data_dict=reuse_rtime_count)
    if x[0] < 0:
        x = x[1:]
        y = [y[i] - y[0] for i in range(1, len(y))]
    # logger.debug(x)
    fig_real = go.Figure()
    fig_real.add_trace(
        go.Scatter(x=[i / 3600 for i in x], y=y, mode='lines', name='Real Time Reuse'),
    )
    fig_real.update_xaxes(title_text="Time (Hour)", type="log")
    fig_real.update_yaxes(title_text="Fraction of requests (CDF)")

    x, y = conv_to_cdf(None, data_dict=reuse_vtime_count)
    if x[0] < 0:
        x = x[1:]
        y = [y[i] - y[0] for i in range(1, len(y))]

    fig_virtural = go.Figure()
    fig_virtural.add_trace(
        go.Scatter(x=[i for i in x], y=y, mode='lines', name='Virtual Time Reuse')
    )
    fig_virtural.update_xaxes(title_text="Virtual time (# requests)", type="log",)
    fig_virtural.update_yaxes(title_text="Fraction of requests (CDF)")

    return fig_real, fig_virtural

def read_graph_reuse_dist(path):
    with open(path, 'r') as ifile:
        _ = ifile.readline()
        desc_line = ifile.readline()
        m = re.match(r"# reuse real time: freq \(time granularity (?P<tg>\d+)\)", desc_line)
        assert m is not None, (
            "the input file might not be reuse data file, desc line "
            + desc_line
            + " data "
            + path
        )

        rtime_granularity = int(m.group("tg"))
        log_base = 1.5

        reuse_rtime_count, reuse_vtime_count = {}, {}

        for line in ifile:
            if line[0] == "#" and "virtual time" in line:
                m = re.match(
                    r"# reuse virtual time: freq \(log base (?P<lb>\d+\.?\d*)\)", line
                )
                assert m is not None, (
                    "the input file might not be reuse data file, desc line "
                    + line
                    + " data "
                    + path
                )
                log_base = float(m.group("lb"))
                break
            elif len(line.strip()) == 0:
                continue
            else:
                reuse_time, count = [int(i) for i in line.split(":")]
                if reuse_time < -1:
                    print("find negative reuse time " + line)
                reuse_rtime_count[reuse_time * rtime_granularity] = count

        for line in ifile:
            if len(line.strip()) == 0:
                continue
            else:
                reuse_time, count = [int(i) for i in line.split(":")]
                if reuse_time < -1:
                    print("find negative reuse time " + line)
                reuse_vtime_count[log_base**reuse_time] = count

    return reuse_rtime_count, reuse_vtime_count


def read_reuse_heatmap(path: str):
    with open(path) as ifile:
        data_line = ifile.readline()
        desc_line = ifile.readline()

        time_granularity, log_base = 0, 0
        if "real time" in desc_line:
            m = re.search(
                r"# reuse real time distribution per window \(time granularity (?P<tg>\d+), time window (?P<tw>\d+)\)",
                desc_line,
            )
            assert m is not None, (
                "the input file might not be reuse heatmap data file, desc line "
                + desc_line
                + "data "
                + path
            )
            time_granularity = int(m.group("tg"))
            time_window = int(m.group("tw"))
        elif "virtual time" in desc_line:
            m = re.search(
                r"# reuse virtual time distribution per window \(log base (?P<lb>\d+\.?\d*), time window (?P<tw>\d+)\)",
                desc_line,
            )
            assert m is not None, (
                "the input file might not be reuse heatmap data file, desc line "
                + desc_line
                + "data "
                + path
            )
            log_base = float(m.group("lb"))
            time_window = int(m.group("tw"))
        else:
            raise RuntimeError(
                "the input file might not be reuse heatmap data file, desc line "
                + desc_line
                + "data "
                + path
            )

        reuse_time_distribution_list = []

        for line in ifile:
            if len(line.strip()) == 0:
                continue
            else:
                reuse_time_distribution_list.append(
                    [int(i) for i in line.strip(",\n").split(",")]
                )

    dim = max([len(l) for l in reuse_time_distribution_list])
    plot_data = np.ones((len(reuse_time_distribution_list), dim))
    for idx, l in enumerate(reuse_time_distribution_list):
        plot_data[idx][: len(l)] = np.cumsum(np.array(l) / sum(l))
    plot_data = plot_data.T

    return plot_data, time_granularity, time_window, log_base


def create_graph_reuse_heatmap(path: str, logger):

    plot_data_rt, time_granularity, time_window, log_base = read_reuse_heatmap(
        path + "_rt"
    )
    assert log_base == 0

    fig_rt = go.Figure(data=go.Heatmap(
        z=plot_data_rt,
        colorscale='Jet',
        zmin=0,
        zmax=1,
        colorbar=dict(title="CDF")
    ))

    fig_rt.update_layout(
        title=f"Reuse Time Heatmap (Real Time) - {path}",
        xaxis=dict(title="Time (hour)", tickvals=[i for i in range(plot_data_rt.shape[1])], ticktext=[f"{i*time_window/3600:.0f}" for i in range(plot_data_rt.shape[1])]),
        yaxis=dict(title="Reuse Time (hour)", tickvals=[i for i in range(plot_data_rt.shape[0])], ticktext=[f"{i*time_granularity/3600:.0f}" for i in range(plot_data_rt.shape[0])])
    )


    plot_data_vt, time_granularity, time_window, log_base = read_reuse_heatmap(
        path + "_vt"
    )
    assert time_granularity == 0

    fig_vt = go.Figure(data=go.Heatmap(
        z=plot_data_vt,
        colorscale='Jet',
        zmin=0,
        zmax=1,
        colorbar=dict(title="CDF")
    ))

    fig_vt.update_layout(
        title=f"Reuse Time Heatmap (Virtual Time) - {path}",
        xaxis=dict(title="Time (hour)", tickvals=[i for i in range(plot_data_vt.shape[1])], ticktext=[f"{i*time_window/3600:.0f}" for i in range(plot_data_vt.shape[1])]),
        yaxis=dict(title="Reuse Time (# request)", tickvals=[i for i in range(plot_data_vt.shape[0])], ticktext=[f"{log_base**i:.0f}" for i in range(plot_data_vt.shape[0])])
    )
    return fig_rt, fig_vt



def read_size_heatmap(path):
    with open(path) as ifile:
        data_line = ifile.readline()
        desc_line = ifile.readline()
        m = re.search(
            r"# (object_size): \w\w\w_cnt \(time window (?P<tw>\d+), log_base (?P<logb>\d+\.?\d*), size_base (?P<sizeb>\d+)\)",
            desc_line,
        )
        assert m is not None, (
            "the input file might not be size heatmap data file, desc line "
            + desc_line
            + " data "
            + path
        )

        time_window = int(m.group("tw"))
        log_base = float(m.group("logb"))
        size_base = int(m.group("sizeb"))
        size_distribution_over_time = []

        for line in ifile:
            count_list = line.strip("\n,").split(",")
            size_distribution_over_time.append(count_list)

    dim = max([len(l) for l in size_distribution_over_time])
    plot_data = np.zeros((len(size_distribution_over_time), dim))

    for idx, l in enumerate(size_distribution_over_time):
        l = np.array(l, dtype=np.float64)
        l = l / np.sum(l)
        plot_data[idx][: len(l)] = l

    return plot_data.T, time_window, log_base, size_base

def create_graph_size_heatmap(path, logger):
    plot_data, time_window, log_base, size_base = read_size_heatmap(
        path + "_req"
    )

    fig_req = go.Figure(data=go.Heatmap(
        z=plot_data,
        colorscale='Jet',
        colorbar=dict(title='Fraction')
    ))

    fig_req.update_layout(
        title="Size Heatmap (Request)",
        xaxis=dict(
            title='Time (hour)',
            tickmode='array',
            tickvals=np.arange(0, plot_data.shape[1], step=int(plot_data.shape[1]/10)),
            ticktext=[f"{x * time_window / 3600:.0f}" for x in np.arange(0, plot_data.shape[1], step=int(plot_data.shape[1]/10))]
        ),
        yaxis=dict(
            title='Request size (Byte)',
            tickmode='array',
            tickvals=np.arange(0, plot_data.shape[0], step=int(plot_data.shape[0]/10)),
            ticktext=[f"{log_base**x * size_base:.0f}" for x in np.arange(0, plot_data.shape[0], step=int(plot_data.shape[0]/10))]
        ),
        template="plotly_white"
    )


    plot_data, time_window, log_base, size_base = read_size_heatmap(
        path + "_obj"
    )

    fig_obj = go.Figure(data=go.Heatmap(
        z=plot_data,
        colorscale='Jet',
        colorbar=dict(title='Fraction')
    ))

    fig_obj.update_layout(
        title="Size Heatmap (Object)",
        xaxis=dict(
            title='Time (hour)',
            tickmode='array',
            tickvals=np.arange(0, plot_data.shape[1], step=int(plot_data.shape[1]/10)),
            ticktext=[f"{x * time_window / 3600:.0f}" for x in np.arange(0, plot_data.shape[1], step=int(plot_data.shape[1]/10))]
        ),
        yaxis=dict(
            title='Object size (Byte)',
            tickmode='array',
            tickvals=np.arange(0, plot_data.shape[0], step=int(plot_data.shape[0]/10)),
            ticktext=[f"{log_base**x * size_base:.0f}" for x in np.arange(0, plot_data.shape[0], step=int(plot_data.shape[0]/10))]
        ),
        template="plotly_white"
    )

    return fig_req, fig_obj


def load_popularity_decay_data(path: str):
    import numpy.ma as ma
    import os

    ifile = open(path)
    _data_line = ifile.readline()
    desc_line = ifile.readline()
    assert "cnt for new" in desc_line, (
        "the input file might not be popularityDecay data file: " + path
    )
    time_window = int(desc_line.split()[11].strip("()"))
    window_cnt_list_list = []

    line = ifile.readline()
    if line == "":
        return None, None
    assert line == "0,\n", f"the first line should be 0, it is {line}" + path
    for line in ifile:
        l = [int(i) for i in line.strip("\n,").split(",")]
        assert l[-1] == 0, "the last element should be 0, " + path
        assert len(l) - 2 == len(
            window_cnt_list_list
        ), path + " data len is inconsistent {} != {}".format(
            len(l) - 2, len(window_cnt_list_list)
        )
        window_cnt_list_list.append(l[:-1])

    trace_length_rtime = len(window_cnt_list_list) * time_window
    print(
        "{} trace length {:.2f} days".format(
            os.path.basename(path), trace_length_rtime / 86400
        )
    )

    ifile.close()

    dim = len(window_cnt_list_list)
    data = np.full((dim, dim), -1, dtype=np.double)
    for idx, l in enumerate(window_cnt_list_list):
        data[idx][: len(l)] = l

    data = ma.array(data, mask=data < 0)
    data = data / np.diag(data)
    data = data.T

    return data, time_window

import os
def create_graph_popularity_decay(
    path_list, logger
):
    plot_data_list = list()
    label_list = list()
    for path in path_list:
        plot_data, time_window = load_popularity_decay_data(path)
        if plot_data is None:
            logger.debug("No data")
            return {}, {}
        plot_data_list.append(plot_data)
        

    label_list = [os.path.basename(path) for path in path_list]
    basename  = label_list[0]
    fig = go.Figure()

    for data_idx, plot_data in enumerate(plot_data_list):
        plot_data_matrix = plot_data.copy()

        for i in range(1, plot_data_matrix.shape[0]):
            plot_data_matrix[i, :-i] = plot_data[i, i:]
            plot_data_matrix[i, -i:] = np.nan

        n_skip = plot_data_matrix.shape[0] // 8
        plot_data_matrix = plot_data_matrix[n_skip:-n_skip, 1 : -n_skip - 1]
        mean_req_prob = np.nanmean(plot_data_matrix, axis=0)

        if "io_traces" in  basename or "alibaba" in basename:
            mean_req_prob = mean_req_prob[: 3600 * 24 * 21 // time_window]
        else:
            mean_req_prob = mean_req_prob[: 3600 * 24 * 5 // time_window]

        x_data = [(i + 1) * time_window for i in range(mean_req_prob.shape[0])]

        if label_list:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=mean_req_prob,
                mode='lines',
                name=label_list[data_idx]
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=mean_req_prob,
                mode='lines'
            ))

    fig.update_layout(
        title="Popularity Decay Over Time",
        xaxis_title="Age",
        yaxis_title="Request probability",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white"
    )

    if "io_traces" in basename or "alibaba" in basename:
        fig.update_xaxes(
            tickvals=[300, 3600, 86400, 86400 * 2, 86400 * 4, 86400 * 8, 86400 * 16],
            ticktext=["5 min", "1 hour", "1 day", "", "4 day", "", "16 day"]
        )
    else:
        fig.update_xaxes(
            tickvals=[300, 3600, 86400, 86400 * 2, 86400 * 4],
            ticktext=["5 min", "1 hour", "1 day", "", "4 day"]
        )
        
    # --------------------

    plot_data, time_window = load_popularity_decay_data(path_list[0])
    
    
    # skip the first window which is always 1
    for i in range(plot_data_matrix.shape[0]):
        plot_data_matrix[i, i] = np.nan

    plot_data_matrix[plot_data_matrix < 1e-18] = np.nan

    fig_heat = go.Figure(data=go.Heatmap(
        z=plot_data_matrix,
        colorscale='PuBu',
        colorbar=dict(title='Request Probability'),
        zmin=np.nanmin(plot_data_matrix),
        zmax=np.nanmax(plot_data_matrix)
    ))

    tickvals = [i for i in range(plot_data_matrix.shape[0])]
    ticktext = ["{:.0f}".format(i * time_window / 3600) for i in range(plot_data_matrix.shape[0])]
    tickvals, ticktext = tickvals[:: len(tickvals) // 4], ticktext[:: len(ticktext) // 4]

    fig_heat.update_layout(
        title="Popularity Decay Heatmap",
        xaxis=dict(
            title="Time (Hour)",
            tickvals=tickvals,
            ticktext=ticktext
        ),
        yaxis=dict(
            title="Creation time (Hour)",
            tickvals=tickvals,
            ticktext=ticktext
        ),
        template="plotly_white"
    )

    return fig, fig_heat