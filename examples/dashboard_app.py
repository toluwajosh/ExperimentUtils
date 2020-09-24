"""
Create a results dashboard with dash
"""

import argparse
import json
from pathlib import Path
from typing import List

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
colors = {"background": "#111111", "text": "#85c1e9", "accent": "#eb984e"}


def get_data() -> List:
    """Callback function to get results from directory

    Returns:
        [List]: A list of results object
    """
    results_dir = Path("./results")
    all_results = [x for x in results_dir.iterdir() if x.suffix == ".json"]
    all_data = []
    for data_path in all_results:
        with open(str(data_path)) as json_file:
            all_data.append(json.load(json_file))
    return all_data


app.layout = html.Div(
    id="all-results",
    style={"backgroundColor": colors["background"]},
    children=[
        html.H1(
            children="Results Dashboard",
            style={"textAlign": "center", "color": colors["text"]},
        ),
        html.Div(
            children="Quick visualization of experiment results.",
            style={"textAlign": "center", "color": colors["accent"]},
        ),
        html.Div(id="results-div"),
        html.Div(id="summaries-div"),
        # to retrieve results automatically at intervals
        dcc.Interval(
            id="interval-component", interval=5 * 1000, n_intervals=0  # in milliseconds
        ),
    ],
)


@app.callback(
    Output("results-div", "children"), [Input("interval-component", "n_intervals")]
)
def update_results_div(value):
    all_data = get_data()
    return (
        dcc.Graph(
            id="results-graph",
            figure={
                "layout": {
                    "title": "AUCs by class for MVTec Dataset",
                    "plot_bgcolor": colors["background"],
                    "paper_bgcolor": colors["background"],
                    "font": {"color": colors["text"]},
                },
                "data": [
                    {
                        "x": [x["category"] for x in data["results"]],
                        "y": [y["score"] for y in data["results"]],
                        "type": "bar",
                        "name": data["method"],
                    }
                    for data in all_data
                ],
            },
        ),
    )


@app.callback(
    Output("summaries-div", "children"), [Input("interval-component", "n_intervals")]
)
def update_summaries_div(value):
    all_data = get_data()
    return (
        dcc.Graph(
            id="summaries-graph",
            figure={
                "layout": {
                    "title": "Experiments Summary",
                    "plot_bgcolor": colors["background"],
                    "paper_bgcolor": colors["background"],
                    "font": {"color": colors["text"]},
                },
                "data": [
                    {
                        "x": list(data["summary"].keys()),
                        "y": list(data["summary"].values()),
                        "type": "bar",
                        "name": data["method"],
                    }
                    for data in all_data
                ],
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Dashboard Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # type: ignore
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address.")
    parser.add_argument("--port", type=str, default="5555", help="Port Number.")
    args = parser.parse_args()
    app.run_server(debug=True, host=args.host, port=args.port)
