#!/usr/local/bin/ifbpy2 --never-use-in-production
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import json
import pandas as pd
import numpy as np


def trace_event(row):
    return dict(
        name=row["Name"],
        tid=row["Context"],
        cat="cuda",
        ts=row["Start"],
        ph="X",
        pid=row["pid"],
        dur=row["Duration"],
        args=dict(
            stream=row["Stream"]
        )
    )


def parse(filename):
    def parse_int(x):
        try:
            return int(x)
        except ValueError:
            return -1

    def parse_pid(device):
        if isinstance(device, str):
            return device
        assert np.isnan(device)
        return "Host"

    df = pd.read_csv(filename)[1:]
    df.Duration = df.Duration.map(lambda x: float(x))
    df.Start = df.Start.map(lambda x: float(x))
    df.Stream = df.Stream.map(lambda x: parse_int(x))
    df.Context = df.Context.map(lambda x: parse_int(x))
    df["pid"] = df.Device.map(lambda x: parse_pid(x))
    return df


def main():
    df = parse(sys.argv[1])
    events = [trace_event(row) for row in df.to_dict("records")]
    print(json.dumps(events, indent=2))

if __name__ == "__main__":
    main()
