import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 24  # Title
plt.rcParams['axes.labelsize'] = 18  # X and Y labels
plt.rcParams['xtick.labelsize'] = 16  # X tick labels
plt.rcParams['ytick.labelsize'] = 16  # Y tick labels

REF_PATH = r"G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\SCIL CSV B-065 Stamp 1\analysis\B-065-001-000 stamp1.csv"
REPS_PATH = [r"G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 1\analysis\B-065-polaris_rep165-1-1.csv",
             r"G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 2\analysis\B-065-polaris_rep265-1-2.csv",
             r"G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\Run 1\Wafer 5\analysis\B-065-polaris_rep565-1-5.csv"]
FIELD_TO_FIG_DICT = {'PHD': 0, 'axial_focal_length': 1, 'trans_focal_length': 2}
FIELD_LIST = ['PHD', 'axial_focal_length', 'trans_focal_length']
REP_IDX = [1, 2, 5]
def filter_only_relevant(ref_df, rep_df, key='column-row-channel'):
    common_ids = set(ref_df[key]).intersection(rep_df[key])
    return rep_df[rep_df[key].isin(common_ids)].copy()


def run():
    ref_df = pd.read_csv(REF_PATH)
    i = 0
    for path in REPS_PATH:
        rep_df = pd.read_csv(path)
        relevant_section = filter_only_relevant(ref_df, rep_df)
        common_ids = relevant_section["column-row-channel"]
        ref_aligned = ref_df[ref_df["column-row-channel"].isin(common_ids)].copy()
        merged = ref_aligned.merge(relevant_section, on="column-row-channel", suffixes=("_ref", "_rep"))
        for field in FIELD_LIST:
            x = merged[f'{field}_ref']
            y = merged[f'{field}_rep']
            plt.figure(FIELD_TO_FIG_DICT[field])
            plt.grid(True)
            if i == 0:
                # plt.figure(FIELD_TO_FIG_DICT[field])
                plt.plot(x, x, label='y=x', color='black')
                plt.title(f'correlation {field}')
            plt.scatter(x, y, label=f'rep {REP_IDX[i]}')
        i += 1
    for k in range(len(FIELD_LIST)):
        plt.figure(k)
        plt.legend()
        plt.xlabel('stamp data')
        plt.ylabel('replication data')
        plt.tight_layout()
    plt.show()


run()


