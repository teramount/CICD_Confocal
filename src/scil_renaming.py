import os
import pandas as pd
# path_to_data = r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\SCIL Polaris\master\CSV FIles after stamp 4'
# path_to_naming_file = r"G:\Shared drives\Design\Reports\scil_polaris\scan plans\B-065_master_Polaris.csv"


def rename(data_path, naming_path, base_name):
    name_file = pd.read_csv(naming_path)
    # base_name = 'B-065-000-000'
    for filename in os.listdir(data_path):
        try:
            name, ext = os.path.splitext(filename)  # split filename and extension
            split_name = name.split('_')
            serial = int(split_name[1])-3
            relevant_line = name_file.iloc[serial]['Component ID']
            new_name = '-'.join([base_name, relevant_line])+ext
            old_path = os.path.join(data_path, filename)
            new_path = os.path.join(data_path, new_name)
            os.rename(old_path, new_path)
        except:
            pass

