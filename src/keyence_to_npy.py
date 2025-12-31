import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv



def convert_to_npy(data_path):
    # data_path = r'G:\.shortcut-targets-by-id\1gxJyFpoZnr6zgsREMz0mBaXXVkI-1ElW\Profiler analysis Software\Raw Data\Confocal Measurements\ASE_lm\Keyence U31, U33'
    for file in os.listdir(data_path):
        try:
            if file.endswith('.csv'):
                found=False
                print(file)
                with open(os.path.join(data_path, file), mode='r') as csv_file:
                    reader = csv.reader(csv_file)
                    for line, row in enumerate(reader):
                        if row and row[0].strip() == 'Height':
                            found = True
                            break
                    if not found:
                        line = 15
                    with open(os.path.join(data_path, file), 'rb') as f:
                        sample_bytes = f.read(4096)
                    sample = ''.join(chr(b) for b in sample_bytes if 32 <= b <= 126 or b in (9, 10, 13))
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
                        delimiter = dialect.delimiter
                    except Exception:
                        comma_count = sample.count(',')
                        semi_count = sample.count(';')
                        tab_count = sample.count('\t')
                        pipe_count = sample.count('|')

                        delimiter = max(
                            {',': comma_count, ';': semi_count, '\t': tab_count, '|': pipe_count},
                            key=lambda k: {',': comma_count, ';': semi_count, '\t': tab_count, '|': pipe_count}[k]
                        )
                    df = pd.read_csv(
                        os.path.join(data_path, file),
                        skiprows=line + 1,
                        header=None,
                        dtype=str,
                        delimiter=delimiter,
                        encoding="latin-1"
                    )

                    df['Merged'] = df.astype(str).agg(' '.join, axis=1)

                    series_data = df['Merged']
                    data_list = []
                    for line in series_data:
                        line = line.replace('"', "").replace("'", "")  # Remove quotes
                        line = line.split(',')  # Split by comma

                        numbers = []
                        for item in line:
                            for val in item.split():
                                if val.lower() == 'nan' or val == '':
                                    numbers.append(np.nan)
                                else:
                                    numbers.append(float(val))

                        row = np.array(numbers, dtype=float)
                        data_list.append(row)
                    data_mat = np.vstack(data_list)
                    plt.figure()
                    plt.imshow(data_mat)
                    plt.title(file.replace('.csv', ''))
                    plt.savefig(os.path.join(data_path, file.replace('.csv', '') + '.png'))
                    plt.close()
                    np.save(os.path.join(data_path, file.replace('.csv', '')), data_mat)
        except:
            pass

# path=''
# if __name__ == '__main__':
#     main(path)
