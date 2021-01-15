import os
import gzip
import json
import pandas as pd


if __name__ == '__main__':

    algorithm_names = ['dgemm', 'bini322', 'schonhage333', 'smirnov224', 'smirnov225', 'smirnov272', 'smirnov323',
                       'smirnov333', 'smirnov334', 'smirnov442', 'smirnov444', 'smirnov552', 'smirnov555'
                       ]
    matrix_sizes = [512, 1024, 2048, 4096, 8192]

    output = []

    # parse the tensorboard data
    for mat_size in matrix_sizes:
        for algo in algorithm_names:

            folder = f"all_threads_{algo}{mat_size}"
            experiment_name = os.listdir(f'./{folder}/plugins/profile/')[0]

            path_to_zip_file = f'./{folder}/plugins/profile/{experiment_name}/householder.trace.json.gz'

            f = gzip.open(path_to_zip_file, 'rb')
            file_content = f.read()
            f.close()

            data = json.loads(file_content)

            linear2_mat_mul_dur = 0
            fastmm_time = 0
            for event in data['traceEvents']:
                if not event.get('name'):
                    continue
                if 'my_model/linear_2/FastMatMul:FastMatMul' == event['name']:
                    linear2_mat_mul_dur += event['dur']
                if 'FastMatMul' in event['name'] and 'ReadVariableOp' not in event['name']:
                    fastmm_time += event['dur']

            row = {'Algorithm': algo,
                   'Matrix_Size': mat_size,
                   'Layer_2_matmul_time': linear2_mat_mul_dur,
                   'Total_fastmm_time': fastmm_time}
            output.append(row)

    # TODO: then parse the std out file which contains the overall times for each run
    output = pd.Dataframe(output)

    output.to_csv('output_name.csv', index=False)
