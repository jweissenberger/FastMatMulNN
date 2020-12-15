import os
import gzip
import json


if __name__ == '__main__':

    algorithm_names = ['bini322', 'schonhage333', 'smirnov224', 'smirnov225', 'smirnov272', 'smirnov323',
                       'smirnov333', 'smirnov334', 'smirnov442', 'smirnov444', 'smirnov552', 'smirnov555'
                       ]
    matrix_sizes = [512, 1024, 2048, 4096, 8192]

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
            print('\n*************************************************')
            print(f"Algorithm: {algo}, Size: {mat_size}")
            print(f"linear2_mat_mul_dur: {linear2_mat_mul_dur}")
            print(f"Total FastMM time: {fastmm_time}")
            print('*************************************************')
