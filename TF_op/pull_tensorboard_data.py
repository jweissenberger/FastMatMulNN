import os
import gzip
import json
import pandas as pd


def find_algorithm(dataset, algo, size, time):
    for index in range(len(dataset)):
        if dataset[index]['Algorithm'] == algo and int(dataset[index]['Matrix_Size']) == int(size):
            dataset[index]['Total_time'] = time

    return dataset


if __name__ == '__main__':

    algorithm_names = ['dgemm', 'bini322', 'schonhage333', 'smirnov224', 'smirnov225', 'smirnov272', 'smirnov323',
                       'smirnov333', 'smirnov334', 'smirnov442', 'smirnov444', 'smirnov552', 'smirnov555'
                       ]
    matrix_sizes = [512, 1024, 2048, 4096, 8192]

    tensorboard_folders_prefix = 'FINAL_12_thread_'

    total_time_log_file_name = "12ThreadTimes.log"

    output_file_name = '12_threads_time.csv'

    output = []

    # parse the tensorboard data
    for mat_size in matrix_sizes:
        for algo in algorithm_names:

            folder = f"{tensorboard_folders_prefix}{algo}{mat_size}"
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

    # file = open(total_time_log_file_name)
    # algorithm = ''
    # total_time = ''
    # mat_size = ''
    # for line in file.readlines():
    #     if "Algorithm:" in line:
    #         algorithm = line.split('Algorithm: ')[-1].replace('\n', '').replace(' ', '')
    #
    #     if "Total time:" in line:
    #         total_time = line.split('Total time: ')[-1].replace('\n', '').replace(' ', '')
    #
    #     if "Matrix size:" in line:
    #         mat_size = line.split('Matrix size: ')[-1].replace('\n', '').replace(' ', '')
    #         output = find_algorithm(dataset=output, algo=algorithm, size=mat_size, time=total_time)
    #         # clear them all out again so that if one is missing its filled with null instead of
    #         algorithm = ''
    #         total_time = ''
    #         mat_size = ''
    #
    # file.close()

    output = pd.DataFrame(output)

    output.to_csv(output_file_name, index=False)
