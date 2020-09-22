from opgen import main

if __name__ == "__main__":

    in_files = ['bini_mat_mul.cc', 'dgemm_mat_mul.cc', 'smirnov224_mat_mul.cc', 'smirnov272_mat_mul.cc',
                'smirnov333_mat_mul.cc', 'smirnov442_mat_mul.cc ', 'smirnov552_mat_mul.cc ', 'strassen_mat_mul.cc',
                'schonhage_mat_mul.cc', 'smirnov225_mat_mul.cc', 'smirnov323_mat_mul.cc', 'smirnov334_mat_mul.cc',
                'smirnov444_mat_mul.cc', 'smirnov555_mat_mul.cc']

    for i in in_files:
        main(in_file=i, out_file=f'src/{i}')