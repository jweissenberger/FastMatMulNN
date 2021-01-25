import pandas as pd


def format_excel_times(nums):
    algos = [
        'dgemm',
        'bini322',
        'schonhage333',
        'smirnov224',
        'smirnov225',
        'smirnov272',
        'smirnov323',
        'smirnov333',
        'smirnov334',
        'smirnov442',
        'smirnov444',
        'smirnov552',
        'smirnov555',
    ]
    nums = nums.split('\n')
    for index in range(len(nums)):
        val = nums[index].replace(',', '').replace(' ', '')

        algo = algos[index]
        print(f"'{algo}': {val},")

    return

times = {
            512:
                {
                    'Layer_2_matmul_time':
                        {
                            'dgemm': 5204562,
                            'bini322': 6254049,
                            'schonhage333': 6552512,
                            'smirnov224': 5707803,
                            'smirnov225': 6086669,
                            'smirnov272': 7415098,
                            'smirnov323': 6377260,
                            'smirnov333': 6738471,
                            'smirnov334': 6625309,
                            'smirnov442': 5885606,
                            'smirnov444': 6232601,
                            'smirnov552': 7555278,
                            'smirnov555': 8441704,

                        },
                    'Total_fastmm_time':
                        {
                            'dgemm': 56020914,
                            'bini322': 65692988,
                            'schonhage333': 68468198,
                            'smirnov224': 61161703,
                            'smirnov225': 65499683,
                            'smirnov272': 76819010,
                            'smirnov323': 67202115,
                            'smirnov333': 70672302,
                            'smirnov334': 69397971,
                            'smirnov442': 61326542,
                            'smirnov444': 64990174,
                            'smirnov552': 78478468,
                            'smirnov555': 91201722,
                        },
                    'Total_time':
                        {
                            'dgemm': 102602406.6,
                            'bini322': 111722768.7,
                            'schonhage333': 114879526.8,
                            'smirnov224': 106553489.5,
                            'smirnov225': 110828566.8,
                            'smirnov272': 123304992,
                            'smirnov323': 112190509.2,
                            'smirnov333': 117007122.5,
                            'smirnov334': 115663285,
                            'smirnov442': 107402000,
                            'smirnov444': 110152837.3,
                            'smirnov552': 124371581.6,
                            'smirnov555': 137145446.6
                        }


                },

            1024:
                {
                    'Layer_2_matmul_time':
                        {
                            'dgemm': 20147256,
                            'bini322': 20842986,
                            'schonhage333': 34781265,
                            'smirnov224': 19767072,
                            'smirnov225': 19880276,
                            'smirnov272': 22809003,
                            'smirnov323': 20601970,
                            'smirnov333': 20984545,
                            'smirnov334': 20791539,
                            'smirnov442': 18748028,
                            'smirnov444': 19090454,
                            'smirnov552': 22947290,
                            'smirnov555': 25453012,
                        },
                    'Total_fastmm_time':
                        {
                            'dgemm': 217775785,
                            'bini322': 224160507,
                            'schonhage333': 224410526,
                            'smirnov224': 210876247,
                            'smirnov225': 216064609,
                            'smirnov272': 240664819,
                            'smirnov323': 221686370,
                            'smirnov333': 225315899,
                            'smirnov334': 223482383,
                            'smirnov442': 200962503,
                            'smirnov444': 205852847,
                            'smirnov552': 247361518,
                            'smirnov555': 266085409,

                        },
                    'Total_time':
                        {
                            'dgemm': 319788230.5,
                            'bini322': 325341809.9,
                            'schonhage333': 327606607.3,
                            'smirnov224': 313803939,
                            'smirnov225': 319622202.7,
                            'smirnov272': 343806884.3,
                            'smirnov323': 322687583.7,
                            'smirnov333': 326072212.7,
                            'smirnov334': 323887511.6,
                            'smirnov442': 302654372,
                            'smirnov444': 306785166.9,
                            'smirnov552': 350370422.1,
                            'smirnov555': 368539347.6,
                        }

                },

            2048:
                {
                    'Layer_2_matmul_time':
                        {
                            'dgemm': 79417385,
                            'bini322': 80304038,
                            'schonhage333': 80811871,
                            'smirnov224': 72053050,
                            'smirnov225': 73657264,
                            'smirnov272': 77366556,
                            'smirnov323': 79306835,
                            'smirnov333': 78806062.5,
                            'smirnov334': 76760756.47,
                            'smirnov442': 68065455.81,
                            'smirnov444': 67921619.85,
                            'smirnov552': 82180545.84,
                            'smirnov555': 84565908.23,

                        },
                    'Total_fastmm_time':
                        {
                            'dgemm': 883736947,
                            'bini322': 853747520,
                            'schonhage333': 846183046,
                            'smirnov224': 784222532,
                            'smirnov225': 802919095,
                            'smirnov272': 832063038,
                            'smirnov323': 838968474,
                            'smirnov333': 838300233.7,
                            'smirnov334': 832228748.1,
                            'smirnov442': 734705494,
                            'smirnov444': 733117652.6,
                            'smirnov552': 862401851.8,
                            'smirnov555': 889499545.6
                        },
                    'Total_time':
                        {
                            'dgemm': 1104671184,
                            'bini322': 1068520050,
                            'schonhage333': 1060379757,
                            'smirnov224': 1004126161,
                            'smirnov225': 1015068388,
                            'smirnov272': 1042685511,
                            'smirnov323': 1056635358,

                        }
                },

            4096:
                {
                    'Layer_2_matmul_time':
                        {
                            'dgemm': 307914489,
                            'bini322': 275708738,
                            'schonhage333': 269802166,
                            'smirnov224': 277366006,
                            'smirnov225': 267059634,
                            'smirnov272': 313783714,
                            'smirnov323': 263701631,
                            'smirnov333': 261718649.6,
                            'smirnov334': 262071888.1,
                            'smirnov442': 249739562,
                            'smirnov444': 245663772.8,
                            'smirnov552': 265342625.5,
                            'smirnov555': 266322043.6,
                        },
                    'Total_fastmm_time':
                        {
                            'dgemm': 3341700942,
                            'bini322': 2975066618,
                            'schonhage333': 2905083734,
                            'smirnov224': 3010584886,
                            'smirnov225': 2882618104,
                            'smirnov272': 3366404823,
                            'smirnov323': 2853180351,
                            'smirnov333': 2795471929,
                            'smirnov334': 2839735252,
                            'smirnov442': 2702525450,
                            'smirnov444': 2638401407,
                            'smirnov552': 2843872289,
                            'smirnov555': 2869207596,

                        },
                    'Total_time':
                        {
                            'dgemm': 3894756343,
                            'bini322': 3529141896,
                            'schonhage333': 3470828834,
                            'smirnov224': 3567043704,
                            'smirnov225': 3435778431,
                            'smirnov272': 3937315582,
                            'smirnov323': 3416982456,

                        }
                },

            8192:
                {
                    'Layer_2_matmul_time':
                        {
                            'dgemm': 1225724851,
                            'bini322': 1103955691,
                            'schonhage333': 1061473749,
                            'smirnov224': 1058924955,
                            'smirnov225': 1051375948,
                            'smirnov272': 1084024336,
                            'smirnov323': 1047990847,
                            'smirnov333': 1023489775,
                            'smirnov334': 1030084611,
                            'smirnov442': 997300045.8,
                            'smirnov444': 966008431.3,
                            'smirnov552': 1057151452,
                            'smirnov555': 1019922978,

                        },
                    'Total_fastmm_time':
                        {
                            'dgemm': 13351140974,
                            'bini322': 12033484529,
                            'schonhage333': 11527636898,
                            'smirnov224': 11502626638,
                            'smirnov225': 11398341520,
                            'smirnov272': 11846696484,
                            'smirnov323': 11333721021,
                            'smirnov333': 11115702798,
                            'smirnov334': 11213186128,
                            'smirnov442': 10813260638,
                            'smirnov444': 10500098408,
                            'smirnov552': 11435623498,
                            'smirnov555': 11002612840,
                        },
                    'Total_time':
                        {
                            'dgemm': 14607375245,
                            'bini322': 13267347882,
                            'schonhage333': 12737720329,
                            'smirnov224': 12724144511,
                            'smirnov225': 12636742262,
                            'smirnov272': 13090272358,
                            'smirnov323': 12551186070,
                        }
                }


        }




if __name__ == '__main__':

    matrix_sizes = [512, 1024, 2048, 4096, 8192]

    rows = []

    for size in matrix_sizes:

        for algo in times[size]['Layer_2_matmul_time'].keys():

            row = {'Algorithm': algo,
                   'Matrix_Size': size,
                   'Layer_2_matmul_time': times[size]['Layer_2_matmul_time'],
                   'Total_fastmm_time': times[size].get('Total_time')}

            rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv('oneThreadTime.csv', index=False)
