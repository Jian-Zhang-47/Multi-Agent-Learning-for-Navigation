import config as cfg
import numpy as np


def dict_to_arr(dict_var):
    row = col = 2 * cfg.M + 1
    array_3d = np.full((cfg.N, row, col), '  ', dtype=object)

    for a_id, sub_dict in dict_var.items():
        for (i, j), value in sub_dict.items():
            array_3d[a_id - 1][i, j] = value

    array_3d_num = np.vectorize(replace_element)(array_3d).astype(float)

    return array_3d_num


def replace_element(elem):
    return cfg.replacement_dict.get(elem)


def return_one_hot_vector(value):
    one_hot_vector = [0 for _ in range(4)]
    one_hot_vector[value] = 1
    return one_hot_vector
