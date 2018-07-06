import numpy as np
import os


def has_abp(abps):
    limit = 30
    for x in range(30):
        chunk = abps[x:limit]
        ahe = list()
        for abp in chunk:
            if (abp <= 67) and (abp >= 10):
                ahe.append(abp)
        if len(ahe) >= 24:
            return True
        limit = limit + 1
    return False

if __name__ == '__main__':
    c1 = np.loadtxt('resources/after_t0/c1_matrix.txt').tolist()
    c2 = np.loadtxt('resources/after_t0/c2_matrix.txt').tolist()
    h1 = np.loadtxt('resources/after_t0/h1_matrix.txt').tolist()
    h2 = np.loadtxt('resources/after_t0/h2_matrix.txt').tolist()

    a = np.loadtxt('resources/tests_a_matrix/data_a_after_t0.txt').tolist()
    b = np.loadtxt('resources/tests_b_matrix/data_b_after_t0.txt').tolist()

    directory = {
        'resources/ahe_results/training_set/c1/': ('results_c1.txt', c1),
        'resources/ahe_results/training_set/c2/': ('results_c2.txt', c2),
        'resources/ahe_results/training_set/h1/': ('results_h1.txt', h1),
        'resources/ahe_results/training_set/h2/': ('results_h2.txt', h2),
        'resources/ahe_results/test_set_a/': ('results_a.txt', a),
        'resources/ahe_results/test_set_b/': ('results_b.txt', b),
    }

    for path, content in directory.items():
        if not os.path.exists(path):
            os.makedirs(path)

        filename, patients = content
        file = open(path + filename, 'w+')
        for patient in patients:
            if has_abp(patient):
                file.write('T\r')
            else:
                file.write('F\r')

        file.close()
