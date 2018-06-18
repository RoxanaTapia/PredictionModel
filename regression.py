import itertools
import numpy as np
from sklearn.linear_model import LinearRegression


def do_regression(key, data):
    for i in range(len(data)):
        y = data[i]
        # select the value between 10 and 130
        # TODO replace outliers with e.g. mean values
        cond = np.where(((y > 130) | (y < 10)), -1, y)
        y = np.delete(cond, np.argwhere(cond == -1))

        x = np.arange(len(y))
        model = LinearRegression()
        model.fit(np.reshape(x, [len(x), 1]), np.reshape(y, [len(y), 1]))
        yy = model.predict(np.reshape(x, [len(x), 1]))

        w = model.coef_[0][0]  # parameters of model
        b = model.intercept_[0]  # intercept of model

        g1 = np.where(y > w * x + b, y, -1)  # upper cluster
        g11 = np.delete(g1, np.argwhere(g1 == -1))
        g11_mean = np.mean(g11)

        g2 = np.where(y < w * x + b, y, -1)  # lower cluster
        g21 = np.delete(g2, np.argwhere(g2 == -1))
        g21_mean = np.mean(g21)

        print('Patient', i + 1, 'in {key}: '.format(key=key), '%0.2f' % g11_mean, '%0.2f' % g21_mean, len(g11),
              len(g21))


def check_abp_range(data):
    in_abp_range = list()
    for abp in data:
        if (abp >= 10) and (abp <= 60):
            in_abp_range.append(abp)
    percentage = (len(in_abp_range) * 100) / len(data)

    return percentage >= 90


def get_result(data):
    for i in range(len(data)):
        abps = data[i]
        # first_half = abps[:(len(abps) / 2)]
        # second_half = abps[(len(abps) / 2):]
        # TODO check every 30 min interval
        first_half, second_half = abps[:int(len(abps) / 2)], abps[int(len(abps) / 2):]

        abp_first_half = check_abp_range(first_half)
        abp_second_half = check_abp_range(second_half)

        if abp_first_half or abp_second_half:
            print("Patient {patient_id} abp result: YES".format(patient_id=i + 1))
        else:
            print("Patient {patient_id} abp result: NO".format(patient_id=i + 1))


if __name__ == '__main__':
    c1 = np.loadtxt('resources/c1_matrix.txt')
    c2 = np.loadtxt('resources/c2_matrix.txt')
    h1 = np.loadtxt('resources/h1_matrix.txt')
    h2 = np.loadtxt('resources/h2_matrix.txt')

    data_sets = {"C1": c1, "C2": c2, "H1": h1, "H2": h2}

    for k, v in data_sets.items():
        do_regression(k, v)
        print()

    # load after t0 values
    c1 = np.loadtxt('resources/after_t0/c1_matrix.txt')
    c2 = np.loadtxt('resources/after_t0/c2_matrix.txt')
    h1 = np.loadtxt('resources/after_t0/h1_matrix.txt')
    h2 = np.loadtxt('resources/after_t0/h2_matrix.txt')

    data_sets = {"C1": c1, "C2": c2, "H1": h1, "H2": h2}
    # check AHE
    for k, v in data_sets.items():
        get_result(v)
        print()

    lst = list(itertools.product([0, 1], repeat=3))
