import csv
import os
from collections import OrderedDict
from datetime import datetime


class Patient:
    def __init__(self, id, t0):
        self.id = id
        self.t0 = t0
        self.data = OrderedDict()

    def add_record(self, time, abp_mean):
        self.data[time] = abp_mean


def filter_values(row):
    try:
        # abp_mean = float(row[4])
        abp_mean = float(row[1])
    except ValueError:
        abp_mean = 0

    raw_time = str(row[0])[2:10] + " " + str(row[0])[11:21]
    # time = datetime.strptime(raw_time, '%H:%M:%S %d/%m/%Y')
    time = datetime.strptime(raw_time, '%H:%M:%S %Y-%m-%d')

    return time, abp_mean


def get_t0(id):
    content = open('resources/T0.txt', mode='r', encoding="utf-8-sig").read().split('\n')
    for c in content:
        extracted_id, time = c.split(',')
        if extracted_id != id[:8]:
            continue
        time = datetime.strptime(time, '%d/%m/%Y %H:%M')
        return time


def read(category, id):
    """
    Reads .csv file for a patient.
    :param category: path to group
    :param id: patient id
    :return: Patient
    :rtype: Patient
    """
    f = open(category + "/" + id, 'rt')
    reader = list(csv.reader(f))
    t0 = get_t0(id)
    patient = Patient(id=id[:8], t0=t0)

    for row in reader:
        if not row:
            continue
        time, abp_mean = filter_values(row)
        # uncomment for trainig set after t0
        # if time >= t0 and len(patient.data) < 60:  # 60 values each hour
            # uncoment for test set after t0
        if len(patient.data) < 60:  # 60 values each hour
            patient.add_record(time=time, abp_mean=abp_mean)

    f.close()
    return patient


def load_data(category):
    patients = list()
    resources = os.listdir(category)
    resources.sort()
    for r in resources:
        p = read(category, r)
        patients.append(p)
    return patients


def write_matrix(patients, filename):
    with open('resources/' + filename, 'w+') as f:
        for p in patients:
            row = ""
            for time, abp_mean in p.data.items():
                row = row + str(abp_mean)
                if list(p.data.keys()).index(time) + 1 < len(p.data.keys()):
                    row = row + " "
                elif patients.index(p) + 1 < len(patients):
                    row = row + '\r'
            f.write(row)
        f.close()


if __name__ == '__main__':
    # h1 = load_data('resources/H1')
    # write_matrix(h1, 'after_t0/h1_matrix.txt')
    #
    # h2 = load_data('resources/H2')
    # write_matrix(h2, 'after_t0/h2_matrix.txt')
    #
    # c1 = load_data('resources/C1')
    # write_matrix(c1, 'after_t0/c1_matrix.txt')
    #
    # c2 = load_data('resources/C2')
    # write_matrix(c2, 'after_t0/c2_matrix.txt')

    # a = load_data('resources/A')
    # write_matrix(a, 'tests_a_matrix/data_a.txt')
    #
    a = load_data('resources/csv_after_t0_tests/A')
    write_matrix(a, 'tests_a_matrix/data_a_after_t0.txt')
    #
    # b = load_data('resources/B')
    # write_matrix(b, 'tests_b_matrix/data_b.txt')

    b = load_data('resources/csv_after_t0_tests/B')
    write_matrix(b, 'tests_b_matrix/data_b_after_t0.txt')
    pass
