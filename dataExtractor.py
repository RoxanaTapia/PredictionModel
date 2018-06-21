import sys

from wfdb import rdrecord, rdsamp


def write_data(input_file, output_file):
    signals, fields = rdsamp(input_file, sampfrom=0, sampto='end', channels=[2])
    print(signals)
    print()
    print(fields)


if __name__ == '__main__':
    print(sys.argv)
    write_data(sys.argv[1], 'test_set_a/1.csv')
