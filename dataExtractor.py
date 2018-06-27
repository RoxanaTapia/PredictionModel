import numpy as np
from wfdb import rdsamp


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def write(patient, filename):
    with open('resources/' + filename, 'a') as f:
        row = ""
        i=1
        for abp_mean in patient:
            row = row + str(abp_mean)

            if i < len(patient):
                row = row + " "
                i = i + 1
            elif i == len(patient):
                row = row + '\r'
                f.write(row)
                f.close()
                return


def get_abp_means(input_file):
    signals, fields = rdsamp(input_file, channels=[2])
    signals.tolist()

    if fields['sig_name'][0] != 'ABP':
        print("Choose a different signal!")
        fields = rdsamp(input_file, channels='all')[1]
        if 'ABP' in fields['sig_name']:
            print("Corrected.")
            index = fields['sig_name'].index('ABP')
            signals = rdsamp(input_file, channels=[index])[0]
        else:
            print('ABP signal not found, skipping.')

    # Todo: Uncomment for c signals (after t0)
    # signals = signals[:450000]

    abps = list()
    print("Extracting ABP values...")
    for abp in signals:
        abps.append(abp[0])
    print("Extraction finished.")

    print("Generating ABPMean by chunks...")
    abp_means = list()
    for chunk in chunks(abps, 7500):
        mean = round(np.mean(chunk), 1)
        abp_means.append(mean)
    print("Finished.")
    return abp_means


if __name__ == '__main__':

    # TODO Uncomment for processing data after t0
    # print("TEST SET A\n")
    # patients = list()
    # for i in range(1, 11):
    #     digit = str(i).zfill(2)
    #     path = "resources/c_signals/test-set-a/1"+digit+"c/1"+digit+"c"
    #     # path = "resources/c_signals/test-set-a/103c/103c"
    #     print("---------------------------------------------")
    #     print("Processing data for: " + path)
    #     try:
    #         means = get_abp_means(path)
    #         patients.append(means)
    #     except Exception as e:
    #         print("xxxxxxxxxxxx   Something went wrong, skipping data... " + str(e))
    #
    # print("---------------------------------------------\n")
    # print("Writing data...")
    # for p in patients:
    #     write(p, "tests_a_matrix/data_a_after_t0.txt")
    # print("Done.")
    #
    # # Process test set b
    # print("\nTEST SET B\n")
    #
    # patients = list()
    # for i in range(1, 41):
    #     digit = str(i).zfill(2)
    #     path = "resources/c_signals/test-set-b/2" + digit + "c/2" + digit + "c"
    #     # path = "resources/test_set_b/222b/222b"
    #     print("---------------------------------------------")
    #     print("Processing data for: " + path)
    #     try:
    #         means = get_abp_means(path)
    #         patients.append(means)
    #     except Exception as e:
    #         print("xxxxxxxxxxxx   Something went wrong, skipping data... " + str(e))
    #
    # print("---------------------------------------------\n")
    # print("Writing data...")
    # for p in patients:
    #     write(p, "tests_b_matrix/data_b_after_t0.txt")
    # print("Done.")

    # TODO Uncomment for processing data before t0

    # print("TEST SET A\n")
    # patients = list()
    # for i in range(1, 11):
    #     digit = str(i).zfill(2)
    #     path = "resources/test_set_a/1"+digit+"b/1"+digit+"b"
    #     print("---------------------------------------------")
    #     print("Processing data for: " + path)
    #     try:
    #         means = get_abp_means(path)
    #         patients.append(means)
    #     except Exception as e:
    #         print("xxxxxxxxxxxx   Something went wrong, skipping data... " + str(e))
    #
    # print("---------------------------------------------\n")
    # print("Writing data...")
    # for p in patients:
    #     write(p, "tests_a_matrix/data_a.txt")
    # print("Done.")

    # Process test set b
    # print("\nTEST SET B\n")
    #
    # patients = list()
    # for i in range(1, 41):
    #     digit = str(i).zfill(2)
    #     path = "resources/test_set_b/2" + digit + "b/2" + digit + "b"
    #     print("---------------------------------------------")
    #     print("Processing data for: " + path)
    #     try:
    #         means = get_abp_means(path)
    #         patients.append(means)
    #     except Exception as e:
    #         print("xxxxxxxxxxxx   Something went wrong, skipping data... " + str(e))
    #
    # print("---------------------------------------------\n")
    # print("Writing data...")
    # for p in patients:
    #     write(p, "tests_b_matrix/data_b.txt")
    print("Done.")
