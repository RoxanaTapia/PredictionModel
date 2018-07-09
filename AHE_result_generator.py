import numpy as np
import os


def has_abp(abps):
    limit = 30
    for x in range(30):
        chunk = abps[x:limit]
        ahe = list()
        for abp in chunk:
            if (abp <= 60) and (abp >= 10):
                ahe.append(abp)
        if len(ahe) >= 27:
            return True
        limit = limit + 1
    return False


def get_comparison(original, predicted):
    correct = 0
    wrong = 0
    accurate_ahe = 0
    wrong_predicted = 0
    total_ahe = 0

    for i in range(len(original)):
        if original[i] and predicted[i]:
            if original[i] == predicted[i]:
                correct = correct + 1
                if original[i] == 'T':
                    total_ahe = total_ahe + 1
                    accurate_ahe = accurate_ahe + 1
            else:
                wrong = wrong + 1
                if original[i] == 'T':
                    total_ahe = total_ahe + 1
                    wrong_predicted = wrong_predicted + 1

    print("Correct {correct}, wrong {wrong}".format(correct=correct, wrong=wrong))
    print("Accurate AHE {accurate_ahe}".format(accurate_ahe=accurate_ahe))
    print("Wrong AHE predictions {wrong_predicted}".format(wrong_predicted=wrong_predicted))
    print("Accuracy: {accuracy}%".format(accuracy=round((accurate_ahe*100)/total_ahe, 2)))


if __name__ == '__main__':
    c1 = np.loadtxt('resources/after_t0/c1_matrix.txt').tolist()
    c2 = np.loadtxt('resources/after_t0/c2_matrix.txt').tolist()
    h1 = np.loadtxt('resources/after_t0/h1_matrix.txt').tolist()
    h2 = np.loadtxt('resources/after_t0/h2_matrix.txt').tolist()

    a = np.loadtxt('resources/tests_a_matrix/data_a_after_t0.txt').tolist()
    b = np.loadtxt('resources/tests_b_matrix/data_b_after_t0.txt').tolist()

    predicted_a = np.loadtxt('Results/prediction_testA.txt').tolist()
    predicted_b = np.loadtxt('Results/prediction_testB.txt').tolist()

    directory = {
        # 'resources/ahe_results/training_set/c1/': ('results_c1.txt', c1),
        # 'resources/ahe_results/training_set/c2/': ('results_c2.txt', c2),
        # 'resources/ahe_results/training_set/h1/': ('results_h1.txt', h1),
        # 'resources/ahe_results/training_set/h2/': ('results_h2.txt', h2),
        # 'resources/ahe_results/test_set_a/': ('results_a.txt', a),
        # 'resources/ahe_results/test_set_b/': ('results_b.txt', b),
        'resources/ahe_results/predicted/test_set_a/': ('results_a.txt', predicted_a),
        'resources/ahe_results/predicted/test_set_b/': ('results_b.txt', predicted_b),
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

    original_a = open('resources/ahe_results/test_set_a/results_a.txt').read().split("\n")
    predicted_a = open('resources/ahe_results/predicted/test_set_a/results_a.txt', "r").read().split("\n")

    print("TEST A: {total} results".format(total=len(original_a)-1))
    get_comparison(original_a, predicted_a)
    print()

    original_b = open('resources/ahe_results/test_set_b/results_b.txt').read().split("\n")
    predicted_b = open('resources/ahe_results/predicted/test_set_b/results_b.txt', "r").read().split("\n")

    print("TEST B: {total} results".format(total=len(original_b)-1))
    get_comparison(original_b, predicted_b)

