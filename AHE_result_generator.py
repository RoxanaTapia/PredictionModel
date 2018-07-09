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
    if total_ahe != 0:
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

    predicted_h1 = np.loadtxt('Results/pred_H1.txt').tolist()
    predicted_h2 = np.loadtxt('Results/pred_H2.txt').tolist()
    predicted_c1 = np.loadtxt('Results/pred_C1.txt').tolist()
    predicted_c2 = np.loadtxt('Results/pred_C2.txt').tolist()

    predicted_h1_lr = np.loadtxt('Results/pred_h1_LR.txt').tolist()
    predicted_h2_lr = np.loadtxt('Results/pred_h2_LR.txt').tolist()
    predicted_c1_lr = np.loadtxt('Results/pred_c1_LR.txt').tolist()
    predicted_c2_lr = np.loadtxt('Results/pred_c2_LR.txt').tolist()

    predicted_a_lr = np.loadtxt('Results/pred_testA_LR.txt').tolist()
    predicted_b_lr = np.loadtxt('Results/pred_testB_LR.txt').tolist()

    directory = {
        # 'resources/ahe_results/training_set/c1/': ('results_c1.txt', c1),
        # 'resources/ahe_results/training_set/c2/': ('results_c2.txt', c2),
        # 'resources/ahe_results/training_set/h1/': ('results_h1.txt', h1),
        # 'resources/ahe_results/training_set/h2/': ('results_h2.txt', h2),
        # 'resources/ahe_results/test_set_a/': ('results_a.txt', a),
        # 'resources/ahe_results/test_set_b/': ('results_b.txt', b),

        # 'resources/ahe_results/predicted/test_set_a/': ('results_a.txt', predicted_a),
        # 'resources/ahe_results/predicted/test_set_b/': ('results_b.txt', predicted_b),

        # 'resources/ahe_results/predicted/training_set/h1/': ('results_h1.txt', predicted_h1),
        # 'resources/ahe_results/predicted/training_set/h2/': ('results_h2.txt', predicted_h2),
        # 'resources/ahe_results/predicted/training_set/c1/': ('results_c1.txt', predicted_c1),
        # 'resources/ahe_results/predicted/training_set/c2/': ('results_c2.txt', predicted_c2),

        'resources/ahe_results/predicted/LR/training_set/h1/': ('results_h1.txt', predicted_h1_lr),
        'resources/ahe_results/predicted/LR/training_set/h2/': ('results_h2.txt', predicted_h2_lr),
        'resources/ahe_results/predicted/LR/training_set/c1/': ('results_c1.txt', predicted_c1_lr),
        'resources/ahe_results/predicted/LR/training_set/c2/': ('results_c2.txt', predicted_c2_lr),

        'resources/ahe_results/predicted/LR/test_set_a/': ('results_a.txt', predicted_a_lr),
        'resources/ahe_results/predicted/LR/test_set_b/': ('results_b.txt', predicted_b_lr),
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

    # original_a = open('resources/ahe_results/test_set_a/results_a.txt').read().split("\n")
    # predicted_a = open('resources/ahe_results/predicted/test_set_a/results_a.txt', "r").read().split("\n")
    #
    # print("TEST A: {total} results".format(total=len(original_a)-1))
    # get_comparison(original_a, predicted_a)
    # print()
    #
    # original_b = open('resources/ahe_results/test_set_b/results_b.txt').read().split("\n")
    # predicted_b = open('resources/ahe_results/predicted/test_set_b/results_b.txt', "r").read().split("\n")
    #
    # print("TEST B: {total} results".format(total=len(original_b)-1))
    # get_comparison(original_b, predicted_b)
    #
    # print("ARIMA")
    #
    # original_h1 = open('resources/ahe_results/training_set/h1/results_h1.txt').read().split("\n")
    # predicted_h1 = open('resources/ahe_results/predicted/training_set/h1/results_h1.txt', "r").read().split("\n")
    #
    # print("TEST H1: {total} results".format(total=len(original_h1) - 1))
    # get_comparison(original_h1, predicted_h1)
    # print()
    #
    # original_h2 = open('resources/ahe_results/training_set/h2/results_h2.txt').read().split("\n")
    # predicted_h2 = open('resources/ahe_results/predicted/training_set/h2/results_h2.txt', "r").read().split("\n")
    #
    # print("TEST H2: {total} results".format(total=len(original_h2) - 1))
    # get_comparison(original_h2, predicted_h2)
    # print()
    #
    # original_c1 = open('resources/ahe_results/training_set/c1/results_c1.txt').read().split("\n")
    # predicted_c1 = open('resources/ahe_results/predicted/training_set/c1/results_c1.txt', "r").read().split("\n")
    #
    # print("TEST C1: {total} results".format(total=len(original_c1) - 1))
    # get_comparison(original_c1, predicted_c1)
    # print()
    #
    # original_c2 = open('resources/ahe_results/training_set/c2/results_c2.txt').read().split("\n")
    # predicted_c2 = open('resources/ahe_results/predicted/training_set/c2/results_c2.txt', "r").read().split("\n")
    #
    # print("TEST C2: {total} results".format(total=len(original_c2) - 1))
    # get_comparison(original_c2, predicted_c2)
    # print()

    print("LR")

    original_h1_lr = open('resources/ahe_results/training_set/h1/results_h1.txt').read().split("\n")
    predicted_h1_lr = open('resources/ahe_results/predicted/LR/training_set/h1/results_h1.txt', "r").read().split("\n")

    print("TEST H1: {total} results".format(total=len(original_h1_lr) - 1))
    get_comparison(original_h1_lr, predicted_h1_lr)
    print()

    original_h2_lr = open('resources/ahe_results/training_set/h2/results_h2.txt').read().split("\n")
    predicted_h2_lr = open('resources/ahe_results/predicted/LR/training_set/h2/results_h2.txt', "r").read().split("\n")

    print("TEST H2: {total} results".format(total=len(original_h2_lr) - 1))
    get_comparison(original_h2_lr, predicted_h2_lr)
    print()

    original_c1_lr = open('resources/ahe_results/training_set/c1/results_c1.txt').read().split("\n")
    predicted_c1_lr = open('resources/ahe_results/predicted/LR/training_set/c1/results_c1.txt', "r").read().split("\n")

    print("TEST C1: {total} results".format(total=len(original_c1_lr) - 1))
    get_comparison(original_c1_lr, predicted_c1_lr)
    print()

    original_c2_lr = open('resources/ahe_results/training_set/c2/results_c2.txt').read().split("\n")
    predicted_c2_lr = open('resources/ahe_results/predicted/LR/training_set/c2/results_c2.txt', "r").read().split("\n")

    print("TEST C2: {total} results".format(total=len(original_c2_lr) - 1))
    get_comparison(original_c2_lr, predicted_c2_lr)
    print()

    # tests

    original_a_lr = open('resources/ahe_results/test_set_a/results_a.txt').read().split("\n")
    predicted_a_lr = open('resources/ahe_results/predicted/LR/test_set_a/results_a.txt', "r").read().split("\n")

    print("TEST A: {total} results".format(total=len(original_a_lr) - 1))
    get_comparison(original_a_lr, predicted_a_lr)
    print()

    original_b_lr = open('resources/ahe_results/test_set_b/results_b.txt').read().split("\n")
    predicted_b_lr = open('resources/ahe_results/predicted/LR/test_set_b/results_b.txt', "r").read().split("\n")

    print("TEST A: {total} results".format(total=len(original_b_lr) - 1))
    get_comparison(original_b_lr, predicted_b_lr)
    print()
