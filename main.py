import csv
import numpy as np
from sklearn import datasets, linear_model

if __name__ == '__main__':

    h1_directory = 'resources/H1'
    h1_files = ['a40439n.csv', 'a40493n.csv', 'a40764n.csv', 'a40834n.csv', 'a40928n.csv', 'a41200n.csv', 'a41447n.csv',
                'a41770n.csv', 'a41882n.csv', 'a41925n.csv', 'a42277n.csv', 'a42397n.csv']
    for file in h1_files:
        csv_file = open(h1_directory + '/' + file, 'rt')
        reader = csv.reader(csv_file, delimiter=',', quotechar='\'')
        next(reader)
        data = []
        for row in reader:
            data.append(row)
        data = np.array(data)
        data_X = data[:, 4]
        regr = linear_model.LinearRegression()
        regr.fit(data_X, data_X)