import csv
import os
os.chdir('C:\libsvm-3.24\python')
from svmutil import *
from svm import *

import numpy as np

if __name__ == '__main__':
    x_test = list()
    x_train = list()
    y_test = list()
    y_train = list()
    with open(r'C:\Users\KuanWenChen\Desktop\NCTU_class\碩一下\ML\HW5\X_test.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        for row in rows:
            x_test.append(row)
    with open(r'C:\Users\KuanWenChen\Desktop\NCTU_class\碩一下\ML\HW5\Y_test.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        for row in rows:
            y_test.append(row)
    with open(r'C:\Users\KuanWenChen\Desktop\NCTU_class\碩一下\ML\HW5\X_train.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        for row in rows:
            x_train.append(row)
    with open(r'C:\Users\KuanWenChen\Desktop\NCTU_class\碩一下\ML\HW5\Y_train.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        for row in rows:
            y_train.append(row)
    x_train = np.array(x_train).astype(float)
    y_train = np.array(y_train).astype(float).reshape(5000)
    x_test = np.array(x_test).astype(float)
    y_test = np.array(y_test).astype(float).reshape(2500)

    #test = svm_train(y_train, x_train, '-v 2')
    """
    best_linear = 0.0
    best_poly = 0.0
    best_rbf = 0.0
    for c in range(1,4):
        for g in range(1,4):
            linear = svm_train(y_train, x_train, f'-t 0 -q -c {c} -g {g} -v 2')
            #predict_linear = svm_predict(y_test, x_test, linear)
            poly = svm_train(y_train, x_train, f'-t 1 -q -c {c} -g {g} -v 2')
            #predict_poly = svm_predict(y_test, x_test, poly)
            rbf = svm_train(y_train, x_train, f'-t 2 -q -c {c} -g {g} -v 2')
            #predict_rbf = svm_predict(y_test, x_test, rbf)
            if best_linear < linear:
                best_linear = linear
                l_c = c
                l_g = g
            if best_poly < poly:
                best_poly = poly
                p_c = c
                p_g = g
            if best_rbf < rbf:
                best_rbf = rbf
                r_c = c
                r_g = g
    print(f"best linear score: {best_linear} with parameters c: {l_c} and gamma: {l_g}")
    print(f"best poly score: {best_poly} with parameters c: {p_c} and gamma: {p_g}")
    print(f"best rbf score: {best_rbf} with parameters c: {r_c} and gamma: {r_g}")

    """
    linear = svm_train(y_train, x_train, '-t 0 -q')
    predict_linear = svm_predict(y_test, x_test, linear)
    poly = svm_train(y_train, x_train, '-t 1 -q')
    predict_poly = svm_predict(y_test, x_test, poly)
    rbf = svm_train(y_train, x_train, '-t 2 -q')
    predict_rbf = svm_predict(y_test, x_test, rbf)
    
    
            
