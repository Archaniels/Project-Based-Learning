## =========== SOURCE CODE PROGRAM ===========
## Carilah dataset gratis, source bebas (contoh: Kaggle)
## Domain AI tidak dibatasi, bisa computer vision, NLP, atau data science.
## Metode/algoritma dibebaskan, pilihlah yang cocok dan bisa digunakan untuk dataset yang kalian gunakan.
## Pilih salah satu algoritma yang anda pelajari pada teknik learning:
    ## Regresi Linear, 
    ## Decision Tree,
    ## Naive Bayes,
    ## Nearest Neighbour,
    ## DLL.
## Sisa ketentuan ada di PDF berikut

## =========== PENJELASAN TUGAS ===========
## 1. Diberikan sebuah dataset berformat .CSV yang berisi data-data yang berkaitan dengan data.
## 2. Tugas adalah untuk menerapkan domain AI yang digunakan, serta metode/algoritma yang telah dipelajari di Learning.
## 3. Domain AI yang digunakan adalah Machine Learning, dengan subdomain adalah regression.
## 4. Metode algorithma yang digunakan adalah Regresi Linear, karena cocok untuk memprediksi harga rumah, berdasarkan fitur numerik dan kategori.

##  =========== PEMBAGIAN TUGAS ===========
##  Sistem 50/50
    ##  Daniyal
    ##  Riziq

## ================================================================================================================== 
##                                                 PROGRAM
## ================================================================================================================== 

import pandas as pd ## untuk memproses file dataset .CSV
import numpy as np ## untuk perhitungan
from collections import Counter
import math
import matplotlib.pyplot as plt ## untuk visualisasi data mungkin?


def preprocessing():
    filepath = r"C:\Users\sxpix\Documents\~~~ CODES\Visual Studio 2022\Project-Based-Learning\ai_dev_productivity.csv"

    # membaca file dataset .CSV
    df = pd.read_csv(filepath)

    # Memisahkan fitur dan label
    X = df.drop(columns=['task_success'])  # Semua kolom kecuali target
    y = df['task_success']  # Kolom target

    # Mengubah ke array numpy agar lebih mudah diproses secara manual
    X = X.to_numpy()
    y = y.to_numpy()

    return X, y

# Membagi data ke training dan testing set (80% train & 20% test / 400 data untuk training & 100 data untuk testing)
def pemisahan_testing_training(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def entropy(y):
    label_counts = Counter(y)
    total = len(y)
    ent = 0.0
    for count in label_counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def information_gain(Kolom_X, y, threshold):
    left_mask = Kolom_X <= threshold
    right_mask = Kolom_X > threshold

    entropy_parent = entropy(y)
    n = len(y)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    if n_left == 0 or n_right == 0:
        return 0

    entropy_kiri = entropy(y[left_mask])
    entropy_kanan = entropy(y[right_mask])

    entropy_final = (n_left / n) * entropy_kiri + (n_right / n) * entropy_kanan
    return entropy_parent - entropy_final

# def struktur_tree():

def main():
    # panggil preprocessing dan ambil data
    X, y = preprocessing()

    # bagi data
    X_train, X_test, y_train, y_test = pemisahan_testing_training(X, y)

    # tampilkan ukuran data
    print("Jumlah data training:", len(X_train))
    print("Jumlah data testing:", len(X_test))

    # cek Information Gain untuk fitur pertama (jam coding) dengan threshold 5.0
    sample = X_train[:, 0]  # hours_coding
    test_information_gain = information_gain(sample, y_train, threshold=5.0)
    print("Information Gain untuk fitur pertama (threshold=5.0):", test_information_gain)


if __name__ == "__main__":
    main()