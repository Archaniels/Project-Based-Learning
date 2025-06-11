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

# memproses data (load file, print header, etc.)
def preprocessing():
    filepath = r"C:\Users\sxpix\Documents\~~~ CODES\Visual Studio 2022\Project-Based-Learning\ai_dev_productivity.csv"

    # membaca file dataset .CSV
    df = pd.read_csv(filepath)

    # menampilkan informasi kolom pada dataset
    print("=================================================================================================================================")
    print("\nInformasi kolom pada dataset:\n")
    print(df.columns.tolist())
    print("\n=================================================================================================================================")

    # menampilkan 5 baris pertama dari dataset
    pd.set_option('display.max_columns', None)
    print("\n5 baris pertama dari dataset:\n")
    print(df.head())
    print("\n=================================================================================================================================")


    # menampilkan deskripsi statistik dari dataset
    print("\nDeskripsi statistik dari dataset:")
    print(df.describe())
    print("\n=================================================================================================================================")

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

# menghitung entropy shannon
def entropy(y):
    label_counts = Counter(y)
    total = len(y)
    ent = 0.0
    for count in label_counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

# menghitung information gain
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

# mengembalikan nilai Information Gain terbaik
def best_information_gain(X_train, X_test, y_train, y_test):
    # cek Information Gain untuk fitur pertama (hours_coding) dengan threshold Q1 (25%)
    sample = X_train[:, 0]  # hours_coding
    ig_hours = information_gain(sample, y_train, threshold=np.percentile(sample, 25))
    print("\nInformation Gain untuk fitur pertama (hours_coding) (threshold=Q1 (Percentile dari 25%)):", ig_hours)

    # cek Information Gain untuk fitur kedua (coffee_intake_mg) dengan threshold percentile Q2 (50%)
    sample = X_train[:, 1]  # coffee_intake_mg
    ig_coffee = information_gain(sample, y_train, threshold=np.percentile(sample, 50))
    print("Information Gain untuk fitur kedua (coffee_intake_mg) (threshold=Q2 (Percentile dari 50% / Median)):", ig_coffee)
    
    # cek Information Gain untuk fitur ketiga (distractions) dengan threshold percentile Q3 (75%)
    sample = X_train[:, 2]  # distractions
    ig_distractions = information_gain(sample, y_train, threshold=np.percentile(sample, 75))
    print("Information Gain untuk fitur ketiga (distractions) (threshold=Q3 (Percentile dari 75%)):", ig_distractions)

    # cek Information Gain untuk fitur keempat (sleep_hours) dengan threshold percentile Q1 (25%)
    sample = X_train[:, 3]  # sleep_hours
    ig_sleep = information_gain(sample, y_train, threshold=np.percentile(sample, 25))
    print("Information Gain untuk fitur keempat (sleep_hours) (threshold=Q1 (Percentile dari 25%)):", ig_sleep)

    # cek Information Gain untuk fitur kelima (commits) dengan threshold percentile Q2 (50%)
    sample = X_train[:, 4]  # commits
    ig_commits = information_gain(sample, y_train, threshold=np.percentile(sample, 50))
    print("Information Gain untuk fitur kelima (commits) (threshold=Q2 (Percentile dari 50% / Median)):", ig_commits)

    # cek Information Gain untuk fitur keenam (bugs_reported) dengan threshold percentile Q3 (75%)
    sample = X_train[:, 5]  # bugs_reported
    ig_bugs = information_gain(sample, y_train, threshold=np.percentile(sample, 75))
    print("Information Gain untuk fitur keenam (bugs_reported) (threshold=Q3 (Percentile dari 75%)):", ig_bugs)

    # cek Information Gain untuk fitur ketujuh (ai_usage_hours) dengan threshold percentile Q1 (25%)
    sample = X_train[:, 6]  # ai_usage_hours
    ig_usage = information_gain(sample, y_train, threshold=np.percentile(sample, 25))
    print("Information Gain untuk fitur ketujuh (ai_usage_hours) (threshold=Q1 (Percentile dari 25%)):", ig_usage)

    # cek Information Gain untuk fitur kedelapan (cognitive_load) dengan threshold percentile Q2 (50%)
    sample = X_train[:, 7]  # cognitive_load
    ig_cognitive = information_gain(sample, y_train, threshold=np.percentile(sample, 50))
    print("Information Gain untuk fitur kedelapan (cognitive_load) (threshold=Q2 (Percentile dari 50% / Median)):", ig_cognitive)

    return ig_hours, ig_coffee, ig_distractions, ig_sleep, ig_commits, ig_bugs, ig_usage, ig_cognitive

# def struktur_tree():

def main():
    # panggil preprocessing dan ambil data
    X, y = preprocessing()

    # bagi data
    X_train, X_test, y_train, y_test = pemisahan_testing_training(X, y)

    # tampilkan ukuran data
    print("\nTotal data dalam dataset .csv:", len(X))
    print("Jumlah data training:", len(X_train))
    print("Jumlah data testing:", len(X_test))

    ig_hours, ig_coffee, ig_distractions, ig_sleep, ig_commits, ig_bugs, ig_usage, ig_cognitive = best_information_gain(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()