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
    jumlah_test = int(len(X) * test_size)
    
    test_index = indices[:jumlah_test]
    training_index = indices[jumlah_test:]
    
    X_train, X_test = X[training_index], X[test_index]
    y_train, y_test = y[training_index], y[test_index]
    
    return X_train, X_test, y_train, y_test

def entropy():

def information_gain():

def struktur_tree():


def main():
    # Panggil preprocessing dan ambil data
    X, y = preprocessing()

    # Bagi data
    X_train, X_test, y_train, y_test = pemisahan_testing_training(X, y)

    # Tampilkan ukuran data
    print("Jumlah data training:", len(X_train))
    print("Jumlah data testing:", len(X_test))

if __name__ == "__main__":
    main()