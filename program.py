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
## 1. Diberikan sebuah dataset berformat .CSV yang berisi data-data yang berkaitan dengan data housing di King County, USA. Dataset ini berisi informasi mengenai harga jual rumah beserta fitur-fiturnya, seperti jumlah kamar tidur, ukuran rumah, lokasi, dan sebagainya.
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

filepath = "C:\Users\sxpix\Documents\~~~ CODES\Visual Studio 2022\Project-Based-Learning\Project-Based-Learning\kc_house_data.csv"

def load_dataset(filepath):
    