#  https://www.youtube.com/watch?v=ywBSlN086kA

import re
import string
import pandas as pd
from functools import reduce
from math import log


# Corpus
# pain fever weakness
corpus = """
pain pain fever thrombosis
pain vomiting edema
batata fever weakness
""".split('\n')[1:-1]

# corpus = ["pain pain pain pain pain pain pain pain pain pain pain pain fever thrombosis"]


# Documentos
l_A = corpus[0].lower().split()
l_B = corpus[1].lower().split()
l_C = corpus[2].lower().split()
# l_D = corpus[3].lower().split()
print(l_A)
print(l_B)
# print(l_C)
# print(l_D)


# Dicionário de termos existentes
# word_set = set(l_A).union(set(l_B)).union(set(l_C)).union(set(l_D))
word_set = set(l_A).union(set(l_B)).union(set(l_C))
print(word_set)


# Matriz
word_dict_A = dict.fromkeys(word_set, 0)
word_dict_B = dict.fromkeys(word_set, 0)
word_dict_C = dict.fromkeys(word_set, 0)
# word_dict_D = dict.fromkeys(word_set, 0)

for word in l_A:
    word_dict_A[word] += 1

for word in l_B:
    word_dict_B[word] += 1

for word in l_C:
    word_dict_C[word] += 1

# for word in l_D:
#     word_dict_D[word] += 1
# df = pd.DataFrame([word_dict_A, word_dict_B, word_dict_C, word_dict_D])
df = pd.DataFrame([word_dict_A, word_dict_B, word_dict_C])
print('\n',df)


# TF
def compute_tf(word_dict, l):
    tf = {}
    sum_nk = len(l)
    for word, count in word_dict.items():
        tf[word] = count/sum_nk
    return tf

tf_A = compute_tf(word_dict_A, l_A)
tf_B = compute_tf(word_dict_B, l_B)
tf_C = compute_tf(word_dict_C, l_C)
# tf_D = compute_tf(word_dict_D, l_D)

# df = pd.DataFrame([tf_A, tf_B, tf_C, tf_D])
df = pd.DataFrame([tf_A, tf_B, tf_C])
print('\n',df)


# IDF
def compute_idf(strings_list):
    print(strings_list)
    n = len(strings_list)
    print(f'n é {n}')
    idf = dict.fromkeys(strings_list[0].keys(), 0)
    for l in strings_list:
        for word, count in l.items():
            if count > 0:
                idf[word] += 1
    
    print(f'idf é {idf}')
    for word, v in idf.items():
        print(f'v -> {v}')
        idf[word] = log(n / float(v))
        print(f'idf[word] --> {idf[word]}')
    return idf

# idf = compute_idf([word_dict_A, word_dict_B, word_dict_C, word_dict_D])
idf = compute_idf([word_dict_A, word_dict_B,  word_dict_C])
df = pd.DataFrame([idf])
print('\n',df)