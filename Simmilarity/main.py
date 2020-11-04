import gensim
import glob

# text1 = [ 'O', 'rato', 'roeu', 'a', 'roupa', 'do', 'rei', 'de', 'Roma']
# text2 = ['Quem', 'com', 'ferro', 'fere', 'com', 'ferro', 'sera', 'ferido' ]
# corpora = [ 'O', 'rato', 'roeu', 'a', 'roupa', 'do', 'rei', 'de', 'Roma',  'Quem', 'com', 'ferro', 'fere', 'com', 'ferro', 'sera', 'ferido' ]

file_name = glob.glob('*.txt')
# file_name = glob.glob('my_text.txt')

corpora = []
for file in file_name:
    try:
        with open(file, 'r',encoding='utf-8') as f:
            corpora.append(f.read())
    except:
        pass
# print(f'corpora --> {corpora}')

# Corpus = set of texts;
clean_texts = []
for corpus in corpora:
    clean_texts.append(gensim.utils.simple_preprocess(corpus))
print('clean_texts', clean_texts,'\n', len(clean_texts))

# Sorted elements of corpus
dictionary = gensim.corpora.Dictionary(clean_texts)
# dictionary.filter_extremes(no_below=1, no_above=2)


#  Bag of words (bow) is the term frequency of each word from a corpus
# corpus_bow = [dictionary.doc2bow(clean_texts[0])]
corpus_bow = [dictionary.doc2bow(text) for text in clean_texts]
print(f'\nCorpus BoW--> {corpus_bow}\n')

# inverse_document_frequency
tf_idf = gensim.models.TfidfModel(corpus_bow)[corpus_bow]
print(f'TD_IDF --> {tf_idf}\n')
print(tf_idf[corpus_bow[0]])

similarity_object = gensim.similarities.Similarity('.', tf_idf[corpus_bow], num_features=len(dictionary))
print('Similarity Object --> ', similarity_object, type(similarity_object), '\n')

query_doc_tf_idf = tf_idf[corpus_bow]
print(f'Query doc tfidf --> {list(query_doc_tf_idf)} \n')


# text = corpora[0]
# print('\ntext --> ',text)

# for text in corpora:

#     # Each word turns an element
#     query_doc = gensim.utils.simple_preprocess(text)
#     print(f'\nQuery doc --> {query_doc} \n')


#     query_doc_bow = [dictionary.doc2bow(query_doc)]
#     # print(f'Query doc bow --> {query_doc_bow} {type(query_doc_bow)} \n')

#     query_doc_tf_idf = tf_idf[query_doc_bow]
#     print(f'Query doc tfidf --> {list(query_doc_tf_idf)} \n')

#     # similarity_scores = list(similarity_object[query_doc_tf_idf])
#     # print(f'Similarity scores --> {similarity_scores}')

#     # max_score = max(similarity_scores)
#     # print(f'Max score --> {max_score}')

#     # max_scored_file = similarity_scores.index(max_score)
#     # print(f'max scored file na posicao --> {max_scored_file}')