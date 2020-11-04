import warnings
warnings.filterwarnings("ignore")

from gensim.summarization import summarize


my_text = """
Automatic summarization is the process of shortening a text document with software, in order to create a summary with the major points of the original document. Technologies that can make a coherent summary take into account variables such as length, writing style and syntax.

Automatic data summarization is part of machine learning and data mining. The main idea of summarization is to find a subset of data which contains the "information" of the entire set. Such techniques are widely used in industry today. Search engines are an example; others include summarization of documents, image collections and videos. Document summarization tries to create a representative summary or abstract of the entire document, by finding the most informative sentences, while in image summarization the system finds the most representative and important (i.e. salient) images.
For surveillance videos, one might want to extract the important events from the uneventful context.

There are two general approaches to automatic summarization: extraction and abstraction. Extractive methods work by selecting a subset of existing words, phrases, or sentences in the original text to form the summary. In contrast, abstractive methods build an internal semantic representation and then use natural language generation techniques to create a summary that is closer to what a human might express. Such a summary might include verbal innovations. Research to date has focused primarily on extractive methods, which are appropriate for image collection summarization and video summarization."""

# Length of text
print(f'Lenght of text {len(my_text)}')

# Amount of words
print(f'The text has {len(my_text.split())} words')

print(my_text, '\n')

# Summarize the text
summarized_text = summarize(my_text)
print(summarized_text)
print(f'\nLength of summarized text {len(summarized_text)}\n')
print(f'Summarized text has {len(summarized_text.split())} words\n')

# How to get the result as a list of string
summarized_list = summarize(my_text, split=True)
print(f'\nSummarizes list {summarized_list}')

# How to set the amount of text you want as summary
#   ratio (default=0.2)
print(f'\n50% of summary output\n {summarize(my_text, ratio=0.5)}\n')

# How to get the maximum amount of words in the summary 
#   word_count
fifty_words_summarized = summarize(my_text, word_count=50)
print(f'\nWords about 50 {fifty_words_summarized}\n')
print(f'To prove {len(fifty_words_summarized.split())}')