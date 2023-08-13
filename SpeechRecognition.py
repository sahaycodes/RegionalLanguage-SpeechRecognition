#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[3]:


get_ipython().system('pip install jiwer')


# In[2]:


from jiwer import wer
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# In[3]:


from IPython import display
import seaborn as sns


# In[8]:


get_ipython().system('pip install librosa')


# In[4]:


import librosa.display
from collections import Counter


# In[ ]:





# Data Analysis of Bengali Language 
# 1.EDA
# 2.audio feature analysis
# 3.noise & variability analysis
# 4.label analysis

# In[19]:


#EDA
sans_wrd=r"D:\Users\User\Downloads\shrutilipi_fairseq\shrutilipi_fairseq\sanskrit\train.wrd"
with open(sans_wrd,"r",encoding="utf-8") as wrd_file:
    lines=wrd_file.readlines()
    sentences=[line.strip() for line in lines]
    word_length=[len(line.strip().split()) for line in lines]
    word=[line.strip() for line in lines]


# In[35]:


plt.hist(word_length,bins=30)
plt.xlabel("Word Count")
plt.ylabel("frequency")
plt.title("Distribution of Sanskrit Word Counts")
plt.show()


# In[36]:


word_length=[len(word)for word in word]
plt.hist(word_length,bins=20)
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.title(" Sanskrit Word Length Distribution")


# In[37]:


from unicodedata import normalize
from collections import defaultdict
import unicodedata



def normalize_sanskrit_word(word):
    try:
        normalize_word=normalize("NFKC",word)
        normalize_word=''.join([c for c in normalize_word if not unicodedata.combining(c)])
        return normalize_word
    except:
        return word
    
    


# In[39]:


sanskrit_words=[word for sentence in sentences for word in sentence.split()]
vocab = set([normalize_sanskrit_word(word) for word in sanskrit_words])
word_to_index={word:index for index , word in enumerate(vocab)}
co_occurence_matrix=np.zeros((len(vocab),len(vocab)))

window_size=2
for sentence in sentences:
    tokens=sentence.split()
    for i , target in enumerate(tokens):
        target=normalize_sanskrit_word(target)
        target_index=word_to_index.get(target,-1)
        if target_index !=-1:
            for j in range(max(0,i-window_size),min(len(tokens),i+window_size+1)):
                context=tokens[j]
                if context != target:
                    context=normalize_sanskrit_word(context)
                    context_idx=word_to_index.get(context,-1)
                    if context_idx != -1:
                        co_occurrence_matrix[target_index][context_idx]+=1

               
            
            
            
co_occurence_matrix_sum= co_occurence_matrix.sum()
if co_occurence_matrix_sum==0:
    co_occurence_matrix_normalized=co_occurence_matrix
else:
    co_occurence_matrix_normalized=co_occurence_matrix/(co_occurence_matrix + 1e-6)



                
    


# In[24]:


#kinda work in progress


#calculate word counts
word_counts=Counter(sanskrit_words)


# In[33]:


#wordcloud formation 
from wordcloud import WordCloud

plt.rcParams['font.sans-serif']=['Noto Sans Devanagri']
wordcloud=WordCloud(width=800,height=400,background_color='white',font_path=r'D:\Users\User\Downloads\Fonts\NotoSansDevanagari-VariableFont_wdth,wght.ttf').generate_from_frequencies(word_counts)


# In[34]:



plt.figure(figsize=(10,80))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.title('Sanskrit WordCloud')
plt.show()


# In[40]:


import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist


# In[42]:


nltk.download('indian')


# In[43]:


sanskrit_txt="तत्त्वमसि महावाक्यं प्रसिद्धं महर्षिभिः। अहमस्मीति पदेनैव जीवो ब्रह्मैव संगच्छति॥"
wordss=nltk.word_tokenize(sanskrit_txt)

n=4
n_grams=list(ngrams(wordss,n))

ngram_freq=FreqDist(n_grams)


# In[ ]:





# In[ ]:




