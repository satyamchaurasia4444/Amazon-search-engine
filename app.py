import numpy as np
import pandas as pd
import nltk
import streamlit as st
from PIL import Image


amazon_df=pd.read_csv('amazon_product.csv')
amazon_df.head()

amazon_df.drop('id',axis=1,inplace=True)
amazon_df.head()

amazon_df.isnull().sum()

from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer('english')
def tokenize_stem(text):
    tokens=nltk.word_tokenize(text.lower())
    stemmed=[stemmer.stem(w) for w in tokens]
    return " ".join(stemmed)


amazon_df['Description']=amazon_df['Description'].apply(tokenize_stem)
amazon_df['Title']=amazon_df['Title'].apply(tokenize_stem)

amazon_df.head()

amazon_df['stemmed_tokens']=amazon_df['Title']+amazon_df['Description']

amazon_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer()

def cosine_sim(txt1, txt2):
    tfidf_matrix = tfidf.fit_transform([txt1, txt2])
    cos_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return cos_sim[0][0]  # return just the scalar value

def search_product(query):
    stemmed_query=tokenize_stem(query)
    # calculating cosine simalarity between query and stemmed tokens columns
    amazon_df['simalarity']=amazon_df['stemmed_tokens'].apply(lambda x:cosine_sim(stemmed_query,x))
    res=amazon_df.sort_values(by=['simalarity'],ascending=False).head(10)[['Title','Description','Category']]
    return res


# web app

img=Image.open('download.png')
st.image(img,width=500)
st.title("Amazon Web Search Engine")

query=st.text_input('Enter the product name')
submit=st.button('Search')
if submit:
    res=search_product(query)
    st.write(res)




















