import streamlit as st
import pickle
import string, nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def transform_text(text):
    ps = PorterStemmer()
    #1 Lower text
    text = text.lower()
    
    #2 tokenization  , break in words
    text = nltk.word_tokenize(text)
    
    #3 remove special char
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    #4 remove stopwords
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    #5 stemming
    for i in text:
        y.append(ps.stem(i))
        
    
    return " ".join(y)
   




tfidf = pickle.load(open('/home/kali/Documents/coding/python/ml_projects/spam/vect.pkl', 'rb'))


model = pickle.load(open('/home/kali/Documents/coding/python/ml_projects/spam/mnb.pkl', 'rb'))


st.title("SMS/EMAIL CLASSIFIER")
input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    


    #preprocess

    
    transformed_sms = transform_text(input_sms)



    #vectorize
    vecor_input = tfidf.transform([transformed_sms])



    #predict
    result = model.predict(vecor_input)[0]

    #disply
    if result==1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")