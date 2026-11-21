import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# પીકલ ફાઈલો લોડ કરવા માટે
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # ૧. ટેક્સ્ટને પ્રીપ્રોસેસ કરો
    transformed_sms = transform_text(input_sms)
    # ૨. વેક્ટરાઈઝ કરો
    vector_input = tfidf.transform([transformed_sms])
    # ૩. પ્રિડિક્ટ કરો
    result = model.predict(vector_input)
    # ૪. આઉટપુટ ડિસ્પ્લે કરો
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")