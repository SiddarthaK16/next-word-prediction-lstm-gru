import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model('next_word_lstm.keras')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

#Function to predict next word

def predict_next_word(model, tokenizer, input_text):
    max_len = model.input_shape[1]

    token_list = tokenizer.texts_to_sequences([input_text])[0]



    token_list = pad_sequences(
        [token_list],
        maxlen=max_len,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)

    # Ignore index 0 (padding)
    predicted_word_index = np.argmax(predicted[0][1:]) + 1

    predicted_word = tokenizer.index_word.get(
        predicted_word_index,
        "Unknown"
    )

    return predicted_word

#streamlit app

st.title("Next word prediction")
input_text = st.text_input("Enter a sentence")

if st.button("Predict next word"):
    max_sequence_len=model.input_shape[1]+1
    next_word = predict_next_word(model, tokenizer, input_text)
    st.write(f"Next Word: {next_word}")


