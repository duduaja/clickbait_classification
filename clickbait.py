import pandas as pd
import numpy as np

#for text pre-processing
import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#model
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional
import sklearn.metrics

import warnings
warnings.filterwarnings("ignore")

#load data
raw_data = pd.read_csv('https://raw.githubusercontent.com/ruzcmc/ClickbaitIndo-textclassifier/master/all_agree.csv')

#text cleaning
def regex(text):
    text = text.lower() 
    text=text.strip()  
    text= re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text= re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text)
    text = re.sub(r'\s*(https|http|com)',' ',text)
    return text

#tokenize sentence, from 'hai my name is annabelle', into ['hai','my','name','is','annabelle']
def tokenize(x):
    return word_tokenize(x)

#remove unnecessary word in list, from ['hai','my','name','is','annabelle'], into ['name','annabelle']
more_stopword = ['dengan', 'ia','bahwa','oleh','sebut','bikin','gara']
list_stopwords = set(stopwords.words('Indonesian')+more_stopword)

def remove_stopwords(x):
    return [word for word in x if word not in list_stopwords]

#unlist the list, from ['name','annabelle'], into 'name annabelle'
def unlist(x):
    return ' '.join(x)

def preprocess(df,col_text,col_label):
    df = df[[col_label,col_text]]
    df.columns = ['y','text']
    df['text'] = df['text'].apply(regex)
    df['text'] = df['text'].apply(tokenize)
    df['text'] = df['text'].apply(remove_stopwords)
    df['text'] = df['text'].apply(unlist)
    return df

data_clean = preprocess(raw_data,'title','label_score')

#downsampling
cb_text = data_clean[data_clean.y==1]
ncb_text = data_clean[data_clean.y==0]

#downsampling
ncb_text = data_clean[data_clean.y==0].sample(n=len(cb_text),random_state=11)
data_balance = cb_text.append(ncb_text).reset_index(drop=True)

# # Get length column for each text
# data_balance['text_length'] = data_balance['text'].apply(len)#Calculate average length by label types
# labels = data_balance.groupby('y').mean()

#split data
X = data_balance['text']
y = data_balance['y']
#split to train and test data
x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3,
                                                   random_state=11)

#preparing modelling
# Defining pre-processing hyperparameters
max_len = 55
trunc_type = "post" 
padding_type = "post" 
oov_tok = "<OOV>" 
vocab_size = 500

tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = oov_tok)
tokenizer.fit_on_texts(x_train)

#sequences and padding
# Sequencing and padding on training and testing 
training_sequences = tokenizer.texts_to_sequences(x_train)
training_padded = pad_sequences (training_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type )

testing_sequences = tokenizer.texts_to_sequences(x_test)
testing_padded = pad_sequences(testing_sequences, maxlen = max_len,
padding = padding_type, truncating = trunc_type)

# #modelling
# vocab_size = 500 # As defined earlier
# embeding_dim = 16
# drop_value = 0.2 # dropout
# n_dense = 24

# model = Sequential()
# model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(24, activation='relu'))
# model.add(Dropout(drop_value))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam' ,
#               metrics=['accuracy'])

# from tensorflow import keras

# # fitting a dense spam detector model
# num_epochs = 40
# early_stop = EarlyStopping(monitor='val_loss', patience=3)

# history = model.fit(training_padded, 
#                     y_train, 
#                     epochs=num_epochs, 
#                     validation_data=(testing_padded, y_test),
#                     callbacks =[early_stop], 
#                     verbose=2)
model = tf.keras.models.load_model('my_model')
model.predict(testing_padded)


#Defining prediction function
def predict_spam(predict_msg):
    new_seq = tokenizer.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    return (model.predict(padded))



#DEPLOY ON STREAMLIT
import streamlit as st


def main():
    st.title('Clickbait Classification')

    #input
    text_input = st.text_input('Headline News')

    #prediction
    if st.button('Predict'):
        list_text_input = []
        list_text_input.append(text_input)
        pred_result = predict_spam(list_text_input)[0][0]
        # st.success('{headline} \n**{pred_result:.2%}** clickbait'.format(headline=text_input,pred_result=pred_result))

        st.markdown('<p style="font-size:50px">{pred_result:.2%} clickbait</p>'.format(pred_result=pred_result), unsafe_allow_html=True)

if __name__=='__main__':
    main()