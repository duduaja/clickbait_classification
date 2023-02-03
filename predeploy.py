import tensorflow as tf
model = tf.keras.models.load_model('my_model')

import pandas as pd
df = pd.read_csv('data_balance.csv')

#split to train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['text'], 
                                                    df['y'], 
                                                    test_size=0.3,
                                                   random_state=11)

# Defining pre-processing hyperparameters
from tensorflow.keras.preprocessing.text import Tokenizer
max_len = 55
trunc_type = "post" 
padding_type = "post" 
oov_tok = "<OOV>" 
vocab_size = 500

tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = oov_tok)
tokenizer.fit_on_texts(x_train)

# def seq_pad(input_series):
#     sequence_result = tokenizer.texts_to_sequences(input_series)
#     pad_result = pad_sequences(testing_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type)
#     return pad_result

from tensorflow.keras.preprocessing.sequence import pad_sequences
def predict_spam(predict_msg):
    new_seq = tokenizer.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating=trunc_type)
    return (model.predict(padded))