import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

TEST_DATA_FILE = "/Users/amitn/Documents/kaggle/toxicity/test.csv"

test = pd.read_csv(TEST_DATA_FILE)
list_sentences_test = test["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

MAX_FEATURES = 50000
MAX_LEN = 200
EMBED_SIZE = 128

tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(list_sentences_test))
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_test, maxlen=MAX_LEN)

def get_model():
    inp = Input(shape=(MAX_LEN, ))
    x = Embedding(MAX_FEATURES, EMBED_SIZE)(inp)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss= 'binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model = get_model()
batch_size = 32

MODEL_FILE_PATH = "/Users/amitn/Documents/kaggle/toxicity/weights_base.best_200.hdf5"
model.load_weights(MODEL_FILE_PATH)
y_test = model.predict(X_t)
sample_submission = pd.read_csv("/Users/amitn/Documents/kaggle/toxicity/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("/Users/amitn/Documents/kaggle/toxicity/baseline.csv", index=False)
