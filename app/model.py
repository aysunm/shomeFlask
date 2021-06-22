import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

tokenizer = None
label_category_tokenizer = None
label_action_tokenizer = None
label_action_needed_tokenizer = None
label_question_tokenizer = None
label_subcategory_tokenizer = None
label_time_tokenizer = None
history_category = None
history_action = None
history_action_needed = None
history_question = None
history_subcategory = None
history_time = None
labelsxcategory = None
labelsxaction = None
labelsxactionneeded = None
labelsxquestion = None
labelsxsubcategory = None
labelsxtime = None

def init_model():
    global tokenizer
    global label_category_tokenizer
    global label_action_tokenizer
    global label_action_needed_tokenizer
    global label_question_tokenizer
    global label_subcategory_tokenizer
    global label_time_tokenizer
    global history_category
    global history_action
    global history_action_needed
    global history_question
    global history_subcategory
    global history_time
    global labelsxcategory
    global labelsxaction
    global labelsxactionneeded
    global labelsxquestion
    global labelsxsubcategory
    global labelsxtime

    with open('tokenizer_sentences.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    label_category_tokenizer = None
    with open('tokenizer_category.pickle', 'rb') as handle:
        label_category_tokenizer = pickle.load(handle)
    labelsxcategory = {value: key for key, value in label_category_tokenizer.word_index.items()}

    with open('tokenizer_action.pickle', 'rb') as handle:
        label_action_tokenizer = pickle.load(handle)
    labelsxaction = {value: key for key, value in label_action_tokenizer.word_index.items()}

    with open('tokenizer_action_needed.pickle', 'rb') as handle:
        label_action_needed_tokenizer = pickle.load(handle)
    labelsxactionneeded = {value: key for key, value in label_action_needed_tokenizer.word_index.items()}

    with open('tokenizer_question.pickle', 'rb') as handle:
        label_question_tokenizer = pickle.load(handle)
    labelsxquestion = {value: key for key, value in label_question_tokenizer.word_index.items()}

    with open('tokenizer_subcategory.pickle', 'rb') as handle:
        label_subcategory_tokenizer = pickle.load(handle)
    labelsxsubcategory = {value: key for key, value in label_subcategory_tokenizer.word_index.items()}

    with open('tokenizer_time.pickle', 'rb') as handle:
        label_time_tokenizer = pickle.load(handle)
    labelsxtime = {value: key for key, value in label_time_tokenizer.word_index.items()}

    history_category = tf.keras.models.load_model("category_model")
    history_action = tf.keras.models.load_model("action_model")
    history_action_needed = tf.keras.models.load_model("action_needed_model")
    history_question = tf.keras.models.load_model("question_model")
    history_subcategory = tf.keras.models.load_model("subcategory_model")
    history_time = tf.keras.models.load_model("time_model")

def predict_model(sentence):
    sentence = sentence.replace("%20", " ")
    print(sentence)
    global tokenizer
    global label_category_tokenizer
    global label_action_tokenizer
    global label_action_needed_tokenizer
    global label_question_tokenizer
    global label_subcategory_tokenizer
    global label_time_tokenizer
    global history_category
    global history_action
    global history_action_needed
    global history_question
    global history_subcategory
    global history_time
    global labelsxcategory
    global labelsxaction
    global labelsxactionneeded
    global labelsxquestion
    global labelsxsubcategory
    global labelsxtime

    max_length = 250
    trunc_type = 'post'
    padding_type = 'post'
    new_command = [sentence]
    seq = tokenizer.texts_to_sequences(new_command)
    padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    pred_category = history_category.predict(padded)
    pred_action = history_action.predict(padded)
    pred_action_needed = history_action_needed.predict(padded)
    pred_question = history_question.predict(padded)
    pred_subcategory = history_subcategory.predict(padded)
    pred_time = history_time.predict(padded)

    result = {
              'category': labelsxcategory[np.argmax(pred_category)],
              'action': labelsxaction[np.argmax(pred_action)],
              'action_needed': labelsxactionneeded[np.argmax(pred_action_needed)],
              'question': labelsxquestion[np.argmax(pred_question)],
              'sub_category': labelsxsubcategory[np.argmax(pred_subcategory)],
              'time': labelsxtime[np.argmax(pred_time)]
              }
    return result

