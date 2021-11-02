import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test-name', help='The name alias for this test', required=True, dest='TEST_NAME', type=str)
parser.add_argument('--output-directory', help='The output directory', dest='OUTPUT_DIRECTORY', type=str, required=True)
parser.add_argument('--print-on-screen', dest='PRINT_ON_SCREEN', action='store_true')
parser.add_argument('--top-n-assignees', default=21, help='Top N assignees to be considered from the classifier',
                    dest='TOP_N_ASSIGNEES', type=int)
parser.add_argument('--random-state', default=42, dest='RANDOM_STATE', type=int)
parser.add_argument('--kfold-n-splits', default=10, help='The number of k fold splits', dest='KFOLD_N_SPLITS', type=int)
parser.add_argument('--epochs', default=15, help='Number of epochs', dest='EPOCHS', type=int)
parser.add_argument('--max-tokens', default=20000,
                    help='The number of the most common words to be considered from the text tokenizer',
                    dest='MAX_TOKENS', type=int)
parser.add_argument('--output-sequence-length', default=100, help='Keep only the first N tokens of each text',
                    dest='OUTPUT_SEQUENCE_LENGTH', type=int)
parser.add_argument('--bow-model', help='If this flag is set, then build a simple BOW model without an embedding layer',
                    dest='BOW_MODEL', action='store_true')
parser.add_argument('--word2vec-model', help='If this flag is set, then build a neural network with an embedding layer',
                    dest='WORD2VEC_MODEL', action='store_true')
parser.add_argument('--keyed-vector',
                    help='If this flag is set, then build a neural network with an embedding layer from a keyed vector object',
                    dest='KEYED_VECTOR', action='store_true')
parser.add_argument('--vectorizer-batch-size', default=200, dest='VECTORIZER_BATCH_SIZE', type=int)
parser.add_argument('--train-batch-size', default=128, dest='TRAIN_BATCH_SIZE', type=int)

parser.add_argument('--trainable-embedding',
                    help='Define this option only when you use a model with an embedding layer',
                    dest='TRAINABLE_EMBEDDING', action='store_true')
parser.add_argument('--dense-layer-n-units', default=100,
                    help='Define this option only when you use a model with an embedding layer',
                    dest='DENSE_LAYER_N_UNITS', type=int)
parser.add_argument('--embedding-dim', default=100,
                    help='Define this option only when you use a model with an embedding layer', dest='EMBEDDING_DIM',
                    type=int)
parser.add_argument('--modelpath', default='',
                    help='The embedding model path. Define this option only when you use a model with an embedding layer',
                    dest='MODELPATH', type=str)

parser.add_argument('--dense-first-layer-n-units-simple', default=100,
                    help='Define this option only when you use a model without an embedding layer',
                    dest='DENSE_FIRST_LAYER_N_UNITS_SIMPLE', type=int)
parser.add_argument('--dense-second-layer-n-units-simple', default=50,
                    help='Define this option only when you use a model without an embedding layer',
                    dest='DENSE_SECOND_LAYER_N_UNITS_SIMPLE', type=int)
parser.add_argument('--output-mode', default='count',
                    help='Define this option only when you use a model without an embedding layer', dest='OUTPUT_MODE',
                    type=str)

args = parser.parse_args()
if args.BOW_MODEL is False and args.MODELPATH == '':
    parser.error('Please specify the path for the embedding model.')

###############################################################################
# PARAMETERS
# General parameters.
TEST_NAME = args.TEST_NAME
KFOLD_N_SPLITS = args.KFOLD_N_SPLITS
MAX_TOKENS = args.MAX_TOKENS
OUTPUT_SEQUENCE_LENGTH = args.OUTPUT_SEQUENCE_LENGTH
TOP_N_ASSIGNEES = args.TOP_N_ASSIGNEES
EPOCHS = args.EPOCHS
BOW_MODEL = args.BOW_MODEL
WORD2VEC_MODEL = args.WORD2VEC_MODEL
KEYED_VECTOR = args.KEYED_VECTOR
OUTPUT_DIRECTORY = args.OUTPUT_DIRECTORY
RANDOM_STATE = args.RANDOM_STATE
VECTORIZER_BATCH_SIZE = args.VECTORIZER_BATCH_SIZE
TRAIN_BATCH_SIZE = args.TRAIN_BATCH_SIZE
PRINT_ON_SCREEN = args.PRINT_ON_SCREEN

# For a Model with embeddings parameters.
TRAINABLE_EMBEDDING = args.TRAINABLE_EMBEDDING
DENSE_LAYER_N_UNITS = args.DENSE_LAYER_N_UNITS
EMBEDDING_DIM = args.EMBEDDING_DIM
MODELPATH = args.MODELPATH

# For a simple BOW model parameters.
DENSE_FIRT_LAYER_N_UNITS_SIMPLE = args.DENSE_FIRST_LAYER_N_UNITS_SIMPLE
DENSE_SECOND_LAYER_N_UNITS_SIMPLE = args.DENSE_SECOND_LAYER_N_UNITS_SIMPLE
OUTPUT_MODE = args.OUTPUT_MODE
###############################################################################

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from sklearn.model_selection import KFold
import pickle
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from sklearn import preprocessing
import sklearn
import texthero as hero
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def load_word_vector_and_vocabulary(modelpath, word2vec_model, keyed_vector):
    if word2vec_model is True:
        if keyed_vector is False:
            model = Word2Vec.load(modelpath)
            word_vectors = model.wv
        else:
            word_vectors = KeyedVectors.load_word2vec_format(modelpath, binary=True)
        embedding_vocabulary = word_vectors.vocab.keys()
        return word_vectors, embedding_vocabulary
    else:
        word_df = pd.read_csv(modelpath)
        word_vectors = word_df.set_index('word')['embedding'].to_dict()
        word_vectors = {idx: eval(word_vectors[idx]) for idx in word_vectors.keys()}
        embedding_vocabulary = list(word_vectors.keys())
        return word_vectors, embedding_vocabulary


def generate_model_with_embedding(train_samples, word_vectors, embedding_vocabulary, class_names, features=None):
    print('# Run generate_model_with_embedding function #')
    if features is not None:
        vectorizer = TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=OUTPUT_SEQUENCE_LENGTH)
        text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(VECTORIZER_BATCH_SIZE)
        vectorizer.adapt(text_ds)
        voc = vectorizer.get_vocabulary()[2:]
        features = list(set(voc + features))
        vectorizer = TextVectorization(output_sequence_length=OUTPUT_SEQUENCE_LENGTH)
        vectorizer.set_vocabulary(features)
        print(f'################# vocabulary size: {len(vectorizer.get_vocabulary())}')
    else:
        vectorizer = TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=OUTPUT_SEQUENCE_LENGTH)
        text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(VECTORIZER_BATCH_SIZE)
        vectorizer.adapt(text_ds)
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    embeddings_index = {}
    for word in embedding_vocabulary:
        embeddings_index[word] = word_vectors[word]
    print("Found %s word vectors." % len(embeddings_index))
    num_tokens = len(voc) + 2
    hits = 0
    misses = 0
    print('number of tokens')
    print(num_tokens)

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    embedding_layer = Embedding(
        num_tokens,
        EMBEDDING_DIM,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=TRAINABLE_EMBEDDING,
    )

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        embedding_layer,
        tf.keras.layers.Dense(DENSE_LAYER_N_UNITS, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(len(class_names), activation='softmax'),
    ])

    model.summary()
    return model


def generate_model_without_embedding(train_samples, class_names):
    print('# Run generate_model_without_embedding function #')
    vectorizer = TextVectorization(max_tokens=MAX_TOKENS, output_mode='count')
    text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(VECTORIZER_BATCH_SIZE)
    vectorizer.adapt(text_ds)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorizer,
        tf.keras.layers.Dense(DENSE_FIRT_LAYER_N_UNITS_SIMPLE, activation='relu'),
        tf.keras.layers.Dense(DENSE_SECOND_LAYER_N_UNITS_SIMPLE, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(class_names), activation='softmax'),
    ])
    model.summary()
    return model


def main(test_name):
    if os.path.exists(OUTPUT_DIRECTORY):
        shutil.rmtree(OUTPUT_DIRECTORY)
    os.mkdir(OUTPUT_DIRECTORY)

    print(f'Starting {test_name}!')
    if PRINT_ON_SCREEN is False:
        sys.stdout = open(f'{OUTPUT_DIRECTORY}/report.txt', 'w')
    print(f'Test name:{test_name}')
    print(sys.argv)
    print(args)
    if BOW_MODEL is True:
        print('##### Generating a BOW model, thus ignoring embeddings models and vocabularies! #####')

    df = pd.read_csv('issues.csv')
    df['text'] = df['title'] + ' ' + df['description']
    df = pd.DataFrame(df[df['text'].notna()])
    df['text'] = hero.remove_diacritics(df['text'])
    with open('features/features_250.pkl', 'rb') as f:
        features = pickle.load(f)
        features = list(set(features))
        print(f'# of unique features:{len(features)}')
    selected_assignees = df['assignee'].value_counts()[:TOP_N_ASSIGNEES].index.tolist()
    selected_df = pd.DataFrame(df[df['assignee'].isin(selected_assignees)])
    print(f'Selected_df size: {len(selected_df)}')

    le = preprocessing.LabelEncoder()
    le.fit(selected_assignees)
    with open(f'{OUTPUT_DIRECTORY}/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    selected_df['class'] = le.transform(selected_df['assignee'])
    selected_df = sklearn.utils.shuffle(selected_df)
    class_names = le.classes_

    if BOW_MODEL is False:
        word_vectors, embedding_vocabulary = load_word_vector_and_vocabulary(MODELPATH, WORD2VEC_MODEL, KEYED_VECTOR)

    kf = KFold(n_splits=KFOLD_N_SPLITS)
    samples = selected_df['text'].values
    labels = selected_df['class'].values
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    best_accuracy = 0
    best_accuracy_precision = 0
    best_accuracy_recall = 0
    best_accuracy_f1_score = 0
    best_model = None
    train_losses = []
    val_losses = []
    histories = []
    for train_indices, test_indices in kf.split(samples):
        train_samples = samples[train_indices]
        val_samples = samples[test_indices]
        train_labels = labels[train_indices]
        val_labels = labels[test_indices]

        if BOW_MODEL is True:
            model = generate_model_without_embedding(train_samples, class_names)
        else:
            model = generate_model_with_embedding(train_samples, word_vectors, embedding_vocabulary, class_names,
                                                  features)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
        x_train = train_samples
        x_val = val_samples
        y_train = train_labels
        y_val = val_labels
        history = model.fit(x_train, y_train, batch_size=TRAIN_BATCH_SIZE,
                            steps_per_epoch=len(x_train) // TRAIN_BATCH_SIZE, epochs=EPOCHS,
                            validation_data=(x_val, y_val))

        history = history.history
        histories.append(history)
        train_losses.append(history['loss'][-1])
        val_losses.append(history['val_loss'][-1])
        accuracy = model.evaluate(x_val, y_val)[1]
        accuracies.append(accuracy)
        y_true = y_val
        y_pred = np.argmax(model.predict(x_val), axis=1)
        precision = precision_score(y_true, y_pred, average='micro')
        precisions.append(precision)
        recall = recall_score(y_true, y_pred, average='micro')
        recalls.append(recall)
        f1 = f1_score(y_true, y_pred, average='micro')
        f1_scores.append(f1)

        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_accuracy_precision = precision
            best_accuracy_recall = recall
            best_accuracy_f1_score = f1
            best_model = model
            best_x_val = x_val
            best_y_val = y_val
    best_model.save(f'{OUTPUT_DIRECTORY}/nn_model')

    y_true = le.inverse_transform(best_y_val)
    y_pred = le.inverse_transform(np.argmax(best_model.predict(best_x_val), axis=1))
    with open(f'{OUTPUT_DIRECTORY}/best_model_confusion_matrix.pkl', 'wb') as f:
        pickle.dump(confusion_matrix(y_true, y_pred, labels=le.classes_), f)
    with open(f'{OUTPUT_DIRECTORY}/histories.pkl', 'wb') as f:
        pickle.dump(histories, f)

    print(f'Accuracies: {accuracies}')
    print(f'Precisions: {precisions}')
    print(f'Recalls: {recalls}')
    print(f'F1 scores: {f1_scores}')
    print(f'Average accuracy {np.average(accuracies)}')
    print(f'Accuracy sd: {np.std(accuracies)}')
    print(f'Best accuracy: {best_accuracy}')
    print(f'Min accuracy: {np.min(accuracies)}')
    print(f'Best accuracy precision: {best_accuracy_precision}')
    print(f'Average precision: {np.average(precisions)}')
    print(f'Best accuracy recall: {best_accuracy_recall}')
    print(f'Average recall: {np.average(recalls)}')
    print(f'Best accuracy f1 score: {best_accuracy_f1_score}')
    print(f'Average f1 score: {np.average(f1_scores)}')
    print(f'Average train loss: {np.average(train_losses)}')
    print(f'Average validation loss: {np.average(val_losses)}')

    if PRINT_ON_SCREEN is False:
        sys.stdout.close()


if __name__ == '__main__':
    main(TEST_NAME)