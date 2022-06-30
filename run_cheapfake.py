#!/usr/bin/env python3.6

from pyexpat.errors import XML_ERROR_FEATURE_REQUIRES_XML_DTD
import cv2
import os
import json
from timeit import default_timer as dt
#import sys
#sys.path.append(os.path.join(os.path.dirname('utils'), '..'))
#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/hdd3/malkaddour/datasets/cheapfakes/mmsys22cheapfakes/utils/')
#import utils
#utils.path.insert(1, '/hdd3/malkaddour/datasets/cheapfakes/mmsys22cheapfakes/utils/')
from utils.config import *
#from config import *
#from utilstext_utils import get_text_metadata
from utils.text_utils import get_text_metadata
#from model_archsmodels import CombinedModelMaskRCNN
from model_archs.models import CombinedModelMaskRCNN
from utils.common_utils import read_json_data
#from utilscommon_utils import read_json_data
#from utilseval_utils import *
from utils.eval_utils import *
from textblob import TextBlob
import numpy as np
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from textblob import TextBlob
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
from sklearn.model_selection import train_test_split
import pickle
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, sigmoid, softmax, tanh
from tensorflow.keras.initializers import RandomNormal
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.client import device_lib
from tensorflow.python.keras import backend as K
from tensorflow.keras.models import model_from_json
from nltk.tokenize import sent_tokenize
import shutil

# Models (create model according to text embedding)
if embed_type == 'use':
    # For USE (Universal Sentence Embeddings)
    model_name = 'img_use_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
else:
    # For Glove and Fasttext Embeddings
    model_name = 'img_lstm_glove_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(use=False, hidden_size=300, embedding_length=word_embeddings.shape[1]).to(device)

    print("Total Params", sum(p.numel() for p in combined_model.parameters() if p.requires_grad))

def get_scores(v_data):
    "This code is borrowed from COSMOS and COSMOS on Steroids"
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """
    checkpoint = torch.load(BASE_DIR + '/models_final/' + model_name + '.pt')
    combined_model.load_state_dict(checkpoint)
    combined_model.to(device)
    combined_model.eval()

    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])
    bbox_list = v_data['maskrcnn_bboxes']
    bbox_classes = [-1] * len(bbox_list)
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    bbox_list.append([0, 0, img_shape[1], img_shape[0]])  # For entire image (global context)
    bbox_classes.append(-1)
    cap1 = v_data['caption1_modified']
    cap2 = v_data['caption2_modified']

    img_tensor = [torch.tensor(img).to(device)]
    bboxes = [torch.tensor(bbox_list).to(device)]
    bbox_classes = [torch.tensor(bbox_classes).to(device)]

    if embed_type != 'use':
        # For Glove, Fasttext embeddings
        cap1_p = text_field.preprocess(cap1)
        cap2_p = text_field.preprocess(cap2)
        embed_c1 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap1_p]).unsqueeze(
            0).to(device)
        embed_c2 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap2_p]).unsqueeze(
            0).to(device)
    else:
        # For USE embeddings
        embed_c1 = torch.tensor(use_embed([cap1]).numpy()).to(device)
        embed_c2 = torch.tensor(use_embed([cap2]).numpy()).to(device)

    with torch.no_grad():
        z_img, z_t_c1, z_t_c2 = combined_model(img_tensor, embed_c1, embed_c2, 1, [embed_c1.shape[1]],
                                               [embed_c2.shape[1]], bboxes, bbox_classes)

    z_img = z_img.permute(1, 0, 2)
    z_text_c1 = z_t_c1.unsqueeze(2)
    z_text_c2 = z_t_c2.unsqueeze(2)

    # Compute Scores
    score_c1 = torch.bmm(z_img, z_text_c1).squeeze()
    score_c2 = torch.bmm(z_img, z_text_c2).squeeze()

    return score_c1, score_c2

def sentiment_vader(sentence):
    
    sid_obj = SentimentIntensityAnalyzer()

    sentiment_dict = sid_obj.polarity_scores(sentence)
    negative = sentiment_dict['neg']
    neutral = sentiment_dict['neu']
    positive = sentiment_dict['pos']
    compound = sentiment_dict['compound']

    if sentiment_dict['compound'] >= 0.05 :
        overall_sentiment = "Positive"

    elif sentiment_dict['compound'] <= - 0.05 :
        overall_sentiment = "Negative"

    else :
        overall_sentiment = "Neutral"
  
    return negative, neutral, positive, compound, overall_sentiment

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''
    from errno import EEXIST
    from os import makedirs, path
    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def analyze_sentiment(caption1, caption2):
        vs1, vs2 = sentiment_vader(caption1), sentiment_vader(caption2)
        blob1, blob2 = TextBlob(caption1), TextBlob(caption2)
        sub1, sub2 = blob1.sentiment_assessments.subjectivity, blob2.sentiment_assessments.subjectivity
        return vs1[3], vs2[3], sub1, sub2

def define_mlp(layer_sizes, inputdim, optimizer='adam',loss='binary_crossentropy'):
    inputs = tf.keras.Input(shape=(inputdim,), name="sentiments")
    x = Dense(layer_sizes[0],activation='relu')(inputs)
    for i, layerdim in enumerate(layer_sizes[2:-1]):
        x = Dense(layerdim, activation='relu')(tf.keras.layers.Dropout(0.5)(x))
    outputs = Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

numvars = 6
Nlayer = 300
no_of_layers = 5
layer_sizes = np.ndarray.tolist((Nlayer * np.ones((1, no_of_layers))).astype(int))[0]
test_samples = read_json_data(os.path.join(DATA_DIR, 'test.json'))

N = len(test_samples)
Xtest, ytest = np.zeros((N, numvars)), np.zeros((N, 1))

# Compute the six features needed to pass to the MLP
print("Computing features of test dataset...")
for i, test_data in enumerate(test_samples):
    #print(test_data['img_local_path'])
    ytest[i] = test_data['context_label'] # label the test vector using 0 for NOOC, 1 for OOC
    cap1, cap2 = test_data['caption1'], test_data['caption2']

    # get polarity and subjectivity values
    p1, p2, s1, s2 = analyze_sentiment(cap1, cap2)
    Xtest[i, :2] = np.array([np.abs(p1 - p2), np.abs(s1 - s2)])

    # Use COSMOS and COSMOS on steroids implementation to compute the image-text features (borrowed from their source code)
    bboxes = test_data['maskrcnn_bboxes']
    score_c1, score_c2 = get_scores(test_data)
    textual_sim = float(test_data['bert_base_score'])
    scores_c1 = top_scores(score_c1)
    scores_c2 = top_scores(score_c2)
    top_bbox_c1, top_bbox_next_c1 = top_bbox_from_scores(bboxes, score_c1)
    top_bbox_c2, top_bbox_next_c2 = top_bbox_from_scores(bboxes, score_c2)
    bbox_overlap = is_bbox_overlap(top_bbox_c1, top_bbox_c2, iou_overlap_threshold)
    bbox_overlap_next = is_bbox_overlap(top_bbox_next_c1, top_bbox_next_c2, iou_overlap_threshold)
    iou = bb_intersection_over_union(top_bbox_c1, top_bbox_c2)
    curfake, curopp = is_fake(test_data)[0], is_opposite(test_data)[0]

    # Put them into the feature matrix
    Xtest[i, 2:] = textual_sim, iou, curfake, curopp

weightpath = os.path.join(BASE_DIR, 'newresults', 'modelweights', 'PT_1000.h5')
print("Defining and evaluating on MLP...")

# Define our MLP using layer_sizes vector containing hidden layer size in each element.
# Set numvars to N, where N is the number of features you wish to train the MLP with. We use 6 for our model
mlp_model = define_mlp(layer_sizes, numvars)
mlp_model.load_weights(weightpath)
mlp_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()])

# Make the predictions
predictions = mlp_model.predict(Xtest)

# Calculate accuracy and get vector of output labels
print("Generating labels and computing accuracy...")
false_context, mlp_false_idx, output_labels  = 0, [], []
metrics = {
    "TP": 0,
    "TN": 0,
    "FP": 0,
    "FN": 0,
}
for k, yt in enumerate(ytest):
    pred, gt = int(np.round_(predictions[k])), int(yt)
    if gt == 1:
        if pred == gt:
            metrics["TP"] += 1
        else:
            metrics["FN"] += 1
    else:
        if pred == gt:
            metrics["TN"] += 1
        else:
            metrics["FP"] += 1
    output_labels.append(pred)
no_of_false = (metrics["FP"] + metrics["FN"])
no_of_true = metrics["TP"] + metrics["TN"]
print(f"Number of false predictions: {no_of_false}/{Xtest.shape[0]}")
print(f"Accuracy of MLP on test set: {no_of_true/Xtest.shape[0]*100}%")

with open("/test_labels.json", "w") as fp:
    json.dump(output_labels, fp)