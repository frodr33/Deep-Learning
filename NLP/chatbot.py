# Building a chatbox with Deep NLP

# Importing Libraries
import numpy as np
import tensorflow as tf
import re
import time


########## Data Preprocessing ############

# Import Data Set
lines = open("movie_lines.txt", encoding = "utf-8", errors = "ignore").read().split('\n')
conversations = open("movie_conversations.txt", encoding = "utf-8", errors = "ignore").read().split('\n')

# Create Dictionary to map each line with ID
# Easiest way to keep track of inputs and outputs
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
        
# Creating a list of all the conversations
conversations_ids = []
for conv in conversations[:-1]:
    _conv = conv.split( " +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conv.split(","))
    
# Getting separate lists of questions and answers
# In conversations_ids, the ith element is the Question
# and the ith + 1 element is the Answer
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) -1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"/@;:<>{}+=~|.?,]", "cannot", text)
    return text
    
# Clean Questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
# Clean Answers..todo
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
    
    
    
    
    
    
    
    