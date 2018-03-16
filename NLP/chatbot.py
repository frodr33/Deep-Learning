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