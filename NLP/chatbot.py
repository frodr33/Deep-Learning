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

