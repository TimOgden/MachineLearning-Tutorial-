import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lematizer = WordNetLemmatizer()
hm_lines = 100000000
