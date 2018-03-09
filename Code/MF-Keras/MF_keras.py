import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("../data/ml-100k/u.data", sep='\t', names="user_id,item_id,rating,timestamp".split(","))
print dataset.head()
print len(dataset.user_id.unique()), len(dataset.item_id.unique())

dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
train, test = train_test_split(dataset, test_size=0.2)
print len(train), len(test)

