import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

dataset = pd.read_csv("../data/ml-100k/u.data", sep='\t', names="user_id,item_id,rating,timestamp".split(","))
print dataset.head()
print len(dataset.user_id.unique()), len(dataset.item_id.unique())

dataset.user_id = dataset.user_id.astype('category').cat.codes.values
dataset.item_id = dataset.item_id.astype('category').cat.codes.values
