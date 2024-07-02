import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

#import data set
dataset = pd.read_csv("data.csv")
print(dataset.columns)
print(dataset['popularity'].dtype)
convert = dataset['popularity']
print(dataset['popularity'].min())
columnrange = convert.max() - convert.min()
print(columnrange)
