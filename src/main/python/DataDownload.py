# Programmed by Joshua
# This file fulfills the function of movie rating predictions and is going to be developed into 
# a movie recommendation system based on Python/Flask/Spark

import os
import urllib
import zipfile
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
import math

# Download data from movielens website
complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
datasets_path = './datasets'
# if data file does not exists
if not os.path.exists(datasets_path):
    os.mkdir('./datasets')
complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')
small_f = urllib.urlretrieve (small_dataset_url, small_dataset_path)
complete_f = urllib.urlretrieve (complete_dataset_url, complete_dataset_path)

# extract the data in zip files
with zipfile.ZipFile(small_dataset_path, "r") as z:
    z.extractall(datasets_path)

with zipfile.ZipFile(complete_dataset_path, "r") as z:
    z.extractall(datasets_path)