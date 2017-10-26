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

import numpy as np

def quiet_logs( sc ):
  logger = sc._jvm.org.apache.log4j
  logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
  logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

# read the data into RDD
# For ratings.csv file
# sc = SparkContext()
sc = SparkContext(appName="MovieRating")
# Disable the verbose logs
quiet_logs( sc )

small_ratings_file = os.path.join('./datasets','ml-latest-small', 'ratings.csv')
small_ratings_raw_data = sc.textFile(small_ratings_file)
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1],tokens[2])).cache()

# small_ratings_data.take(3)
# [(u'1', u'6', u'2.0'), (u'1', u'22', u'3.0'), (u'1', u'32', u'2.0')]

# For movies.csv file
small_movies_file = os.path.join('./datasets','ml-latest-small', 'movies.csv')
small_movies_raw_data = sc.textFile(small_movies_file)
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]
small_movies_data = small_movies_raw_data.filter(lambda line: line!=small_movies_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0],tokens[1])).cache()

# small_movies_data.take(3)
# [(u'1', u'Toy Story (1995)'),
# (u'2', u'Jumanji (1995)'),
# (u'3', u'Grumpier Old Men (1995)')]

# Selecting ALS parameters using the small dataset
# Here I use validation; can be adapted to use cross-validation
training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


seed = 5L
iterations = 10
#lambda_s = [0.001, 0.1, 0.2, 0.3]
lambda_s = np.linspace(0.01,0.5,50)
#ranks = [4, 8, 12]
ranks = range(1,21)
errors = [0] * (len(ranks)*len(lambda_s))
print(errors)
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
best_lambda = -1
for rank in ranks:
    for lambda_i in lambda_s:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                          lambda_=lambda_i)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        # rates_and_preds.take(1)
        # [((361, 589), (5.0, 3.691408043965179))]
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[err] = error
        err += 1
        print 'For rank %s, %f the RMSE is %s' % (rank, lambda_i, error)
        if error < min_error:
            min_error = error
            best_rank = rank
            best_lambda = lambda_i

print 'The best model was trained with rank %s, lambda %f' % (best_rank, best_lambda)

# The best parameter is rank 1,lambda 0.1.
# The RMSE is 0.89 

# Test 
model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                      lambda_=best_lambda)
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print 'For testing data the RMSE is %s' % (error)
'''
# Using the complete dataset to build the final model; re-do the above
# Load the complete dataset file
complete_ratings_file = os.path.join('./datasets', 'ml-latest', 'ratings.csv')
complete_ratings_raw_data = sc.textFile(complete_ratings_file)
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]
# Parse
complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line!=complete_ratings_raw_data_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]),int(tokens[1]),float(tokens[2]))).cache()
print "There are %s recommendations in the complete dataset" % (complete_ratings_data.count())

# to avoid stackover flow
sc.setCheckpointDir('checkpoint/')

training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0L)
complete_model = ALS.train(training_RDD, best_rank, seed=seed, 
                           iterations=iterations, lambda_=best_lambda)

test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print 'For testing data the RMSE is %s' % (error)
'''

# Persist the model
model_path = os.path.join('./models', 'movie_lens_als')

# Save and load model
model.save(sc, model_path)
same_model = MatrixFactorizationModel.load(sc, model_path)
