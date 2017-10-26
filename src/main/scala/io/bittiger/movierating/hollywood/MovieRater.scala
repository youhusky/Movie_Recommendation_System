package io.bittiger.movierating.hollywood

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

/**
  * Created by Joshua
  *
  * small_dataset_url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
  * complete_dataset_url = "http://files.grouplens.org/datasets/movielens/ml-latest.zip"
  *
  */
object MovieRater extends App {

  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  Logger.getRootLogger.setLevel(Level.WARN)

  if (args.length != 4) {
    println("Usage: spark-submit --master yarn-client --class io.bittiger.movierating.hollywood.MovieRater " +
      "hollywood-*-SNAPSHOT-jar-with-dependencies.jar ratingsCsvHdfsPath movieCsvHdfsPath newUserProfileCsvHdfsPath idOfUserToRecommendMovies")
    sys.exit(1)
  }

  // set up environment
  val conf = new SparkConf().setAppName("MovieRater")
//    .set("spark.executor.memory", "2g")
  val sc = new SparkContext(conf)

  val ratingFilePath = args(0)
  val movieFilePath = args(1)
  val userRatingFilePath = args(2)
  val idOfUserToRecommend = args(3)
  println("========================== STARTING ==============================" )

  // load ratings and movie titles, and parse
  val ratingsData = sc.textFile(ratingFilePath)
  val ratingsRDD = removeCsvHeader(ratingsData).map(_.split(',') match { case Array(user, movie, rating, timestamp) =>
    Rating(user.toInt, movie.toInt, rating.toDouble)
  })
  ratingsRDD.cache

  val movieData = sc.textFile(movieFilePath)
  val movieRDD = removeCsvHeader(movieData).map( line => {
    val fields =  line.split(",")
    // format: (movieId, movieName)
    (fields(0).toInt, fields(1))
  }).collect.toMap

  val userRatingData = sc.textFile(userRatingFilePath)
  val userRatingRDD = removeCsvHeader(userRatingData).map(_.split(',') match { case Array(user, movie, rating) =>
    Rating(user.toInt, movie.toInt, rating.toDouble) })
  val userRatingSeq = userRatingRDD.collect.toSeq
  println("Current user <" + idOfUserToRecommend + "> profile:>>> \t" + userRatingSeq)

  val numRatings = ratingsRDD.count
  val numUsers = ratingsRDD.map(_.user).distinct.count
  val numMovies = ratingsRDD.map(_.product).distinct().count
  println("Got " + numRatings + " ratings from " + numUsers + " users on "
    + numMovies + " movies.")

  //split into 3 for training, validation, and testing
  val splitArr = ratingsRDD.randomSplit(Array(6, 2, 2), 0L)
  val trainingRDD = splitArr(0).union(userRatingRDD)
  val validationRDD = splitArr(1)
  val testRDD = splitArr(2)

  //persist frequently used RDDs
  trainingRDD.cache
  validationRDD.cache
  testRDD.cache

  val numTraining = trainingRDD.count
  val numValidation = validationRDD.count
  val numTest = testRDD.count
  println("Training: " + numTraining + ", validation: " + numValidation + ", test: " + numTest)

  // train models and evaluate them on the validation set
  println("=============>>>> Training... <<<<=================")
  val ranks = List(8, 12)
  val lambdas = List(0.1, 10.0)
  val numIters = List(10, 20)
  var bestModel: Option[MatrixFactorizationModel] = None
  var bestValidationRmse = Double.MaxValue
  var bestRank = 0
  var bestLambda = -1.0
  var bestNumIter = -1
  for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
    val model = ALS.train(trainingRDD, rank, numIter, lambda)
    val validationRmse = computeRmse(model, validationRDD, numValidation)
    println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
      + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
    if (validationRmse < bestValidationRmse) {
      bestModel = Some(model)
      bestValidationRmse = validationRmse
      bestRank = rank
      bestLambda = lambda
      bestNumIter = numIter
    }
  }

  // evaluate the best model on the test set
  println("=============>>>> Evaluating... <<<<=================")
  val testRmse = computeRmse(bestModel.get, testRDD, numTest)
  println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
    + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")

  // create a naive baseline and compare it with the best model
  println("=============>>>> Validating... <<<<=================")
  val meanRating = trainingRDD.union(validationRDD).map(_.rating).mean
  val baselineRmse =
    math.sqrt(testRDD.map(x => (meanRating - x.rating) * (meanRating - x.rating)).mean)
  val improvement = (baselineRmse - testRmse) / baselineRmse * 100
  println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

  // make personalized recommendations
  val myRatedMovieIds = userRatingSeq.map(_.product).toSet
  val candidates = sc.parallelize(movieRDD.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
  val recommendations = bestModel.get
    .predict(candidates.map((idOfUserToRecommend.toInt, _)))
    .collect()
    .sortBy(- _.rating)
    .take(50)
  var i = 1
  println("=============>>>>" + recommendations.length + " Movies recommended for User... <<<<=================")
  recommendations.foreach { r =>
    println("%2d".format(i) + ": " + movieRDD(r.product))
    i += 1
  }

  println("========================== COMPLETED ==============================" )


  /////////////////////////////// util functions ///////////////////////////

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }

  def removeCsvHeader(rdd: RDD[String]): RDD[String] = {
    rdd.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
  }

}
