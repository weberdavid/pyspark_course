from pyspark.sql import SparkSession
from pyspark.ml.feature import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

spark = SparkSession \
    .builder \
    .appName("quiz_modeltuning") \
    .getOrCreate()


# read data; dataset is from https://stackoverflow.blog/2009/06/04/stack-overflow-creative-commons-data-dump/
sto_data = spark.read.json("data/train_onetag_small.json")  # sto = stack overflow


# create train test split
train, test = sto_data.randomSplit([90.0, 10.0], seed=42)

regexTokenizer = RegexTokenizer(inputCol="Body", outputCol="words", pattern="\\W")
sto_data = regexTokenizer.transform(sto_data)   # new column, where split words are saved as list

cv = CountVectorizer(inputCol="words", outputCol="TF", vocabSize=1000)
cvmodel = cv.fit(sto_data)
sto_data = cvmodel.transform(sto_data)

idf = IDF(inputCol="TF", outputCol="features")
idfmodel = idf.fit(sto_data)
sto_data = idfmodel.transform(sto_data)

indexer = StringIndexer(inputCol="oneTag", outputCol="label")
indexermodel = indexer.fit(sto_data)
sto_data = indexermodel.transform(sto_data)

lr = LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)


pipeline = Pipeline(stages=[regexTokenizer, cv, idf, indexer, lr])


# cross validation
paramGrid = ParamGridBuilder() \
    .addGrid(cv.vocabSize, [1000, 5000]) \
    .addGrid(lr.regParam, [0.0, 0.1]) \
    .addGrid(lr.maxIter, [10.0]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)

# train = train.drop(train.words)
cvModel = crossval.fit(train)
cvModel.avgMetrics
results = cvModel.transform(test)
results.filter(results.label == results.prediction).count()    # gives number of accurately predicted labels on test set
results.count()

spark.stop()
