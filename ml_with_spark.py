from pyspark.sql import SparkSession
from pyspark.ml.feature import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, concat
from pyspark.sql.types import IntegerType
import regex as re
# SPARK only takes numerical variables for ML

spark = SparkSession \
    .builder \
    .appName("ml_with_spark") \
    .getOrCreate()

# read data; dataset is from https://stackoverflow.blog/2009/06/04/stack-overflow-creative-commons-data-dump/
sto_data = spark.read.json("data/train_onetag_small.json")  # sto = stack overflow
sto_data.head()


# Tokenization: Splitting strings into separate words: https://spark.apache.org/docs/latest/ml-features.html#tokenizer
regexTokenizer = RegexTokenizer(inputCol="Body", outputCol="words", pattern="\\W")
sto_data = regexTokenizer.transform(sto_data)   # new column, where split words are saved as list


# count number of words for each body
body_length = udf(lambda x: len(x), IntegerType())
sto_data = sto_data.withColumn("BodyLength", body_length(sto_data.words))


# count the number of paragraphs and links in each body
no_para = udf(lambda x: len(re.findall("</p>", x)), IntegerType())
no_links = udf(lambda x: len(re.findall("</a>", x)), IntegerType())

sto_data = sto_data.withColumn("NumParagraphs", no_para(sto_data.Body))
sto_data = sto_data.withColumn("NumLinks", no_links(sto_data.Body))
print(sto_data.take(2))  # display 2 rows, nicer format than .head()


# Vector Assembler: Taking three columns into a Vector, prerequisite for normalization of numeric features
assembler = VectorAssembler(inputCols=["BodyLength", "NumParagraphs", "NumLinks"], outputCol="NumFeatures")
sto_data = assembler.transform(sto_data)
print(sto_data.take(2))

# Normalization of Vectors
scaler = Normalizer(inputCol="NumFeatures", outputCol="ScaledNumFeatures")
sto_data = scaler.transform(sto_data)
print(sto_data.take(2))

# Scale
scaler = StandardScaler(inputCol="NumFeatures", outputCol="ScaledNumFeatures_scaler")
scalerModel = scaler.fit(sto_data)
sto_data = scalerModel.transform(sto_data)
print(sto_data.take(2))


# PART 2: Further Feature Engineering
# Count Vectorizer
# Says, how often a particular word appears and creates a vocabulary
cv = CountVectorizer(inputCol="words", outputCol="TF", vocabSize=1000)
cvmodel = cv.fit(sto_data)
sto_data = cvmodel.transform(sto_data)
print(sto_data.take(2))

# show vocabulary
print(cvmodel.vocabulary)
print(cvmodel.vocabulary[-10:])


# Inter-document Frequency: puts absolute word numbers from before as relative numbers within the dataset
idf = IDF(inputCol="TF", outputCol="TFIDF")
idfmodel = idf.fit(sto_data)
sto_data = idfmodel.transform(sto_data)
print(sto_data.take(2))

# StringIndexer: takes a string and gives it an index - so that it is numerical
indexer = StringIndexer(inputCol="oneTag", outputCol="label")
indexermodel = indexer.fit(sto_data)
sto_data = indexermodel.transform(sto_data)
print(sto_data.take(2))


# QUIZ
# Q1: Question Id = 1112; How many words does the body contain?
q1112 = sto_data.select(["Id", "BodyLength"]).where(sto_data.Id == 1112)
print(q1112.show())

# Q2: Create a new column, that concatenates question title and body; apply the function that counts the number of words
# in this column. Whats the value in this column for qId: 5123?
sto_data = sto_data.withColumn("TitleBody", concat("Title", "Body"))
regexTokenizer = RegexTokenizer(inputCol="TitleBody", outputCol="TitleBodyWords", pattern="\\W")
sto_data = regexTokenizer.transform(sto_data)   # new column, where split words are saved as list

no_title_body = udf(lambda x: len(x))
sto_data = sto_data.withColumn("TitleBodyCount", no_title_body(sto_data.TitleBodyWords))

q5123 = sto_data.select(["Id", "TitleBodyWords", "TitleBodyCount"]).where(sto_data.Id == 5123)
print(q5123.show())

# Q3: Using normalizer, whats the normalized value for qId: 512?
sto_data = sto_data.withColumn("TitleBodyCount", sto_data.TitleBodyCount.cast(IntegerType()))
assembler = VectorAssembler(inputCols=["TitleBodyCount"], outputCol="TitleBodyVector")
sto_data = assembler.transform(sto_data)

scaler = Normalizer(inputCol="TitleBodyVector", outputCol="TitleBodyNormalizer")
sto_data = scaler.transform(sto_data)

q512 = sto_data.select(["Id", "TitleBodyNormalizer"]).where(sto_data.Id == 512)
print(q512.show())

# Q4: Using the StandardScaler (mean and std), whats the normalized value for qId: 512?
sto_data = sto_data.drop("TitleBodyScaler")
scaler = StandardScaler(inputCol="TitleBodyVector", outputCol="TitleBodyScaler", withStd=True, withMean=True)
scalerModel = scaler.fit(sto_data)
sto_data = scalerModel.transform(sto_data)

q512 = sto_data.select(["Id", "TitleBodyScaler"]).where(sto_data.Id == 512)
print(q512.show())

# Q5: Using MinMaxScaler, whats the normalized value for qId: 512?
scaler = MinMaxScaler(inputCol="TitleBodyVector", outputCol="TitleBodyMinMaxScaler")
scalerModel = scaler.fit(sto_data)
sto_data = scalerModel.transform(sto_data)

q512 = sto_data.select(["Id", "TitleBodyMinMaxScaler"]).where(sto_data.Id == 512)
print(q512.show())


# LINEAR REGRESSION
lr = LinearRegression(maxIter=5, regParam=0.0, fitIntercept=False, solver="normal")
train_data = sto_data.select(col("NumParagraphs").alias("label"), col("TitleBodyVector").alias("features"))
lrModel = lr.fit(train_data)
lrModel.coefficients
lrModel.intercept
lrModel.summary
lrModel.summary.r2

# LOGISTIC REGRESSION
train_data = sto_data.select(col("label").alias("label"), col("TFIDF").alias("features"))
lr = LogisticRegression(maxIter=10, regParam=0.0)
lrModel = lr.fit(train_data)
lrModel.coefficientMatrix
lrModel.summary.accuracy
lrModel.interceptVector

# K-MEANS CLUSTERING
train_data = sto_data.select(col("TitleBodyCount").alias("feature"))
assembler = VectorAssembler(inputCols=["feature"], outputCol="features")
train_data = assembler.transform(train_data)
train_data = train_data.drop("feature")
kmeans = KMeans().setK(5).setSeed(42)
model = kmeans.fit(train_data)

centers = model.clusterCenters()
for center in centers:
    print(center)

# OTHER QUESTIONS:
# Q1: How many times greater is the TitleBodyCount of the longest question than the TitleBodyCount
# of the shortest question (rounded to the nearest whole number)?
sto_data.createOrReplaceTempView("sto_data")
spark.sql("""
            select max(TitleBodyCount) / min(TitleBodyCount)
            FROM sto_data
        """).show()

# Q2: What is the mean and standard deviation of the TitleBodyCount?
# create a temporary view to run sql queries

spark.sql("""
            select mean(TitleBodyCount), std(TitleBodyCount)
            FROM sto_data
        """).show()


# PIPELINES: https://spark.apache.org/docs/latest/ml-pipeline.html
idf = IDF(inputCol="TF", outputCol="features")
idfmodel = idf.fit(sto_data)
sto_data = idfmodel.transform(sto_data)
lr = LogisticRegression(maxIter=10, regParam=0.0, elasticNetParam=0)
pipeline = Pipeline(stages=[regexTokenizer, cv, idf, indexer, lr])
regModel = pipeline.fit(sto_data)

sto_data.filter(sto_data.label == sto_data.prediction).count()  # gives number of accurately predicted labels on train


# MODEL SELECTION & TUNING
train, test = sto_data.randomSplit([0.6, 0.4], seed=42)
test, validation = test.randomSplit([0.5, 0.5], seed=42)

regModel = pipeline.fit(train)
results = regModel.transform(test)
results.filter(results.label == results.prediction).count()    # gives number of accurately predicted labels on test set

# crossvalidation
paramGrid = ParamGridBuilder() \
    .addGrid(cv.vocabSize, [10000, 20000]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=3)

cvModel = crossval.fit(train)
cvModel.avgMetrics
results = cvModel.transform(test)
results.filter(results.label == results.prediction).count()    # gives number of accurately predicted labels on test set


spark.stop()





