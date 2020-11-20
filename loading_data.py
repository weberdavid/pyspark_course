from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("first data example") \
    .getOrCreate()

spark.sparkContext.getConf().getAll()
print(spark)

# load the csv file
path = "data/sparkify_log_small.json"
user_log = spark.read.json(path)

# print file schema
user_log.printSchema()

# same but different
user_log.describe()

# which row to show
user_log.show(n=1)

# how many to display
user_log.take(2)

# save as csv
out_path = "data/sparkify_log_small.csv"
user_log.write.save(out_path, format = "csv", header = True)

# read csv
user_log_csv = spark.read.csv(out_path, header = True)

# print csv schema (same as json)
user_log_csv.printSchema()

# print the user ID (showing top 20 rows)
user_log_csv.select("userID").show()


