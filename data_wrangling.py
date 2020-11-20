from pyspark.sql import SparkSession
from pyspark.sql import functions as fct
from pyspark.sql import Window
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ss = SparkSession \
    .builder \
    .appName("wrangling_with_data") \
    .getOrCreate()

path = "data/sparkify_log_small.json"
user_data = ss.read.json(path)

# Explore Data
user_data.printSchema()
user_data.describe().show()  # describe = variable + datatype; takes variable as parameter
                             # show = count, mean, stddev, min, max for each var
# counts number of users
user_data.count()

# selects variable and drops duplicates
user_data.select("page").dropDuplicates().sort(fct.desc("page")).show()


# Select specific variables from a single user
user_data.select(["userID", "firstname", "lastname", "level"]).where(user_data.userId == "1046").collect()

# calculating stuff udf = user defined function
get_hour = fct.udf(lambda x: dt.datetime.fromtimestamp(x / 1000.0). hour)

# create new column "hour" and fill with calculated hour of timestamp
user_data = user_data.withColumn("hour", get_hour(user_data.ts))
# show head - returns first row?
user_data.head()

# How many songs are listened per hour?
songs_in_hour = user_data.filter(user_data.page == "NextSong").groupby(user_data.hour)\
    .count().orderBy(user_data.hour.cast("float"))
songs_in_hour.show()

# convert to pandas df for visualization
songs_in_hour_pd = songs_in_hour.toPandas()
# convert hour to numeric
songs_in_hour_pd.hour = pd.to_numeric(songs_in_hour_pd.hour)

# do the scatterplot
plt.scatter(songs_in_hour_pd["hour"], songs_in_hour_pd["count"])
plt.xlim(-1, 24);
plt.ylim(0, 1.2 * max(songs_in_hour_pd["count"]))
plt.xlabel("Hour")
plt.ylabel("Count of Songs Played");


# look at missing values: only NAs in userId and or sessionId
valid_users = user_data.dropna(how = "any", subset = ["userId", "sessionId"])
valid_users.count()

# drop duplicates: drop Dup, sort after User ID
valid_users.select("userId").dropDuplicates().sort("userId").show()

# drop empty strings
valid_users = valid_users.filter(valid_users["userId"] != "")
valid_users.count()


# what about users downgrading accounts
valid_users.filter(valid_users["page"] == "Submit Downgrade").show()

# give downgraders a flag; first create function; second give flag to all users
flag_downgrade_event = fct.udf(lambda x: 1 if x == "Submit Downgrade" else 0, IntegerType())
# withColumn = creates new column, or takes current and pastes values in
valid_users = valid_users.withColumn("downgraded", flag_downgrade_event("page"))

# work with window
windowval = Window.partitionBy("userId").orderBy(fct.desc("ts")).rangeBetween(Window.unboundedPreceding, 0)
# new column phase, that is the sum of downgraded
valid_users = valid_users.withColumn("phase", fct.sum("downgraded").over(windowval))
# select variables of user and sort and collect data?
valid_users.select(["userId", "firstname", "ts", "page", "level", "phase"])\
    .where(user_data.userId == "1138").sort("ts").collect()


# Answering the quiz
# Which page did user id "" NOT visit?
user_data.where(user_data["userId"] == "").groupby(user_data.page).count().show()

# How many female users in the dataset?
user_data.select(["userId"]).dropDuplicates().where(user_data["gender"] == "F").count()

# How many songs were played, from the most played artist?
user_data.filter(user_data.page == "NextSong")\
    .select("Artist")\
    .groupby("Artist")\
    .count()\
    .sort(fct.desc("count"))\
    .show(1)
