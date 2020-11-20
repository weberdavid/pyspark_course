from pyspark.sql import SparkSession
import datetime

spark = SparkSession \
    .builder \
    .appName("Data wrangling with Spark SQL") \
    .getOrCreate()

# spark.stop() to end context

path = "data/sparkify_log_small.json"
user_log = spark.read.json(path)

user_log.printSchema()

# Same stuff as before, but can be done with sql as well (telling spark what to do instead of how)

# create a temporary view to run sql queries
user_log.createOrReplaceTempView("user_data_table")

# create queries
spark.sql('''
            SELECT *
            FROM user_data_table
            LIMIT 2
            ''').show()  # .show is required to surpass lazyevaluation of spark

spark.sql('''
            SELECT userId, count(page)
            FROM user_data_table
            GROUP BY userId
            ''').show()

spark.sql('''
            SELECT userId, firstname, page, song
            FROM user_data_table
            WHERE userId = '1046'
            ''').collect()  # Attention - difference between show and collect


# User Defined Functions UDF
# must be registered first
spark.udf.register("get_hour", lambda x: int(datetime.datetime.fromtimestamp(x / 1000).hour))

spark.sql('''
            SELECT userId, AVG(get_hour(ts)) as avg_hour
            FROM user_data_table
            GROUP BY userId
            ''').show()

songs_per_hour = spark.sql('''
                            SELECT get_hour(ts) as hour, count(song) as SongCounts
                            FROM user_data_table
                            WHERE page = "NextSong"
                            GROUP BY hour
                            ORDER BY cast(hour as int) ASC
                            ''')
songs_per_hour.show()


# Converting results to pandas

songs_per_hour_pd = songs_per_hour.toPandas()
print(songs_per_hour_pd)


# QUIZ

# Q1 Which page did user "" not visit?
spark.sql('''
            SELECT page
            FROM user_data_table
            WHERE userId = ""
            GROUP BY page
            ''').show()

# Q3 How many female users in the dataset?
spark.sql('''
            SELECT count(distinct(userId))
            FROM user_data_table
            WHERE gender = 'F'
            ''').show()

# Q4 How many songs were played from the most played artist?
spark.sql('''
            SELECT artist, count(artist) as count
            FROM user_data_table
            WHERE page = 'NextSong'
            GROUP BY artist
            ORDER BY count DESC
            LIMIT 1
            ''').show()
