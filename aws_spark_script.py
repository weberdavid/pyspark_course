from pyspark.sql import SparkSession

if __name__ == "__main__":
    """
        example program to show how to submit sth
    """

    spark = SparkSession \
        .builder \
        .appName("LowerTheSongTitles") \
        .getOrCreate()

    log_of_songs = [
        "Despacito",
        "Nice for what",
        "No tears left to cry",
        "Despacito",
        "Havana",
        "In my feelings",
        "Nice for what",
        "despacito",
        "All the stars"
    ]

    distributed_song_log = spark.SparkContext.parallelize(log_of_songs)

    print(distributed_song_log.map(lambda x: x.lower()).collect())

    incorrect_records = spark.SparkContext.accumulator(0, 0)
    print(incorrect_records.value)

    # define function, where incorrect_record increases for every corrupt record
    def add_incorrect_record():
        global incorrect_records
        incorrect_records += 1


    # define what corrupt record means and use the function to increase the count
    # output: collect() data and then look at incorrect_records.value

    spark.stop()
