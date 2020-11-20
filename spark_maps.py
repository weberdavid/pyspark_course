import findspark
from pyspark import SparkContext, SparkConf

findspark.init()

configure = SparkConf().setAppName("spark_maps_and_lazy_eval").setMaster("local")

sc = SparkContext(conf=configure)

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

distributed_song_log = sc.parallelize(log_of_songs)
print(distributed_song_log)


def convert_songs_to_lowercase(song):
    return song.lower()


convert_songs_to_lowercase(log_of_songs[1])

# calculating results
result = distributed_song_log.map(convert_songs_to_lowercase).collect()

# proof of lazy eval
distributed_song_log.collect()

# writing an anonymous function for it
distributed_song_log.map(lambda song: song.lower()).collect()

print(result)
