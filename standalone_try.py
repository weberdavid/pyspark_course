import findspark
import pyspark

findspark.init()

sc = pyspark.SparkContext(master="spark://192.168.178.45:7077",
                          appName="standalone_try")
# sc.stop() to end context

print(sc)
















