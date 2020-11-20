# Repository for the Spark MOOC on Udacity
This is the repository for a MOOC on Udacity about Spark  
[Course Link](https://www.udacity.com/course/learn-spark-at-udacity--ud2002) |
[Spark Documentation](https://spark.apache.org/docs/latest/index.html) |
[Spark Download](https://spark.apache.org/downloads.html)


## Setup
1. Download and Install Spark
2. Install `pyspark` via pip:
    ```bash
    pip install pyspark
    ```
 3. ... or Anconda:
    ```bash
    conda install pyspark
    ```

## Spark Commands - How to start a Master Node (locally)
1. On your machine, navigate to:
    ```bash
    /usr/local/Cellar/apache-spark/2.4.5/libexec
    ```
2. Start the Master Node:
    ```bash
    ./sbin/start-master.sh -h <ip-address where to run>
    ```
3. Stop the Master Node:
    ```bash
    ./sbin/stop-master.sh
    ```


## Connect to an AWS EMR instance
[Documentation](https://docs.aws.amazon.com/emr/latest/ManagementGuide)

Connect to instance:
```bash
ssh -i <path>/<key_name>.pem hadoop@ec2-###-###-###-###.compute.amazonaws.com
```

## Transmit Files to HDFS

1. Connect to instance using SSH or Browser + Proxy

2. Transmit files to HDFS:
    ```bash
    scp -i <path>/<key_name>.pem ~/Desktop/sparkify_log_small.json hadoop@ec2-###-###-###-###.compute.amazonaws.com:~/
    ```
   
3. Create new HDFS Folder:
    ```bash
    hdfs dfs -mkdir user/<newFolder>
    ```
   
4. Need Help?
    ```bash
    hdfs #or hfds dfs
    ```
   
5. Move a file to the current cluster:
    ```bash
    hdfs dfs -copyFromLocal <file> /user/<folder>
    ```

6. Submit a script on hdfs with spark:
    ```bash
    which spark-submit = /usr/bin/spark-submit
    /usr/bin/spark-submit --master yarn ./<script>.py 
    ```


## Glossary

- Accumulators = global variables for debugging code
    ```python
  from pyspark import SparkContext
  errors = SparkContext.accumulator(0,0)
    ```