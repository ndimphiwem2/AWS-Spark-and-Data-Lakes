Project: Data Lake 

The purpose of the project is to build an ETL pipeline that extracts song_data and log_data from S3, processes them using Spark, and loads the data back into S3 as a set of dimensional tables. This will allow analytics team to continue finding insights in what songs their users are listening to.


Song DataSet


The first dataset is a subset of real data from the Million Song Dataset. Each file is in JSON format and contains metadata about a song and the artist of that song.

The files are partitioned by the first three letters of each song's track ID. For example, here are filepaths to two files in this dataset.

s3://udacity-dend/song_data/A/B/C/TRABCEI128F424C983.json
s3://udacity-dend/song_data/A/A/B/TRAABJL12903CDCF1A.json


Log dataset

The second dataset consists of log files in JSON format generated by this event simulator based on the songs in the dataset above. These simulate activity logs from a music streaming app based on specified configurations.

The log files in the dataset are partitioned by year and month. For example, here are filepaths to two files in this dataset.

s3://udacity-dend/log_data/2018/11/2018-11-12-events.json
s3://udacity-dend/log_data/2018/11/2018-11-13-events.json


How to run the project

Set environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.
Create an S3 bucket and replace the output_data variable in the main() function with s3a://<bucket name>/


Run ETL Pipeline

run it using the line below on terminal
    
 python etl.py
