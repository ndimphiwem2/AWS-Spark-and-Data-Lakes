import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType as R, StructField as Fld, \
     DoubleType as Dbl, LongType as Long, StringType as Str, \
     IntegerType as Int, DecimalType as Dec, DateType as Date, \
     TimestampType as Stamp

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    
    """this function creates a spark session and will
    retun the spark session object"""
    
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    
    """This function extract the song data stored as a Json file
    from AWS S3 bucket and process the song data by extracting
    the columns to create songs table and artist table
    then write those tables to parquet files and load them back to
    AWS S3 bucket.
    
    arg1: spark: spark session object
    arg2: input_data: S3 bucket with input data
    arg3: output_data: S3 bucket for output data
    """
    # get filepath to song data file
    song_data = input_data + "song_data/A/A/A/*.json"
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id', 'title', 'artist_id', 'year', 'duration']).distinct()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(output_data+'songs/'+'songs.parquet', partitionBy=['year','artist_id'])

    # extract columns to create artists table
    artists_table = df.select(["artist_id", "artist_name", "artist_location", "artist_latitude", "artist_longitude"]).distinct()
    
    # write artists table to parquet files
    artists_table.write.parquet(output_data + 'artists/' + 'artists.parquet', partitionBy=['artist_id'] )


def process_log_data(spark, input_data, output_data):
    
    """This function extract the log data stored as a Json file
    from AWS S3 bucket and process the log data by extracting
    the columns to create user table and time table
    then write those tables to parquet files and load them back to
    AWS S3 bucket.
    
    arg1: spark: spark session object
    arg2: input_data: S3 bucket with input data
    arg3: output_data: S3 bucket for output data
    """
    # get filepath to log data file
    log_data = input_data + "log_data/*/*/*.json"

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where('page="NextSong"')

    # extract columns for users table    
    artists_table = df.select(["userId", "firstName", "lastName", "gender", "level"]).distinct()
    
    # write users table to parquet files
    artists_table.write.parquet(output_data + 'users/' + 'users.parquet', partitionBy = ['userId'])

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp((x / 1000)), Stamp())
    df = df.withColumn("timestamp", get_timestamp(col("ts")))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp((x / 1000)), Stamp())
    df = df.withColumn("datetime", get_datetime(col("ts")))
    
    # extract columns to create time table
    time_table = df.selectExpr("timestamp as start_time",
                               "hour(timestamp) as hour",
                               "dayofmonth(timestamp) as day",
                               "weekofyear(timestamp) as week",
                               "month(timestamp) as month",
                               "year(timestamp) as year",
                               "dayofweek(timestamp) as weekday"
                               ).dropDuplicates(["start_time"])
    
    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(output_data + "time_table.parquet",
                             partitionBy = ["year", "month"],
                             mode = "overwrite")

    # read in song data to use for songplays table
    song_data = input_data + "song_data/A/A/A/*.json"
    song_df = spark.read.json(song_data)

    # extract columns from joined song and log datasets to create songplays table 
    song_df.createOrReplaceTempView("song_data")
    df.createOrReplaceTempView("log_data")
    
    songplays_table = spark.sql("""
                                SELECT monotonically_increasing_id() as songplay_id,
                                ld.timestamp as start_time,
                                year(ld.timestamp) as year,
                                month(ld.timestamp) as month,
                                ld.userId as user_id,
                                ld.level as level,
                                sd.song_id as song_id,
                                sd.artist_id as artist_id,
                                ld.sessionId as session_id,
                                ld.location as location,
                                ld.userAgent as user_agent
                                FROM log_data ld
                                JOIN song_data sd
                                ON (ld.song = sd.title
                                AND ld.length = sd.duration
                                AND ld.artist = sd.artist_name)
                                """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(output_data + 'songplays/' + 'songplays.parquet',partitionBy=['year', 'month'])


def main():
    
    """ Run the ETL to process the song_data and the log_data files"""
    
    spark = create_spark_session()
    
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://ndimbucket-data-lake/"
    
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
