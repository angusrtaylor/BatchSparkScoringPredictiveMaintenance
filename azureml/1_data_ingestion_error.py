
## Setup our environment by importing required libraries

# Github has been having some timeout issues. This should fix the problem for this dataset.
import socket
socket.setdefaulttimeout(90)

import glob
import os
# Read csv file from URL directly
import pandas as pd

import urllib
from datetime import datetime
# Setup the pyspark environment
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
args = parser.parse_args()

print("--data_dir ", args.data_dir)

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

basedataurl = "https://media.githubusercontent.com/media/Microsoft/SQL-Server-R-Services-Samples/master/PredictiveMaintenanceModelingGuide/Data/"

# We will store each of these data sets in DBFS.

# These file names detail which blob each files is stored under. 
MACH_DATA = 'machines_data'
MAINT_DATA = 'maint_data'
ERROR_DATA = 'errors_data'
TELEMETRY_DATA = 'telemetry_data'
FAILURE_DATA = 'failure_data'


print("spark.driver.memory: ", sc._conf.get('spark.driver.memory'))
print("spark.executor.instances : ", sc._conf.get('spark.executor.instances'))
print("spark.executor.cores: ", sc._conf.get('spark.executor.cores'))
print("spark.executor.memory: ", sc._conf.get('spark.executor.memory'))
print(" ")
print(sc._conf.getAll())


# load raw data from the GitHub URL
datafile = "telemetry.csv"

# Download the file once, and only once.
if not os.path.isfile(os.path.join(args.data_dir, datafile)):
    urllib.request.urlretrieve(basedataurl+datafile, os.path.join(args.data_dir, datafile))
    
# Read into pandas
telemetry = pd.read_csv(os.path.join(args.data_dir, datafile), encoding='utf-8')

# handle missing values
# define groups of features 
features_datetime = ['datetime']
features_categorical = ['machineID']
features_numeric = list(set(telemetry.columns) - set(features_datetime) - set(features_categorical))

# Replace numeric NA with 0
telemetry[features_numeric] = telemetry[features_numeric].fillna(0)

# Replace categorical NA with 'Unknown'
telemetry[features_categorical]  = telemetry[features_categorical].fillna("Unknown")

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
# This line takes about 9.5 minutes to run.
telemetry_spark = spark.createDataFrame(telemetry, verifySchema=False)

print(" ")
print("Writing dataframe")
import datetime
print(datetime.datetime.now().time())

# Write the telemetry data set to intermediate storage
telemetry_spark.write.mode('overwrite').parquet(os.path.join(args.data_dir, TELEMETRY_DATA))
