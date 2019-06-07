
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

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
args = parser.parse_args()

print("--data_dir ", args.data_dir)

spark = SparkSession.builder.getOrCreate()

basedataurl = "https://media.githubusercontent.com/media/Microsoft/SQL-Server-R-Services-Samples/master/PredictiveMaintenanceModelingGuide/Data/"

# We will store each of these data sets in DBFS.

# These file names detail which blob each files is stored under. 
MACH_DATA = 'machines_data'
MAINT_DATA = 'maint_data'
ERROR_DATA = 'errors_data'
TELEMETRY_DATA = 'telemetry_data'
FAILURE_DATA = 'failure_data'

# load raw data from the GitHub URL
datafile = "machines.csv"

# Download the file once, and only once.
if not os.path.isfile(os.path.join(args.data_dir, datafile)):
    urllib.request.urlretrieve(basedataurl+datafile, os.path.join(args.data_dir, datafile))
    
# Read into pandas
machines = pd.read_csv(os.path.join(args.data_dir, datafile))

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
mach_spark = spark.createDataFrame(machines, 
                                   verifySchema=False)

# Write the Machine data set to intermediate storage
mach_spark.write.mode('overwrite').parquet(os.path.join(args.data_dir, MACH_DATA))

# load raw data from the GitHub URL
datafile = "errors.csv"

# Download the file once, and only once.
if not os.path.isfile(os.path.join(args.data_dir, datafile)):
    urllib.request.urlretrieve(basedataurl+datafile, os.path.join(args.data_dir, datafile))
    
# Read into pandas
errors = pd.read_csv(os.path.join(args.data_dir, datafile), encoding='utf-8')

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
error_spark = spark.createDataFrame(errors, 
                               verifySchema=False)

# Write the Errors data set to intermediate storage
error_spark.write.mode('overwrite').parquet(os.path.join(args.data_dir, ERROR_DATA))

# load raw data from the GitHub URL
datafile = "maint.csv"

# Download the file once, and only once.
if not os.path.isfile(os.path.join(args.data_dir, datafile)):
    urllib.request.urlretrieve(basedataurl+datafile, os.path.join(args.data_dir, datafile))
    
# Read into pandas
maint = pd.read_csv(os.path.join(args.data_dir, datafile), encoding='utf-8')

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
maint_spark = spark.createDataFrame(maint, 
                              verifySchema=False)

# Write the Maintenance data set to intermediate storage
maint_spark.write.mode('overwrite').parquet(os.path.join(args.data_dir, MAINT_DATA))

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

# Write the telemetry data set to intermediate storage
telemetry_spark.write.mode('overwrite').parquet(os.path.join(args.data_dir, TELEMETRY_DATA))

# load raw data from the GitHub URL
datafile = "failures.csv"

# Download the file once, and only once.
if not os.path.isfile(os.path.join(args.data_dir, datafile)):
    urllib.request.urlretrieve(basedataurl+datafile, os.path.join(args.data_dir, datafile))
    
# Read into pandas
failures = pd.read_csv(os.path.join(args.data_dir, datafile), encoding='utf-8')

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
failures_spark = spark.createDataFrame(failures, 
                                       verifySchema=False)

# Write the failures data set to intermediate storage
failures_spark.write.mode('overwrite').parquet(os.path.join(args.data_dir, FAILURE_DATA))