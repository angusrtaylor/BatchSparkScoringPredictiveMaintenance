
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
datafilepath = os.path.join(args.data_dir, datafile)

# Download the file once, and only once.
#if not os.path.isfile(datafilepath):
#    urllib.request.urlretrieve(basedataurl+datafile, datafilepath)
    
os.listdir(args.data_dir)

# Read into pandas
machines = pd.read_csv(datafilepath)

# The data was read in using a Pandas data frame. We'll convert 
# it to pyspark to ensure it is in a Spark usable form for later 
# manipulations.
mach_spark = spark.createDataFrame(machines, 
                                   verifySchema=False)

# Write the Machine data set to intermediate storage
#mach_spark.write.mode('overwrite').saveAsTable(MACH_DATA)
#mach_spark.write.option("path", args.data_dir).mode('overwrite').saveAsTable(MACH_DATA)
#mach_spark.write.mode('overwrite').parquet(os.path.join(args.data_dir, MACH_DATA))

#machines = spark.table(MACH_DATA).cache()
#machines = spark.read.option("path", args.data_dir).table(MACH_DATA).cache()
machines = spark.read.parquet(os.path.join(args.data_dir, MACH_DATA))

print(machines.head(1))