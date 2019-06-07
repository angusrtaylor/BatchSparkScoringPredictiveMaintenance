
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
datafile = os.path.join(args.data_dir, "machines.csv")

# Download the file once, and only once.
if not os.path.isfile(datafile):
    urllib.request.urlretrieve(basedataurl+datafile, datafile)
    
os.listdir(os.path.join(args.data_dir))