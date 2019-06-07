
# import the libraries
from pyspark.ml import PipelineModel
# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--scoring_table', type=str, default='scoring_data')
parser.add_argument('--model', type=str, default='RandomForest')
parser.add_argument('--results_data', type=str, default='results_output')
args = parser.parse_args()

score_data = spark.read.parquet(os.path.join(args.data_dir, args.scoring_table)).cache()

# We'll use the known label, and key variables.
label_var = ['label_e']
key_cols =['machineID','dt_truncated']

# Then get the remaing feature names from the data
input_features = score_data.columns

# We'll use the known label, key variables and 
# a few extra columns we won't need.
remove_names = label_var + key_cols + ['failure','model_encoded','model' ]

# Remove the extra names if that are in the input_features list
input_features = [x for x in input_features if x not in set(remove_names)]

input_features
# assemble features
va = VectorAssembler(inputCols=(input_features), outputCol='features')

# assemble features
score_data = va.transform(score_data).select('machineID','dt_truncated','label_e','features')

# set maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", 
                               outputCol="indexedFeatures", 
                               maxCategories=10).fit(score_data)

# Load the model from local storage
model_pipeline = PipelineModel.load(os.path.join(args.data_dir, 'models', args.model + ".pqt"))


# score the data. The Pipeline does all the same operations on this dataset
predictions = model_pipeline.transform(score_data)

#write results to data store for persistance.
predictions.write.mode('overwrite').parquet(os.path.join(args.data_dir, args.results_data))