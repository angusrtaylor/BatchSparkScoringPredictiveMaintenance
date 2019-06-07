
# import the libraries
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
# for creating pipelines and model
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--training_table', type=str, default='training_data')
parser.add_argument('--model', type=str, default='RandomForest')
args = parser.parse_args()

train_data = spark.read.parquet(os.path.join(args.data_dir, args.training_table)).cache()


# define list of input columns for downstream modeling

# We'll use the known label, and key variables.
label_var = ['label_e']
key_cols =['machineID','dt_truncated']

# Then get the remaing feature names from the data
input_features = train_data.columns

# We'll use the known label, key variables and 
# a few extra columns we won't need.
remove_names = label_var + key_cols + ['failure','model_encoded','model' ]

# Remove the extra names if that are in the input_features list
input_features = [x for x in input_features if x not in set(remove_names)]

# COMMAND ----------

# MAGIC %md Spark models require a vectorized data frame. We transform the dataset here and then split the data into a training and test set. We use this split data to train the model on 9 months of data (training data), and evaluate on the remaining 3 months (test data) going forward.

# COMMAND ----------

# assemble features
va = VectorAssembler(inputCols=(input_features), outputCol='features')
train_data = va.transform(train_data).select('machineID','dt_truncated','label_e','features')

# set maxCategories so features with > 10 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", 
                               outputCol="indexedFeatures", 
                               maxCategories=10).fit(train_data)

# fit on whole dataset to include all labels in index
labelIndexer = StringIndexer(inputCol="label_e", outputCol="indexedLabel").fit(train_data)

training = train_data

model_type = args.model

# train a model.
if model_type == 'DecisionTree':
  model = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
                                 # Maximum depth of the tree. (>= 0) 
                                 # E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes.'
                                 maxDepth=15,
                                 # Max number of bins for discretizing continuous features. 
                                 # Must be >=2 and >= number of categories for any categorical feature.
                                 maxBins=32, 
                                 # Minimum number of instances each child must have after split. 
                                 # If a split causes the left or right child to have fewer than 
                                 # minInstancesPerNode, the split will be discarded as invalid. Should be >= 1.
                                 minInstancesPerNode=1, 
                                 # Minimum information gain for a split to be considered at a tree node.
                                 minInfoGain=0.0, 
                                 # Criterion used for information gain calculation (case-insensitive). 
                                 # Supported options: entropy, gini')
                                 impurity="gini")

  ##=======================================================================================================================
  ## GBTClassifer is only valid for Binary Classifiers, this is a multiclass (failures 1-4) so no GBTClassifier
#elif model_type == 'GBTClassifier':
#  model = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures",
#                        maxIter=200, stepSize=0.1,
#                        maxDepth=15,
#                        maxBins=32, 
#                        minInstancesPerNode=1, 
#                        minInfoGain=0.0)
  ##=======================================================================================================================
else:
  model = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", 
                                      # Passed to DecisionTreeClassifier
                                      maxDepth=15, 
                                      maxBins=32, 
                                      minInstancesPerNode=1, 
                                      minInfoGain=0.0,
                                      impurity="gini",
                                      # Number of trees to train (>= 1)
                                      numTrees=200, 
                                      # The number of features to consider for splits at each tree node. 
                                      # Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n].
                                      featureSubsetStrategy="sqrt", 
                                      # Fraction of the training data used for learning each  
                                      # decision tree, in range (0, 1].' 
                                      subsamplingRate = 0.632)

# chain indexers and model in a Pipeline
pipeline_cls_mthd = Pipeline(stages=[labelIndexer, featureIndexer, model])

# train model.  This also runs the indexers.
model_pipeline = pipeline_cls_mthd.fit(training)

# save model
model_pipeline.write().overwrite().save(os.path.join(args.data_dir, 'models', model_type + ".pqt"))