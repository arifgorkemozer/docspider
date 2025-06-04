# python3 benchmark_pipeline.py <experiment_tag>
# add --skip_execution_step for skipping running nosql queries

import argparse
import json
import os
import pandas
import shutil

from common import *
from bp_step1_metadata import generate_collection_metadata
from bp_step2_run_queries import run_predicted_queries
from bp_step3_compare_results import compare_expected_predicted_query_results

parser = argparse.ArgumentParser()
parser.add_argument("experiment_tag", help="Tag for the experiment")
parser.add_argument("--skip_execution_step", action="store_true", help="Skip running generated queries or not")
args = parser.parse_args()

# generic definitions
SKIP_QUERY_EXECUTION = args.skip_execution_step

# generic paths
ROOT_PATH = ".." # parent directory
DOCSPIDER_DATASET_PATH = f"{ROOT_PATH}/docspider_ground_truth_dataset"
COLLECTIONS_JSON_PATH = f"{DOCSPIDER_DATASET_PATH}/collections.json"
DEV_DATASET_JSON_PATH = f"{DOCSPIDER_DATASET_PATH}/dev.json"
DEV_GOLD_MONGODB_QUERY_FILENAME = "dev_gold.tsv"
DEV_GOLD_MONGODB_QUERY_PATH = f"{DOCSPIDER_DATASET_PATH}/{DEV_GOLD_MONGODB_QUERY_FILENAME}"
ALL_EXPERIMENTS_ROOT_PATH = f"{ROOT_PATH}/experiments"
METADATA_FILENAME = "docspider_collection_column_metadata.json"
METADATA_PATH = f"{ROOT_PATH}/{METADATA_FILENAME}"

# experiment specific paths
EXPERIMENT_TAG = args.experiment_tag
EXPERIMENT_PATH = f"{ALL_EXPERIMENTS_ROOT_PATH}/{EXPERIMENT_TAG}"
PREDICTED_MONGODB_QUERY_FILENAME = "predicted_nosql.tsv"
PREDICTED_MONGODB_QUERY_PATH = f"{EXPERIMENT_PATH}/{PREDICTED_MONGODB_QUERY_FILENAME}"
EXPECTED_QUERY_RESULTS_FOLDERNAME = "expected_results"
ACHIEVED_QUERY_RESULTS_FOLDERNAME = "achieved_results"
EXPECTED_QUERY_RESULTS_PATH = f"{EXPERIMENT_PATH}/{EXPECTED_QUERY_RESULTS_FOLDERNAME}"
ACHIEVED_QUERY_RESULTS_PATH = f"{EXPERIMENT_PATH}/{ACHIEVED_QUERY_RESULTS_FOLDERNAME}"

# read query hardness
hardness_map = {}

dev_json_file = open(DEV_DATASET_JSON_PATH, "r")
dev_json = json.load(dev_json_file)

for i, dev_json_entry in enumerate(dev_json):
    hardness_map[dev_json_entry["question_id"]] = dev_json_entry["difficulty"]

######################### BEGIN - CREATE FILES AND FOLDERS NEEDED #########################
items_under_main_folder = os.listdir(ROOT_PATH)
items_under_experiment = os.listdir(EXPERIMENT_PATH)

# create results folders if needed
if EXPECTED_QUERY_RESULTS_FOLDERNAME not in items_under_experiment:
    os.mkdir(EXPECTED_QUERY_RESULTS_PATH)

if ACHIEVED_QUERY_RESULTS_FOLDERNAME not in items_under_experiment:
    os.mkdir(ACHIEVED_QUERY_RESULTS_PATH)

# copy gold query file if needed
if DEV_GOLD_MONGODB_QUERY_FILENAME not in items_under_experiment:
    shutil.copy(DEV_GOLD_MONGODB_QUERY_PATH, EXPERIMENT_PATH)

# read gold queries
gold_nosql_file = pandas.read_csv(f"{EXPERIMENT_PATH}/{DEV_GOLD_MONGODB_QUERY_FILENAME}", sep=TAB_CHAR)
gold_nosql_map = {}

for i, row in gold_nosql_file.iterrows():
    query_index = i+1
    query = row["query"]
    db = row["db"]
    gold_nosql_map[query_index] = (query, db)

# STEP 1 - GENERATE TABLE COLUMN METADATA
if METADATA_FILENAME in items_under_main_folder:
    with open(METADATA_PATH, "r") as inp:
        metadata = json.load(inp)
else:
    metadata = generate_collection_metadata(collections_json_path=COLLECTIONS_JSON_PATH, metadata_output_path=METADATA_PATH)

# STEP 2 - RUN GOLD NOSQL QUERIES AND LLM-GENERATED QUERIES
if not SKIP_QUERY_EXECUTION:
    run_predicted_queries(
        predictions_output_path=PREDICTED_MONGODB_QUERY_PATH,
        expected_query_results_path=EXPECTED_QUERY_RESULTS_PATH,
        achieved_query_results_path=ACHIEVED_QUERY_RESULTS_PATH,
        gold_nosql_map=gold_nosql_map
    )

# STEP 3 - COMPARE QUERY RESULTS OF GOLD MONGODB QUERIES AND LLM-GENERATED QUERIES
comparison_output_file = open(f"comparison_results_{EXPERIMENT_TAG}.txt", "w")

for sensitivity in SENSITIVITY_OPTIONS:
    compare_expected_predicted_query_results(
        current_sensitivity=sensitivity,
        experiment_tag=EXPERIMENT_TAG,
        experiment_path=EXPERIMENT_PATH,
        predictions_output_path=PREDICTED_MONGODB_QUERY_PATH,
        expected_query_results_path=EXPECTED_QUERY_RESULTS_PATH,
        achieved_query_results_path=ACHIEVED_QUERY_RESULTS_PATH,
        gold_nosql_map=gold_nosql_map,
        hardness_map=hardness_map,
        output_file=comparison_output_file,
    )