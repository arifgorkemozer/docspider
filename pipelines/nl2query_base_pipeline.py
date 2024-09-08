# python3 nl2query_base_pipeline.py <root_path> <experiment_tag> <experiment_type> <dataset_type>
# python3 nl2query_base_pipeline.py .. <tag> nl2mongo dev
# add --include_goldsql for adding gold sql to the prompts
# add --skip_llm_step for skipping nosql query generation wit llm
# add --skip_execution_step for skipping running sql and nosql queries

import argparse
import json
import os
import pandas
import shutil

from common import *
from step1_metadata import generate_table_column_metadata
from step2_prompts_basic import generate_prompts_to_csv
from step3_llm import generate_db_queries_with_llm
from step4_run_queries import run_predicted_queries
from step5_compare_results import compare_expected_predicted_query_results

parser = argparse.ArgumentParser()
parser.add_argument("root_path", help="Root path for reference")
parser.add_argument("experiment_tag", help="Tag for the experiment")
parser.add_argument("experiment_type", choices=[EXPERIMENT_NL2SQL, EXPERIMENT_NL2MONGO], help="Task type, from NL to SQL/Mongo")
parser.add_argument("dataset_type", choices=["train", "dev"], help="Use spider train or dev set")
parser.add_argument("--include_goldsql", action="store_true", help="Include gold SQL to prompt or not")
parser.add_argument("--skip_llm_step", action="store_true", help="Skip generating queries with LLM or not")
parser.add_argument("--skip_execution_step", action="store_true", help="Skip running generated queries or not")
args = parser.parse_args()

# generic definitions
DATASET_JSON_FILENAME = "train_spider.json" if args.dataset_type == "train" else "dev.json"
GOLD_SQL_FILENAME = "custom_train_gold.sql" if args.dataset_type == "train" else "custom_dev_gold.sql"
INCLUDE_GOLD = args.include_goldsql
SKIP_LLM_GENERATION = args.skip_llm_step
SKIP_QUERY_EXECUTION = args.skip_execution_step

# generic paths
ROOT_PATH = args.root_path[0:-1] if args.root_path.endswith("/") else args.root_path
TABLES_JSON_PATH = f"{ROOT_PATH}/spider/tables.json"
METADATA_OUTPUT_PATH = f"{ROOT_PATH}/spider_table_column_metadata.json"
NL_SOURCE_JSON_PATH = f"{ROOT_PATH}/spider/{DATASET_JSON_FILENAME}"
ALL_EXPERIMENTS_ROOT_PATH = f"{ROOT_PATH}/spider_mongodb_sqlite_comparison"
GOLD_SQL_FILEPATH = f"{ROOT_PATH}/spider/{GOLD_SQL_FILENAME}"
HARDNESS_FILEPATH = f"{ROOT_PATH}/docspider_ground_truth_dataset/hardness_{args.dataset_type}.csv"
GROUND_TRUTH_FILEPATH = f"{ROOT_PATH}/docspider_ground_truth_dataset/docspider_ground_truth_{args.dataset_type}.csv"

# experiment specific paths
EXPERIMENT_TAG = args.experiment_tag
EXPERIMENT_TYPE = args.experiment_type
EXPERIMENT_PATH = f"{ALL_EXPERIMENTS_ROOT_PATH}/{EXPERIMENT_TAG}"
PROMPT_OUTPUT_FILENAME = "prompts.csv"
PREDICTIONS_OUTPUT_FILENAME = "predicted_nosql.tsv"
PROMPT_OUTPUT_PATH = f"{EXPERIMENT_PATH}/{PROMPT_OUTPUT_FILENAME}"
PREDICTIONS_OUTPUT_PATH = f"{EXPERIMENT_PATH}/{PREDICTIONS_OUTPUT_FILENAME}"
EXPECTED_QUERY_RESULTS_FOLDERNAME = "expected_results"
ACHIEVED_QUERY_RESULTS_FOLDERNAME = "achieved_results"
EXPECTED_QUERY_RESULTS_PATH = f"{EXPERIMENT_PATH}/{EXPECTED_QUERY_RESULTS_FOLDERNAME}"
ACHIEVED_QUERY_RESULTS_PATH = f"{EXPERIMENT_PATH}/{ACHIEVED_QUERY_RESULTS_FOLDERNAME}"

# read query hardness
hardness_map = {}
prompt_with_hardness_file = pandas.read_csv(HARDNESS_FILEPATH, sep=SEMICOLON_CHAR)

for i, row in prompt_with_hardness_file.iterrows():
    hardness_map[i+1] = row["hardness"]

######################### BEGIN - CREATE FILES AND FOLDERS NEEDED #########################
items_under_main_folder = os.listdir(ROOT_PATH)
items_under_all_experiments_folder = os.listdir(ALL_EXPERIMENTS_ROOT_PATH)

# create experiment folder if needed
if EXPERIMENT_TAG not in items_under_all_experiments_folder:
    os.mkdir(EXPERIMENT_PATH)

items_under_experiment = os.listdir(EXPERIMENT_PATH)

# create results folders if needed
if EXPECTED_QUERY_RESULTS_FOLDERNAME not in items_under_experiment:
    os.mkdir(EXPECTED_QUERY_RESULTS_PATH)

if ACHIEVED_QUERY_RESULTS_FOLDERNAME not in items_under_experiment:
    os.mkdir(ACHIEVED_QUERY_RESULTS_PATH)

# copy gold sql file if needed
if GOLD_SQL_FILENAME not in items_under_experiment:
    shutil.copy(GOLD_SQL_FILEPATH, EXPERIMENT_PATH)

# read gold sqls
gold_sql_file = pandas.read_csv(f"{EXPERIMENT_PATH}/{GOLD_SQL_FILENAME}", sep=TAB_CHAR)
gold_sql_map = {}

for i, row in gold_sql_file.iterrows():
    query_index = i+1
    query = row["query"]
    db = row["db"]
    gold_sql_map[query_index] = (query, db)

# on standard mode, generate metadata, prompts, queries; run them and compare results
else:
    # STEP 1 - GENERATE TABLE COLUMN METADATA
    if METADATA_OUTPUT_PATH in items_under_main_folder:
        with open(METADATA_OUTPUT_PATH, "r") as inp:
            metadata = json.load(inp)
    else:
        metadata = generate_table_column_metadata(TABLES_JSON_PATH, METADATA_OUTPUT_PATH)


    # STEP 2 - GENERATE LLM PROMPTS
    generate_prompts_to_csv(
        table_column_metadata=metadata,
        nl_source_json_path=NL_SOURCE_JSON_PATH,
        output_filename=PROMPT_OUTPUT_PATH,
        ground_truth_path=GROUND_TRUTH_FILEPATH,
        include_gold=INCLUDE_GOLD
    )
    prompt_file = pandas.read_csv(PROMPT_OUTPUT_PATH, sep=SEMICOLON_CHAR)

    # STEP 3 - GENERATE DB QUERIES WITH LLM
    # gpt_model_name="gpt-4-1106-preview",
    # gpt_model_name="gpt-3.5-turbo-0613",
    # gpt_model_name="ft:gpt-3.5-turbo-0613:personal::8nwiLepO",
    # gpt_model_name="gpt-4o-2024-08-06",
    # gpt_model_name="ft:gpt-4o-2024-08-06:personal::A19WGCNH",
    if not SKIP_LLM_GENERATION:
        generate_db_queries_with_llm(
            gpt_model_name="ft:gpt-4o-2024-08-06:personal::A19WGCNH",
            prompt_output_path=PROMPT_OUTPUT_PATH,
            predictions_output_path=PREDICTIONS_OUTPUT_PATH,
            # start_from_query_id=404,
        )

    # STEP 4 - RUN GOLD SQLS AND LLM-GENERATED QUERIES
    if not SKIP_QUERY_EXECUTION:
        run_predicted_queries(
            experiment_type=EXPERIMENT_TYPE,
            predictions_output_path=PREDICTIONS_OUTPUT_PATH,
            expected_query_results_path=EXPECTED_QUERY_RESULTS_PATH,
            achieved_query_results_path=ACHIEVED_QUERY_RESULTS_PATH,
            spider_database_path=f"{ROOT_PATH}/spider/database",
            gold_sql_map=gold_sql_map
        )

# STEP 5 - COMPARE QUERY RESULTS OF GOLD SQLS AND LLM-GENERATED QUERIES
for sensitivity in SENSITIVITY_OPTIONS:
    compare_expected_predicted_query_results(
        current_sensitivity=sensitivity,
        experiment_tag=EXPERIMENT_TAG,
        experiment_type=EXPERIMENT_TYPE,
        experiment_path=EXPERIMENT_PATH,
        predictions_output_path=PREDICTIONS_OUTPUT_PATH,
        expected_query_results_path=EXPECTED_QUERY_RESULTS_PATH,
        achieved_query_results_path=ACHIEVED_QUERY_RESULTS_PATH,
        gold_sql_map=gold_sql_map,
        hardness_map=hardness_map,
    )