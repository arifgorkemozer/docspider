import os
import pandas
import re
import subprocess
import sqlite3

from common import *

def run_sql_query(sqlite_file_path, sql_query, out_file):
    sqlite_db = sqlite3.connect(sqlite_file_path)
    sqlite_cursor = sqlite_db.cursor()
    sqlite_query_result = sqlite_cursor.execute(sql_query)
    sqlite_query_result_rows = sqlite_query_result.fetchall()
    out_file.write(str(sqlite_query_result_rows))

def run_mongo_query(mongo_db_name, mongo_query, out_file):
    mongodb_query_result = subprocess.check_output(["mongosh", mongo_db_name, "--eval", mongo_query, "--quiet"])
    out_file.write(re.sub("DeprecationWarning.*\n", "", mongodb_query_result.decode("utf-8")))

def run_predicted_queries(experiment_type, predictions_output_path, expected_query_results_path, achieved_query_results_path, spider_database_path, gold_sql_map):
    predicted_query_file = pandas.read_csv(predictions_output_path, sep=TAB_CHAR)

    for i, row in predicted_query_file.iterrows():
        query_index = int(row["query_id"])
        predicted_query = row["pred_nosql"]

        print(f"Running predicted query #{query_index}")

        sql, database = gold_sql_map[query_index]
        
        with open(f"{expected_query_results_path}/result{query_index}.txt", "w") as result_file_expected:
            try:
                sqlite_parent_folder_path = f"{spider_database_path}/{database}"
                files = os.listdir(sqlite_parent_folder_path)
                sqlite_filename = None

                for filename in files:
                    if filename.endswith(".sqlite"):
                        sqlite_filename = filename
                        break
                
                if sqlite_filename:
                    run_sql_query(
                        sqlite_file_path=f'{sqlite_parent_folder_path}/{sqlite_filename}',
                        sql_query=sql,
                        out_file=result_file_expected
                    )
            except Exception as ex:
                print(f"#SQL EXECUTION ERROR: {query_index}")
                result_file_expected.write("[]")

        if experiment_type == "nl2sql":
            with open(f"{achieved_query_results_path}/result{query_index}.txt", "w") as result_file_achieved:
                try:
                    sqlite_parent_folder_path = f"{spider_database_path}/{database}"
                    files = os.listdir(sqlite_parent_folder_path)
                    sqlite_filename = None

                    for filename in files:
                        if filename.endswith(".sqlite"):
                            sqlite_filename = filename
                            break
                    
                    if sqlite_filename:
                        run_sql_query(
                            sqlite_file_path=f'{sqlite_parent_folder_path}/{sqlite_filename}',
                            sql_query=predicted_query,
                            out_file=result_file_achieved
                        )
                except Exception as ex:
                    print(f"#PREDICTED SQL EXECUTION ERROR: {query_index}")
                    result_file_achieved.write("[]")    
        else: # nl2mongo
            with open(f"{achieved_query_results_path}/result{query_index}.txt", "w") as result_file_achieved:
                try:
                    run_mongo_query(
                        mongo_db_name=f"spider_{database}",
                        mongo_query=predicted_query,
                        out_file=result_file_achieved
                    )
                except Exception as ex:
                    print(f"#PREDICTED MONGODB EXECUTION ERROR: {query_index}")
                    result_file_achieved.write("[]")
