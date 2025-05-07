import os
import json
import sqlite3
from pymongo import MongoClient
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("database_folder_path", help="Path to database folder that contains databases. e.g: ./spider_data/database")
args = parser.parse_args()

DATABASE_FOLDER_PATH = args.database_folder_path

mongo_client = MongoClient(host="localhost", port=27017)
databases = sorted(os.listdir(DATABASE_FOLDER_PATH))
processed = 0

insert_cmds = []

for db_index, db_name in enumerate(databases):
    if db_name in [".DS_Store"]:
        continue
    
    db_path = f"{DATABASE_FOLDER_PATH}/{db_name}"
    mongo_database_name = db_name
    files = sorted(os.listdir(db_path))
    sqlite_filename = None

    for filename in files:
        if filename.endswith(".sqlite"):
            sqlite_filename = filename
            break


    if sqlite_filename:
        print("Processing database:", db_name)
        schema_sqlite_file_path = f"{db_path}/{sqlite_filename}"

        # drop database if exists
        mongo_client.drop_database(mongo_database_name)
        print("\tDropped database:", db_name)
        mongo_database = mongo_client[mongo_database_name]

        # connect to sqlite db and get table names
        sqlite_db = sqlite3.connect(schema_sqlite_file_path)
        sqlite_db.text_factory = lambda b: b.decode(errors = 'ignore')
        sqlite_cursor = sqlite_db.cursor()
        sqlite_query_result = sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [row[0] for row in sqlite_query_result.fetchall()]

        # for all tables in sqlite database
        # get all rows from the database
        # get column names from the database
        for table_name in table_names:
            print("\tAdding rows to the collection:", table_name)
            sqlite_query_result = sqlite_cursor.execute(f"SELECT * FROM {table_name}")
            all_rows = sqlite_query_result.fetchall()
            sqlite_query_result = sqlite_cursor.execute(f"PRAGMA table_info({table_name})")
            table_columns = [column_info[1] for column_info in sqlite_query_result.fetchall()]

            mongo_collection = mongo_database[table_name]

            for row in all_rows:
                entry_for_mongo = {}
                for column_index, column_name in enumerate(table_columns):
                    entry_for_mongo[column_name] = row[column_index]

                mongo_collection.insert_one(entry_for_mongo)

        processed += 1

    else:
        print("Skipped:", db_name)

print("Total databases:", len(databases))
print("Processed databases:" , processed)

# usage: python3 migrate_spider_data_to_mongodb.py ./spider_data/database
