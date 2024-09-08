import json

def generate_table_column_metadata(tables_json_path, metadata_output_path):
    tables_json_file = open(tables_json_path, "r")
    tables_json = json.load(tables_json_file)
    metadata = {}

    for entry in tables_json:
        db_id = entry["db_id"]
        table_names = entry["table_names_original"]
        all_column_names_orig = entry["column_names_original"]
        all_column_names = entry["column_names"]
        all_column_types = entry["column_types"]
        primary_key_ids = entry["primary_keys"]
        foreign_key_pairs = entry["foreign_keys"]
        column_index_map = {}

        if db_id not in metadata.keys():
            metadata[db_id] = {
                "tables": {}
            }

        # table column name mapping
        for table_index, table_name in enumerate(table_names):
            for column_index, column_name_obj in enumerate(all_column_names_orig):
                column_table_index = column_name_obj[0]
                column_name = column_name_obj[1]

                if column_table_index != table_index:
                    continue
                
                column_index_map[column_index] = (table_name, column_name)

        # table column names and types
        for table_index, table_name in enumerate(table_names):
            metadata[db_id]["tables"][table_name] = {
                "columns": [],
                "primary_keys": [],
            }

            for column_index, column_name_obj in enumerate(all_column_names_orig):
                column_table_index = column_name_obj[0]
                column_name = column_name_obj[1]
                column_type = all_column_types[column_index]

                if column_table_index != table_index:
                    continue

                metadata[db_id]["tables"][table_name]["columns"].append({
                    "name": column_name,
                    "type": column_type
                })

            # table primary keys
            metadata[db_id]["tables"][table_name]["primary_keys"] = []

            if table_index < len(primary_key_ids):
                primary_key_index = primary_key_ids[table_index]
                metadata[db_id]["tables"][table_name]["primary_keys"].append(column_index_map[primary_key_index][1])

            # table foreign keys
            if "foreign_keys" not in metadata[db_id].keys():
                metadata[db_id]["foreign_keys"] = []

                for foreign_key_pair in foreign_key_pairs:
                    referencing_table = column_index_map[foreign_key_pair[0]][0]
                    referencing_column = column_index_map[foreign_key_pair[0]][1]

                    main_table = column_index_map[foreign_key_pair[1]][0]
                    main_column = column_index_map[foreign_key_pair[1]][1]                    

                    metadata[db_id]["foreign_keys"].append({
                        "main_table": main_table,
                        "main_column": main_column,
                        "ref_table": referencing_table,
                        "ref_column": referencing_column
                    })

    with open(metadata_output_path, "w") as out:
        out.write(json.dumps(metadata))
    
    return metadata
