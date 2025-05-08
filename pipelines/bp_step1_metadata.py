import json

def generate_collection_metadata(collections_json_path, metadata_output_path):
    collections_json_file = open(collections_json_path, "r")
    collections_json = json.load(collections_json_file)
    metadata = {}

    for entry in collections_json:
        db_id = entry["db_id"]
        collection_names = entry["collection_names"]
        column_names = entry["column_names"]
        column_index_map = {}

        if db_id not in metadata.keys():
            metadata[db_id] = {
                "collections": {}
            }

        # collection-column name mapping
        for collection_index, collection_name in enumerate(collection_names):
            for column_index, column_name_obj in enumerate(column_names):
                column_table_index = column_name_obj[0]
                column_name = column_name_obj[1]

                # only consider columns of the current collection
                if column_table_index != collection_index:
                    continue
                
                column_index_map[column_index] = (collection_name, column_name)

        # collection column names and types
        for collection_index, collection_name in enumerate(collection_names):
            metadata[db_id]["collections"][collection_name] = {
                "columns": [],
            }

            for column_index, column_name_obj in enumerate(column_names):
                column_table_index = column_name_obj[0]
                column_name = column_name_obj[1]

                # only consider columns of the current collection
                if column_table_index != collection_index:
                    continue

                metadata[db_id]["collections"][collection_name]["columns"].append(column_name)

    with open(metadata_output_path, "w") as out:
        out.write(json.dumps(metadata))
    
    return metadata
