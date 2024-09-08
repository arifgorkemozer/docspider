import json
import pandas

from common import *

ground_truth_by_database_names_nl_only = {}

def generate_prompt_for_db_query(db_info, nl_query, gold_sql):
    global ground_truth_by_database_names_nl_only

    result = "Write only the MongoDB with no explanation for the query using the following schema. Do not select extra columns that are not explicitly requested."
    result += NEWLINE_CHAR
    result += "Schema:"
    result += NEWLINE_CHAR
    
    for table_name, table_info in db_info['tables'].items():
        column_str = ", ".join([column_info['name'] for column_info in table_info["columns"]])
        result += f"{table_name}({column_str})"
        result += NEWLINE_CHAR

    if len(db_info['foreign_keys']) > 0:
        result += "Foreign keys:"
        result += NEWLINE_CHAR
        for foreign_key in db_info['foreign_keys']:
            result += f"{foreign_key['ref_table']}.{foreign_key['ref_column']} = {foreign_key['main_table']}.{foreign_key['main_column']}"
            result += NEWLINE_CHAR

    result += f"Question:"
    result += NEWLINE_CHAR
    result += nl_query
    result += NEWLINE_CHAR

    if gold_sql:
        result += "Gold SQL:"
        result += NEWLINE_CHAR
        result += gold_sql
        result += NEWLINE_CHAR

    return result

def generate_prompts_to_csv(table_column_metadata, nl_source_json_path, output_filename, ground_truth_path=None, include_gold=True):
    global ground_truth_by_database_names_nl_only

    ONLY_CONSIDER_GROUND_TRUTH_QUERIES = ground_truth_path is not None
    
    if ONLY_CONSIDER_GROUND_TRUTH_QUERIES:
        gt_file = pandas.read_csv(ground_truth_path, sep=';')

        for i, row in gt_file.iterrows():
            database = row["database"]
            nl_query = row["Query"]

            if database not in ground_truth_by_database_names_nl_only.keys():
                ground_truth_by_database_names_nl_only[database] = []

            ground_truth_by_database_names_nl_only[database].append(nl_query)

    with open(output_filename, "w") as out_file:
        # write headers
        out_file.write(f"query_id{SEMICOLON_CHAR}question{SEMICOLON_CHAR}database{SEMICOLON_CHAR}prompt{SEMICOLON_CHAR}gold_sql")
        out_file.write(NEWLINE_CHAR)

        with open(nl_source_json_path, "r") as training_file:
            training_json = json.load(training_file)

            for query_index, query_sample in enumerate(training_json):
                db_id = remove_double_space(query_sample["db_id"].replace(DOUBLE_QUOTE_CHAR, SINGLE_QUOTE_CHAR))
                nl_query = remove_double_space(query_sample["question"].replace(DOUBLE_QUOTE_CHAR, SINGLE_QUOTE_CHAR))
                answer = remove_double_space(query_sample["query"].replace(DOUBLE_QUOTE_CHAR, SINGLE_QUOTE_CHAR))

                # only consider ground truth queries if ground truth provided
                # otherwise generate prompts for all queries
                nl_queries_in_ground_truth = ground_truth_by_database_names_nl_only[db_id] if ONLY_CONSIDER_GROUND_TRUTH_QUERIES else []

                if not ONLY_CONSIDER_GROUND_TRUTH_QUERIES or nl_query in nl_queries_in_ground_truth:
                    prompt = generate_prompt_for_db_query(
                        db_info=table_column_metadata[db_id],
                        nl_query=nl_query,
                        gold_sql=answer if include_gold else None
                    ).replace(DOUBLE_QUOTE_CHAR, SINGLE_QUOTE_CHAR)

                    out_file.write(f"{query_index+1}{SEMICOLON_CHAR}\"{nl_query}\"{SEMICOLON_CHAR}\"{db_id}\"{SEMICOLON_CHAR}\"{prompt}\"{SEMICOLON_CHAR}\"{answer}\"")
                    out_file.write(NEWLINE_CHAR)
