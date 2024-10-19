import os
import pandas
import re

from common import *

def find_all_keywords_in_mongo_result(mongodb_result_str):
    JSON_PUNC_START_CHARS = ['[','{']
    JSON_PUNC_END_CHARS = [']','}']
    JSON_PUNC_CHARS = [' ', ','] + JSON_PUNC_START_CHARS + JSON_PUNC_END_CHARS

    keywords = set()
    i = 0
    j = 0

    while i < len(mongodb_result_str) and j < len(mongodb_result_str):
        # find occurence of :
        while j < len(mongodb_result_str) and mongodb_result_str[j] != ':':
            j += 1
        
        # wrap the keyword with " if it is not wrapped - for proper json format
        if j < len(mongodb_result_str) - 1 and mongodb_result_str[j-1] != '"':
            i = j

            abort = False

            # find the beginning of keyword
            while i > 0 and (mongodb_result_str[i] not in JSON_PUNC_CHARS):
                if mongodb_result_str[i] == '\'':
                    abort = True
                    break

                i -= 1
            
            if not abort:
                i += 1
                keyword = mongodb_result_str[i:j]
                keywords.add(keyword)
            
        i = j
        j = j + 1
    
    return list(keywords)


def compare_expected_predicted_query_results(current_sensitivity, experiment_tag, experiment_type, experiment_path, predictions_output_path, achieved_query_results_path, expected_query_results_path, gold_sql_map, hardness_map):
    result_filenames = sorted(os.listdir(achieved_query_results_path))

    total = 0
    accurates = []
    accurates_by_hardness = {
        "easy": {
            "correct": 0,
            "total": 0,
        },
        "medium": {
            "correct": 0,
            "total": 0,
        },
        "hard": {
            "correct": 0,
            "total": 0,
        },
        "extra": {
            "correct": 0,
            "total": 0,
        },
    }

    def get_accuracy_str_and_percentage(hardness):
        if accurates_by_hardness[hardness]["total"] == 0:
            return "0/0", 0

        else:
            return f'{accurates_by_hardness[hardness]["correct"]}/{accurates_by_hardness[hardness]["total"]}', accurates_by_hardness[hardness]["correct"] / accurates_by_hardness[hardness]["total"]

    for result_filename in result_filenames:
        query_id = int(result_filename.replace("result", EMPTY_STR).replace(".txt", EMPTY_STR))
        hardness = hardness_map[query_id]

        if hardness in accurates_by_hardness.keys():
            achieved_query_result_file = open(f"{achieved_query_results_path}/{result_filename}", "r")
            expected_query_result_file = open(f"{expected_query_results_path}/result{query_id}.txt", "r")

            # process sqlite results
            expected_result_object = eval("".join([line.strip() for line in expected_query_result_file.readlines() if line.strip() != ""]))

            if len(expected_result_object) > 0:
                total += 1
                accurates_by_hardness[hardness]["total"] += 1

                if experiment_type == EXPERIMENT_NL2SQL:
                    achieved_result_object = eval("".join([line.strip() for line in achieved_query_result_file.readlines() if line.strip() != ""]))

                else: # nl2mongo
                    # process mongodb results
                    mongodb_result_str = "".join([line.strip() for line in achieved_query_result_file.readlines() if line.strip() != ""])

                    # remove 'type it for more'
                    mongodb_result_str = mongodb_result_str.replace('Type "it" for more', '')

                    # replace nulls
                    mongodb_result_str = mongodb_result_str.replace('null', '"NULL"')

                    # replace _id: ObjectId 's
                    mongodb_result_str = re.sub('\s*_?id:\s*ObjectId\(\'[a-z0-9]+\'\),?', EMPTY_STR, mongodb_result_str)

                    # replace _id: null 's
                    mongodb_result_str = re.sub('\s*_?id:\s*null,?', EMPTY_STR, mongodb_result_str)

                    if "Long(" + SINGLE_QUOTE_CHAR in mongodb_result_str:
                        while True:
                            try:
                                long_index = mongodb_result_str.index("Long(" + SINGLE_QUOTE_CHAR)
                                start_index = long_index + len("Long(" + SINGLE_QUOTE_CHAR)
                                end_index = long_index + len("Long(" + SINGLE_QUOTE_CHAR)

                                while mongodb_result_str[end_index] != SINGLE_QUOTE_CHAR:
                                    end_index += 1
                                
                                mongodb_result_str = mongodb_result_str[0: long_index] + mongodb_result_str[start_index:end_index] + mongodb_result_str[end_index+2 :]
                            except ValueError:
                                break
                    
                    if mongodb_result_str == "":
                        mongodb_result_str = "[]"
                        achieved_result_object = []

                    # wrap non quoted keywords in json
                    elif COLON_CHAR in mongodb_result_str:
                        keywords = find_all_keywords_in_mongo_result(mongodb_result_str)
                        keywords = sorted(keywords, key=lambda x: len(x), reverse=True)

                        for keyword in keywords:
                            mongodb_result_str = mongodb_result_str.replace(f"{keyword}:" , '"' + keyword + '":')

                        if not mongodb_result_str.startswith("[") and not mongodb_result_str.endswith("]"):
                            mongodb_result_str = "[" + mongodb_result_str + "]"
                        
                        try:
                            mongodb_original_result_object = eval(mongodb_result_str)
                        except:
                            mongodb_original_result_object = []

                        achieved_result_object = []

                        for original_result_row in mongodb_original_result_object:
                            try:
                                achieved_result_object.append(tuple(original_result_row.values()))
                            except:
                                achieved_result_object.append((original_result_row,))

                    # this is probably a list of results, use unjoined version
                    else:
                        mongodb_original_result_object = [line.strip() for line in achieved_query_result_file.readlines() if line.strip() != ""]
                        achieved_result_object = [(original_result_row,) for original_result_row in mongodb_original_result_object]

                try:
                    both_results_are_equal = True
                    expected_result_rowcount = len(expected_result_object)
                    achieved_result_rowcount = len(achieved_result_object)

                    if expected_result_rowcount == achieved_result_rowcount: # esitlik icin ayni sayida row dönmeli
                        order_required = "order by" in gold_sql_map[query_id][0].lower()

                        if current_sensitivity in [SENSITIVITY_SAME, SENSITIVITY_UNORDERED]:                            
                            for row_id, row_expected in enumerate(expected_result_object):
                                row_expected_set = set(row_expected)

                                if current_sensitivity == SENSITIVITY_UNORDERED:
                                    order_required = False

                                if order_required:
                                    row_achieved_set = set(achieved_result_object[row_id])

                                    if row_expected_set != row_achieved_set:
                                        both_results_are_equal = False
                                        break
                                else:
                                    found = False

                                    for row_achieved in achieved_result_object:
                                        row_achieved_set = set(row_achieved)

                                        if row_achieved_set == row_expected_set: # farkli row order, ayni columnlar
                                            found = True
                                            break

                                    if not found:
                                        both_results_are_equal = False
                                        break

                        if current_sensitivity in [SENSITIVITY_EXTRA_FIELDS, SENSITIVITY_UNORDERED_EXTRA_FIELDS]:
                            if current_sensitivity == SENSITIVITY_UNORDERED_EXTRA_FIELDS:
                                order_required = False

                            if order_required:
                                for row_id, row_expected in enumerate(expected_result_object):
                                    row_expected_set = set(row_expected)
                                    row_achieved_set = set(achieved_result_object[row_id])

                                    if not row_achieved_set.issuperset(row_expected_set):
                                        both_results_are_equal = False
                                        break
                            else:
                                for row_expected in expected_result_object:
                                    found = False
                                    row_expected_set = set(row_expected)

                                    for row_achieved in achieved_result_object:
                                        row_achieved_set = set(row_achieved)

                                        if row_achieved_set.issuperset(row_expected_set):
                                            found = True
                                            break
                                    
                                    if not found:
                                        both_results_are_equal = False
                                        break                        
                        
                        if both_results_are_equal:
                            accurates.append(query_id)
                            accurates_by_hardness[hardness]["correct"] += 1
                except Exception as ex:
                    pass

    # print accuracies on terminal
    easy_acc = get_accuracy_str_and_percentage("easy")
    medium_acc = get_accuracy_str_and_percentage("medium")
    hard_acc = get_accuracy_str_and_percentage("hard")
    extra_acc = get_accuracy_str_and_percentage("extra")

    print(experiment_tag, "-", current_sensitivity)
    print("Accuracy:", 100.0 * len(accurates) / total)
    print("Easy:", easy_acc[0], "Accuracy:", 100.0 * easy_acc[1])
    print("Medium:", medium_acc[0], "Accuracy:", 100.0 * medium_acc[1])
    print("Hard:", hard_acc[0], "Accuracy:", 100.0 * hard_acc[1])
    print("Extra:", extra_acc[0], "Accuracy:", 100.0 * extra_acc[1])
    print("-------")

    ground_truth_file = open(f"{experiment_path}/ground-truth-{current_sensitivity}.tsv", "w")
    predictions_file = pandas.read_csv(predictions_output_path, sep=TAB_CHAR)
    predictions = {}

    for i, row in predictions_file.iterrows():
        query_id = row["query_id"]
        pred_nosql = row["pred_nosql"]
        predictions[query_id] = pred_nosql

    accurates = sorted(accurates)

    ground_truth_file.write(f"query id{TAB_CHAR}hardness{TAB_CHAR}nosql_query{NEWLINE_CHAR}")

    for query_id in accurates:
        ground_truth_file.write(f"{query_id}{TAB_CHAR}{hardness_map[query_id]}{TAB_CHAR}{predictions[query_id]}{NEWLINE_CHAR}")