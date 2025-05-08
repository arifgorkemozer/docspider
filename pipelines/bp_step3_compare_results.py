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


def translate_mongo_result(mongodb_result_str):
    # add commas between entries
    result_str = mongodb_result_str.replace('}{', '},{')

    # replace nulls
    result_str = result_str.replace(r'_id:\s*null', '_id: "null"')

    # replace _id: ObjectId()
    result_str = re.sub('\s*_?id:\s*ObjectId\(\'[a-z0-9]+\'\),?', EMPTY_STR, result_str)

    # replace _id: { ... } with objects
    result_str = re.sub(r'_id:\s*{[^}]+}', EMPTY_STR, result_str)

    # replace <keyword>: Long()
    if "Long(" + SINGLE_QUOTE_CHAR in result_str:
        while True:
            try:
                long_index = result_str.index("Long(" + SINGLE_QUOTE_CHAR)
                start_index = long_index + len("Long(" + SINGLE_QUOTE_CHAR)
                end_index = long_index + len("Long(" + SINGLE_QUOTE_CHAR)

                while result_str[end_index] != SINGLE_QUOTE_CHAR:
                    end_index += 1
                
                result_str = result_str[0: long_index] + result_str[start_index:end_index] + result_str[end_index+2 :]
            except ValueError:
                break

    if result_str == "":
        result_str = "[]"
        result_object = []
    
    # wrap non quoted keywords in json
    elif COLON_CHAR in result_str:
        keywords = find_all_keywords_in_mongo_result(result_str)
        keywords = sorted(keywords, key=lambda x: len(x), reverse=True)

        for keyword in keywords:
            result_str = result_str.replace(f"{keyword}:" , '"' + keyword + '":')

        if not result_str.startswith("[") and not result_str.endswith("]"):
            result_str = "[" + result_str + "]"
        
        try:
            mongodb_original_result_object = eval(result_str)
        except:
            mongodb_original_result_object = []

        result_object = []

        for original_result_row in mongodb_original_result_object:
            try:
                result_object.append(tuple(original_result_row.values()))
            except:
                result_object.append((original_result_row,))
        
    # asadad
    else:
        result_str = ",".join("\"" + x + "\"" for x in result_str.split("\n") if x.strip() != "")

        if not result_str.startswith("[") and not result_str.endswith("]"):
            result_str = "[" + result_str + "]"

        mongodb_original_result_object = eval(result_str)
        result_object = [(original_result_row,) for original_result_row in mongodb_original_result_object]

    return result_object


def compare_expected_predicted_query_results(current_sensitivity, experiment_tag, experiment_path, predictions_output_path, achieved_query_results_path, expected_query_results_path, gold_nosql_map, hardness_map, output_file):
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
        # print("Comparing query #" + str(query_id))

        if hardness in accurates_by_hardness.keys():
            total += 1
            accurates_by_hardness[hardness]["total"] += 1

            achieved_query_result_file = open(f"{achieved_query_results_path}/{result_filename}", "r")
            achieved_result_str = "".join([line for line in achieved_query_result_file.readlines() if line.strip() != ""])
            achieved_result_object = translate_mongo_result(achieved_result_str)

            expected_query_result_file = open(f"{expected_query_results_path}/result{query_id}.txt", "r")
            expected_result_str = "".join([line for line in expected_query_result_file.readlines() if line.strip() != ""])
            expected_result_object = translate_mongo_result(expected_result_str)

            try:
                both_results_are_equal = True
                expected_result_rowcount = len(expected_result_object)
                achieved_result_rowcount = len(achieved_result_object)

                if expected_result_rowcount == achieved_result_rowcount:
                    order_required = "order by" in gold_nosql_map[query_id][0].lower()

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

    output_file.write(experiment_tag + " - " + current_sensitivity)
    output_file.write(NEWLINE_CHAR)
    output_file.write("Accuracy: " + str(100.0 * len(accurates) / total))
    output_file.write(NEWLINE_CHAR)
    output_file.write("Easy: " + str(easy_acc[0]) + " Accuracy: " + str(100.0 * easy_acc[1]))
    output_file.write(NEWLINE_CHAR)
    output_file.write("Medium: " + str(medium_acc[0]) + " Accuracy: " + str(100.0 * medium_acc[1]))
    output_file.write(NEWLINE_CHAR)
    output_file.write("Hard: " + str(hard_acc[0]) + " Accuracy: " + str(100.0 * hard_acc[1]))
    output_file.write(NEWLINE_CHAR)
    output_file.write("Extra: " + str(extra_acc[0]) + " Accuracy: " + str(100.0 * extra_acc[1]))
    output_file.write(NEWLINE_CHAR)
    output_file.write("-------")
    output_file.write(NEWLINE_CHAR)

    ground_truth_file = open(f"{experiment_path}/ground-truth-{current_sensitivity}.tsv", "w")
    predictions_file = pandas.read_csv(predictions_output_path, sep=TAB_CHAR)
    predictions = {}

    for i, row in predictions_file.iterrows():
        query_id = row["query_id"]
        pred_nosql = row["pred_nosql"]
        predictions[query_id] = pred_nosql

    accurates = sorted(accurates)

    ground_truth_file.write(f"query_id{TAB_CHAR}hardness{TAB_CHAR}nosql_query{NEWLINE_CHAR}")

    for query_id in accurates:
        ground_truth_file.write(f"{query_id}{TAB_CHAR}{hardness_map[query_id]}{TAB_CHAR}{predictions[query_id]}{NEWLINE_CHAR}")

