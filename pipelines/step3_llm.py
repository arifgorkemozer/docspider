import openai
import pandas

from common import *

openai.api_key = "<api-key>"

def generate_db_queries_with_llm(gpt_model_name, prompt_output_path, predictions_output_path, start_from_query_id=0):
    prompt_file = pandas.read_csv(prompt_output_path, sep=SEMICOLON_CHAR)

    pred_query_file = open(predictions_output_path, "a")
    tested_prompt_lines = []

    for i, row in prompt_file.iterrows():
        query_id = row["query_id"]
        prompt = row["prompt"]
        db_name = row["database"]
        gold_sql = row["gold_sql"]

        tested_prompt_lines.append((query_id, prompt))

    pred_query_file.write(f"query_id{TAB_CHAR}pred_nosql{NEWLINE_CHAR}")

    for query_id_prompt in tested_prompt_lines:
        query_id, prompt = query_id_prompt

        if query_id > start_from_query_id:
            print(f"Processing prompt #{query_id}")
            
            response = openai.ChatCompletion.create(
                model=gpt_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            )

            pred_query = response["choices"][0]["message"]["content"].replace(NEWLINE_CHAR, SPACE_CHAR).replace(TAB_CHAR, SPACE_CHAR)
            pred_query_file.write(f"{query_id}{TAB_CHAR}{remove_double_space(pred_query)}{NEWLINE_CHAR}")
