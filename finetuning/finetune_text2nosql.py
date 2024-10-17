from pprint import pprint
from tqdm import tqdm
import pandas as pd
import torch
from datasets import Dataset, load_dataset, DatasetDict
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import wandb
import argparse

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
#MODEL_NAME = "deepseek-ai/deepseek-coder-33b-instruct"

parser = argparse.ArgumentParser() 
parser.add_argument('--model_id', default='mistralai/Mistral-7B-Instruct-v0.2', type=str, help='dataset name')
parser.add_argument('--batch', default=8, type=int, help='batch size')
parser.add_argument('--llama_prompt', default=True, type=bool, help='prompt format')
parser.add_argument('--skip_train', default=False, type=bool, help='skip to inference step')
parser.add_argument('--lr', default=2e-5, type=float, help='learning rate')
parser.add_argument('--epoch', default=3, type=int, help='epoch')
parser.add_argument('--cache_dir', default=None, type=str, help='hf model cache dir')
args = parser.parse_args()

isLlama = args.llama_prompt
MODEL_NAME = args.model_id

N_EPOCHS = args.epoch
DO_TRAIN = not args.skip_train
LR = args.lr
BATCH_SIZE = args.batch
EVAL_STEPS = 0.1
SAVE_DIR = MODEL_NAME.split('/')[-1]  + "_v2"
OUTPUT_DIR = "output_nosql/" + SAVE_DIR
WANDB_PROJECT = SAVE_DIR + " text2nosql batch: " + str(BATCH_SIZE)

params = {}
params["MODEL_NAME"] = MODEL_NAME
params["LR"] = LR
params["EVAL_STEPS"] = EVAL_STEPS
params["BATCH_SIZE"] = BATCH_SIZE
params["N_EPOCHS"] = N_EPOCHS
params["DO_TRAIN"] = DO_TRAIN
params["OUTPUT_DIR"] = OUTPUT_DIR
params["WANDB_PROJECT"] = WANDB_PROJECT

print(params)

def generate_prompt(prompt, system_prompt="", label=""):
    if isLlama:
        return f"<s>[INST] {system_prompt} {prompt} [/INST] {label}".strip()
    else:
        end_idx = prompt.find("Schema:")
        instr = prompt[:end_idx]
        prompt = prompt[end_idx:]
        return f"""{instr}
### Instruction:
{prompt}
### Response:
{label}
"""

def generate_text(data_point):
    text = generate_prompt(data_point["Prompt"], system_prompt="", label=data_point['target'])

    return {
        "query" : data_point['target'],
        "schema" : data_point['database'],
        "text" : text
    }

def process_dataset(data: Dataset):
    return (
        data.shuffle(seed=42)
        .map(generate_text)
        .remove_columns(
            [
                "query_id",
                "hardness",
                "database",
                "gold_sql",
                "gpt4 answer",
                "deepseek answer",
                "Query",
                "Prompt",
                "target",
            ]
        )
    )

def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
	    cache_dir=args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, 
                                              cache_dir=args.cache_dir )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


PATH = "data/"
train_df = pd.read_csv(PATH + "spider_nosql_train.csv", sep=";", encoding="utf-8")
test_df = pd.read_csv(PATH + "spider_nosql_dev.csv", sep=";", encoding="utf-8")
test_df = test_df.drop('gpt3.5 answer', axis=1)

dataset_train = Dataset.from_pandas(train_df)
dataset_train = dataset_train.train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    "train": dataset_train["train"],
    "validation": dataset_train["test"],
    "test": Dataset.from_pandas(test_df)
})

dataset["train"] = process_dataset(dataset["train"])
dataset["eval"] = process_dataset(dataset["validation"])
#dataset["test"] = process_dataset(dataset["test"])


model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False

if DO_TRAIN:
    lora_r = 32
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = [
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj",
    ]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    # add LoRA adaptor
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


    training_arguments = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        logging_steps=1,
        learning_rate=LR,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=N_EPOCHS,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        warmup_ratio=0.05,
        save_strategy="steps",
        save_steps=EVAL_STEPS*2,
        group_by_length=True,
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name=WANDB_PROJECT,
        #save_safetensors=True,
        lr_scheduler_type="linear",
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    print(training_arguments.to_json_string())
    trainer = SFTTrainer( #
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        #peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    trainer.train()

    trainer.save_model()

    model = trainer.model

# TEST SET
examples = []

for idx, row in test_df.iterrows():
    text = generate_prompt(row["Prompt"], system_prompt="", label="")

    examples.append( {
            "query" : row['target'],
            "schema" : row["database"],
            "prompt" : text
        }
    )
        
test_df = pd.DataFrame(examples)

if not DO_TRAIN:
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    model = model.merge_and_unload()
    model.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

def generateResponse(model, text: str):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    inputs_length = len(inputs["input_ids"][0])
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, top_p=0.9, top_k=75, temperature=0.01)
#num_beams=3, length_penalty=-0.3, early_stopping=True)
    return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)


references = []
predictions = []
"""
for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    response = generateResponse(model, row.prompt)

    begin_idx = response.find("db.")
    if begin_idx >= 0:
        response = response[begin_idx:]
        end_idx = response.find("```")
        response = response[:end_idx].strip()

    predictions.append(response.strip())
    references.append(row.query)
"""
target_queries = [
    "Find the listing ids and names in Istanbul with hosts that their government id is verified",
    "Find the listing ids and names in Istanbul with hosts that their facebook account is verified",
    "Show the listing ids and names in Brazil with the exact location unknown",
    "Show the listing ids and names with free parking and hair dryer in Porto, with at least 100 reviews",
    "Show the listing ids and names in Montreal with a flexible cancellation policy",
    "Find listing ids and names with at least 3 bedrooms in Barcelona ",
    "Find pet-friendly listing ids and names in Sydney with a rating of 95 or above",
    "Show the listing ids and names in New York with a minimum stay of 3 nights or less",
    "Find listing ids and names in Rio De Janeiro with at least 2 bathrooms",
    "Show the listing ids and names with a sea view in Istanbul",
    "Find listing ids and names in Montreal with review score for cleanliness over 8",
    "Show the listing ids and names in New York with hosts that offer breakf1ast",
    "What are the listing ids and names with EV charger and free parking in Hong Kong?",
    "List the listing ids and names that offer indoor fireplace and beach view in Rio De Janeiro",
    "Find listing ids and names in Hong Kong with review score for communication over 9 and offers Wifi and Cable TV",
    "List listing ids with names which have private rooms in New York with hosts that verified email",
    "List listing ids with names which have shared rooms in New York with a flexible cancellation policy",
    "Find listing ids with names which have private rooms that requires minimum stay of 3 nights in New York",
    "Find listing ids with names in Hong Kong with no reviews from the customers",
    "Show the listing ids with names which have shared rooms with couch type beds in Barcelona",
    "Show listing ids with names which have private rooms that are cheaper than 1000 dollars in New York",
    "Find the listing ids and names in Istanbul with hosts that provide Wifi",
]

prompt_template = f"""Write only the MongoDB with no explanation for the query using the following schema. Do not select extra columns that are not explicitly requested. There are column names that have dot character in them, they represent nested document structure. For nested columns, wrap column name with double quote characters while using them in the MongoDB query.
Schema:
listings_and_reviews(listing_url, name, summary, space, description, neighborhood_overview, notes, transit, access, interaction, house_rules, property_type, room_type, bed_type, minimum_nights, maximum_nights, cancellation_policy, last_scraped, calendar_last_scraped, first_review, last_review, accommodates, bedrooms, beds, number_of_reviews, bathrooms, amenities, price, extra_people, guests_included, images.thumbnail_url, images.medium_url, images.picture_url, images.xl_picture_url, host.host_id, host.host_url, host.host_name, host.host_location, host.host_about, host.host_thumbnail_url, host.host_picture_url, host.host_neighbourhood, host.host_is_superhost, host.host_has_profile_pic, host.host_identity_verified, host.host_listings_count, host.host_total_listings_count, host.host_verifications, address.street, address.suburb, address.government_area, address.market, address.country, address.country_code, address.location.type, address.location.coordinates, address.location.is_location_exact, availability.availability_30, availability.availability_60, availability.availability_90, availability.availability_365, review_scores.review_scores_accuracy, review_scores.review_scores_cleanliness, review_scores.review_scores_checkin, review_scores.review_scores_communication, review_scores.review_scores_location, review_scores.review_scores_value, review_scores.review_scores_rating, reviews.date, reviews.listing_id, reviews.reviewer_id, reviews.reviewer_name, reviews.comments)
Question:
"""

target_queries = target_queries[-2:]
for i in tqdm(range(len(target_queries))):
    query = target_queries[i]
    prompt = prompt_template + query
    response = generateResponse(model, prompt)

    begin_idx = response.find("db.")
    if begin_idx >= 0:
        response = response[begin_idx:]

        response = response.replace("```", "")
        end_idx = response.find("Explanation")
        if end_idx >= 0:
            response = response[:end_idx].strip()
    
    predictions.append(response.strip())
    references.append(query)

        
result_df = pd.DataFrame( {'pred_query': predictions, 'gold_query': references})
#result_df = pd.DataFrame( {'pred_query': predictions, 'gold_query': references, "db_id":test_df["schema"].to_list()})
result_df.to_csv(SAVE_DIR + "-" + "nested-ft2-text2nosql.csv", sep="\t", index=True, escapechar='\\')
