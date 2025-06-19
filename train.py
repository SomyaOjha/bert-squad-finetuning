import json
import pandas as pd
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

def load_squad_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    contexts, questions, answers = [], [], []

    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                contexts.append(context)
                questions.append(qa['question'])
                if qa['answers']:
                    answers.append(qa['answers'])
                else:
                    answers.append([{'text': '', 'answer_start': 0}])
    return contexts, questions, answers

def preprocess_function(examples):
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding="max_length",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True
    )

    offset_mapping = tokenized_examples.pop("offset_mapping")
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index][0]

        if not answer['text']:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answer['answer_start']
            end_char = start_char + len(answer['text'])
            token_start = 0
            token_end = 0
            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:
                    token_start = idx
                if start < end_char <= end:
                    token_end = idx
                    break
            start_positions.append(token_start)
            end_positions.append(token_end)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

def predict_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    return tokenizer.decode(inputs["input_ids"][0][start_idx:end_idx+1])

if __name__ == "__main__":
    file_path = 'train-v2.0.json'
    contexts, questions, answers = load_squad_data(file_path)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    df = pd.DataFrame({
        "context": contexts,
        "question": questions,
        "answers": answers
    })

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = Dataset.from_dict(train_df.to_dict(orient="list"))
    val_dataset = Dataset.from_dict(val_df.to_dict(orient="list"))

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["context", "question", "answers"])
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["context", "question", "answers"])

    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_strategy="epoch",
        save_total_limit=1,
        warmup_steps=500,
        learning_rate=3e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    model.save_pretrained('./saved_model')
    tokenizer.save_pretrained('./saved_model')

    for i in range(5):
        print(f"\nExample {i+1}")
        print(f"Q: {questions[i]}")
        print(f"Context: {contexts[i][:300]}...")
        print(f"Predicted Answer: {predict_answer(questions[i], contexts[i])}")
