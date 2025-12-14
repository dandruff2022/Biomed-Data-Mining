import json
import numpy as np
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import accuracy_score, f1_score



DATA_PATH = "train.json"

MODEL_NAME = "bert-base-uncased"   
SEED = 42



def normalize_text(s: str) -> str:
    s = s.replace("\n", " ")
    s = " ".join(s.split())
    return s


def load_data(path: str):
    """
    {
        "id": int,
        "sentence": str,
        "answer": str,   
        "label": int    
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    labels= [int(x["label"]) for x in data]
    min_label = min(labels)
    max_label= max(labels)

    num_labels=max_label - min_label + 1

    id2label:Dict[int, str] = {}
    for x in data:
        raw_label = int(x["label"])
        idx = raw_label - min_label   
        id2label[idx] = str(x["answer"])

    for i in range(num_labels):
        id2label.setdefault(i, str(i))

    return data,min_label, num_labels, id2label


class DrugDataset(Dataset):

    def __init__(self,
                 items: List[Dict[str, Any]],
                 tokenizer: BertTokenizerFast,
                 label_offset: int = 0):
        self.items = items
        self.tokenizer = tokenizer
        self.label_offset = label_offset  

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]

        text = normalize_text(ex["sentence"])
        raw_label = int(ex["label"])
        label = raw_label - self.label_offset

        encoding =self.tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"]= torch.tensor(label, dtype=torch.long)
        return item


def split_train_val(data, val_ratio=0.1):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(data))
    rng.shuffle(idx)

    n_val=int(len(data) * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_data =[data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    return train_data, val_data


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro =f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}



def main():
    set_seed(SEED)

    data, min_label, num_labels, id2label = load_data(DATA_PATH)
    train_items, val_items = split_train_val(data, val_ratio=0.1)

    tokenizer= BertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds= DrugDataset(train_items, tokenizer, label_offset=min_label)
    val_ds =DrugDataset(val_items, tokenizer, label_offset=min_label)

    label2id = {v: k for k, v in id2label.items()}

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    collator= DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="./bert_output",
        overwrite_output_dir=True,      
        do_train=True,
        do_eval=True,           

        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,

        learning_rate=2e-5,
        weight_decay=0.01,

        logging_steps=50,               
        save_steps=500,                
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())

    trainer.save_model("./bert_drug_advice_best")
    tokenizer.save_pretrained("./bert_drug_advice_best")


if __name__ == "__main__":
    main()
