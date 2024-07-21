#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch

np.random.seed(42)


# In[40]:


from datasets import load_dataset, Dataset
import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding


# GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
# 
# https://huggingface.co/datasets/nyu-mll/glue
# 
# The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.
# 
# sentence1: a string feature.  
# sentence2: a string feature.  
# label: a classification label, with possible values including not_equivalent (0), equivalent (1).  
# idx: a int32 feature.  

# In[3]:


dataset = load_dataset("glue", "mrpc", split="train")
vdataset = load_dataset("glue", "mrpc", split="validation")


# In[4]:


for feature in dataset.features.keys():
    print(feature)
    print(dataset[feature][:2])
    print()


# In[5]:


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


# In[6]:


model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id)


# In[7]:


for name, value in model.named_parameters():
    print(name)


# In[8]:


model.can_generate()


# In[9]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# In[10]:


list(tokenizer.get_vocab().keys())[-10:]


# In[11]:


tokenizer.decode(token_ids=2000)


# In[12]:


tokenizer.encode(text='to')


# In[13]:


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[14]:


def encode(examples):
    return tokenizer(
        examples["sentence1"], 
        examples["sentence2"], 
        truncation=True, 
        padding="max_length"
        )


# In[15]:


dataset = dataset.map(encode, batched=True)
vdataset = vdataset.map(encode, batched=True)


# We must change the label from 'labels' to 'label'

# In[16]:


dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
vdataset = vdataset.map(lambda examples: {"labels": examples["label"]}, batched=True)


# In[17]:


dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
vdataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])


# input_ids refers to the tokens present in each document.

# In[18]:


print(dataset["input_ids"][:2])


# Token_type_ids contains an integer for every token. These indicate which text segment they correspond to.

# In[19]:


print(dataset["token_type_ids"][:2])


# The attention_mask is a binary tensor.  
# Its purpose is to distinguish between real tokens and padding.  
# It is created by the tokenizer to capture long-range dependencies in text relationships

# In[20]:


print(dataset["attention_mask"][:2])


# In[21]:


len(dataset["attention_mask"][2])


# In[22]:


#Aca iria dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)


# Evaluation of predictions:

# In[23]:


accuracy = evaluate.load("accuracy")


# In[24]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Now we define train hyperparams. 

# In[25]:


training_args = TrainingArguments(
    output_dir="GLUEBERT",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)


# In[26]:


dataset.split


# Now we  Pass training arguments to Trainer (model, dataset, tokenizer, data collator metrics function).
# 

# In[27]:


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=vdataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# In[28]:


trainer.train()


# We now create a homebrewed test set

# In[73]:


test_ex1="These are the same sentence"
test_ex2="These aren't the same sentence"

test_ex3="i am alfredo-sampron"
test_ex4="i don't know who alfredo-sampron is"


# In[82]:


homebrewed_test_set = Dataset.from_dict({"sentence1": [test_ex1, test_ex3],
                                         "sentence2": [test_ex2, test_ex4], 
                                         "idx": [100000, 100001]})


# We must also encode the test set

# In[83]:


homebrewed_test_set = homebrewed_test_set.map(encode, batched=True)


# trainer.predict returns a namedtuple with
# 1) predictions (np.ndarray)
# 2) label_ids 
# 3) metrics

# In[84]:


predictions = trainer.predict(homebrewed_test_set)


# In[86]:


predictions[0]

