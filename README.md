# P_Tuning
This repository is the implementation of prompt based parameter efficient finetuning technique called P TUNING

# What is P-Tuning? 

P-tuning is a method for understanding language tasks using models. It's a type of soft prompt technique with added features. Unlike prefix tuning, P-tuning allows prompt tokens anywhere in the input, not just the beginning, and adds them only once instead of throughout layers. Adding anchor tokens can enhance performance by highlighting input features. Overall, P-tuning boosts efficiency compared to manual prompts, enabling GPT-style models to rival BERT-style ones in language understanding tasks.

## Model

RoBERTa-large : It is a large-scale pre-trained language model developed by Facebook AI. It's based on the BERT architecture but trained on a larger corpus for longer, with more varied data sources and without certain pre-training tasks. This results in improved performance on various natural language processing tasks compared to its predecessor.

## dataset

MRPC subset of GLUE : The GLUE (General Language Understanding Evaluation) benchmark is a collection of diverse natural language understanding tasks aimed at evaluating the performance of models across various NLP tasks. MRPC (Microsoft Research Paraphrase Corpus) is one of the datasets included in GLUE. It consists of sentence pairs extracted from news articles, where each pair is labeled as either a paraphrase or non-paraphrase. This dataset is commonly used for tasks such as paraphrase identification and evaluation of model performance on sentence similarity.

## libraries used

- peft: for model pruning and quantization
- transformers: transformers: For utilizing and fine-tuning the model.
- datasets: For handling and processing the data.
- numpy: For numerical computations.
- torch: For building and training neural networks.

## Hyper parameters

- learning_rate=1e-3
- batch_size=32
- num_train_epochs=10
- weight_decay=0.01

## Training results

| Epoch | Training Loss | Validation Loss | Accuracy | F1       |
|-------|---------------|-----------------|----------|----------|
| 1     | No log        | 0.609398        | 0.679420 | 0.801579 |
| 2     | No log        | 0.666898        | 0.673043 | 0.801966 |
| 3     | No log        | 0.614854        | 0.681739 | 0.804278 |
| 4     | No log        | 0.690202        | 0.665507 | 0.799025 |
| 5     | No log        | 0.590705        | 0.696812 | 0.806368 |
| 6     | No log        | 0.615271        | 0.680580 | 0.802580 |
| 7     | No log        | 0.594967        | 0.693913 | 0.807860 |
| 8     | No log        | 0.605051        | 0.689855 | 0.807346 |
| 9     | 0.620900      | 0.591118        | 0.697391 | 0.808931 |
| 10    | 0.620900      | 0.591311        | 0.697391 | 0.808229 |


# Usage

```
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

peft_model_id = "likhith231/roberta-large-peft-p-tuning"
config = PeftConfig.from_pretrained(peft_model_id)
inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(inference_model, peft_model_id)

classes = ["not equivalent", "equivalent"]

sentence1 = "Coast redwood trees are the tallest trees on the planet and can grow over 300 feet tall."
sentence2 = "The coast redwood trees, which can attain a height of over 300 feet, are the tallest trees on earth."

inputs = tokenizer(sentence1, sentence2, truncation=True, padding="longest", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits
    print(outputs)

paraphrased_text = torch.softmax(outputs, dim=1).tolist()[0]
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrased_text[i] * 100))}%")

```