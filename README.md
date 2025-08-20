# LLM Engineering: Foundations and Applications

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch_|_Transformers_|_NLTK-orange?style=for-the-badge)

A curated collection of hands-on projects exploring the fundamental building blocks of Natural Language Processing (NLP) and Large Language Models (LLMs).

---

## Table of Contents
1.  [**N-Gram Language Models**](#1-n-gram-language-models): Building statistical language models from scratch.
2.  [**Exploring Tokenization**](#2-exploring-tokenization): A deep dive into tokenization from classic methods to modern subword algorithms.
3.  [**NLP Data Pipelines with PyTorch**](#3-nlp-data-pipelines-with-pytorch): Engineering robust data loaders for NLP tasks.
4.  [**Chatting with LLMs**](#4-chatting-with-llms): Interacting with pre-trained models using the Hugging Face ecosystem.
5.  [**Hugging Face Inference: From Manual Steps to the `pipeline()`**](#5-hugging-face-inference-from-manual-steps-to-the-pipeline): Contrasting manual inference with the high-level `pipeline()` function.

---

## 1. N-Gram Language Models

* **Notebook**: `n-grams.ipynb`

This project delves into the statistical foundations of language modeling by building and analyzing N-gram models (Unigrams, Bigrams, and Trigrams). It provides a clear, step-by-step implementation of how to calculate word probabilities and generate text based on the patterns learned from a corpus.

### Key Concepts Covered:
-   **Language Modeling**: Understanding the core task of predicting the next word.
-   **Probability & Frequency Distribution**: Using `NLTK` to count word and n-gram frequencies.
-   **Conditional Probability**: The mathematical basis for predicting a word given its context (`P(word | context)`).
-   **Text Generation**: Implementing a simple generation loop to create new text based on the learned Unigram, Bigram, and Trigram models.

### Example: Bigram Prediction Function
This function predicts the most likely next word given a single context word, forming the core of the Bigram model.
```python
def make_predictions(context_words, freq_grams, normalize=1, vocabulary=vocabulary):
    """
    Predict the next word probabilities given a sequence of context words.
    """
    next_word_probs = {}
    context_size = len(next(iter(freq_grams)))
    tokens = preprocess(context_words)[:context_size - 1]
    
    for word in vocabulary:
        ngram = tokens + [word]
        prob = freq_grams.get(tuple(ngram), 0) / normalize if normalize != 0 else freq_grams.get(tuple(ngram), 0)
        next_word_probs[word] = prob
    
    return sorted(next_word_probs.items(), key=lambda x: x[1], reverse=True)

# Usage
my_words = "are"
predictions = make_predictions(my_words, freq_bigrams, normalize=fdist['are'])
print(predictions[0:3])
# Output: [('no', 0.333...), ('of', 0.0), ('inside', 0.0)]

```

## 2. Exploring Tokenization

**Notebook:** `exploring-tokenization-from-nltk-to-transformers.ipynb`

A comprehensive exploration of **tokenization**, the critical first step in any NLP pipeline.  
This notebook compares various strategies — from simple splits to the advanced subword algorithms that enable modern Transformers to handle vast vocabularies and out-of-vocabulary words gracefully.

### Tokenization Strategies Compared
- **Word-Based:** Classic tokenization using libraries like NLTK and spaCy.  
- **Character-Based:** A simple but limited approach of splitting text into individual characters.  
- **Subword-Based:** The modern standard, balancing vocabulary size and semantic meaning. Examples include:
  - **WordPiece** (used by BERT)  
  - **SentencePiece** (used by XLNet and T5)  

- **PyTorch `torchtext`:**  
  - Building a vocabulary from an iterator.  
  - Handling special tokens (`<unk>`, `<pad>`, `<bos>`, `<eos>`).  
  - Numericalizing text.

### Example: Subword Tokenization with BERT
> "tokenization" → `token` + `##ization`  
This allows the model to recognize the root word ("token") and its suffix ("ization") independently.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("He taught me tokenization.")

print(tokens)
# Output: ['he', 'taught', 'me', 'token', '##ization', '.']
```


## 3. NLP Data Pipelines with PyTorch

**Notebook:** `nlp-data-pipelines.ipynb`

This notebook focuses on the crucial engineering task of creating efficient, scalable data loading pipelines for NLP models. It demonstrates how to handle large, variable-length text datasets and prepare them for training deep learning models using PyTorch's powerful data utilities.

### Core Components Mastered:
- `torch.utils.data.Dataset`: Creating custom dataset classes to wrap our raw text data.
- `torch.utils.data.DataLoader`: For efficient batching, shuffling, and parallel processing of data.
- **Custom `collate_fn`**: The key to solving the variable-length sequence problem. This function groups individual samples into a batch, tokenizes them, and pads them to create uniform-sized tensors.

### End-to-End Example:
Building a complete data pipeline for a German-to-English machine translation task using the Multi30k dataset, including language-specific tokenizers and vocabularies.

### Example: Custom Collate Function for Padding
This function takes a batch of raw sentences, tokenizes and numericalizes them, and then pads each sequence to the maximum length in the batch.

```python
def collate_fn(batch):
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    # Pad sequences to have equal lengths within the batch
    padded_batch = pad_sequence(tensor_batch, batch_first=True, padding_value=PAD_IDX)
    return padded_batch

# The DataLoader uses this function to create perfectly shaped tensor batches
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
```




## 4. Chatting with LLMs

**Notebook:** `chat_with_llms.ipynb`

A practical, hands-on guide to interacting with powerful, pre-trained Large Language Models from the Hugging Face Hub. This project shows how to quickly build a conversational AI chatbot by leveraging off-the-shelf models.

### Models Explored:
- `facebook/blenderbot-400M-distill`
- `google/flan-t5-base`

### Methodology: The Generation Loop
The notebook implements a simple but effective loop that demonstrates the core interaction pipeline with an LLM:

1. **Load:** Use `AutoTokenizer` and `AutoModelForSeq2SeqLM` to download and load a pre-trained model.  
2. **Encode:** Convert user input text into a sequence of token IDs that the model can understand.  
3. **Generate:** Feed the token IDs to the model, which generates a sequence of output token IDs as a response.  
4. **Decode:** Convert the model's output token IDs back into human-readable text.  

### Example: The Chatbot Function
``` python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def chat_with_bot():
    while True:
        input_text = input("You: ")
        if input_text.lower() in ["quit", "exit", "bye"]:
            break

        # Tokenize input, generate response, and decode
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=150)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print("Chatbot:", response)

chat_with_bot()

```


## 5. Hugging Face Inference: From Manual Steps to the `pipeline()`

**Notebook:** `practical_guide_to_nlp_with_transformers.ipynb`

A hands-on guide to performing inference with the Hugging Face `transformers` library. This notebook introduces the two primary workflows for using pre-trained models, contrasting the detailed, manual process with the powerful, high-level `pipeline()` function for a variety of common NLP tasks.

### Approaches Demonstrated:
- **Manual Inference:** The foundational, step-by-step process of loading a tokenizer and model, preparing inputs, running inference, and manually post-processing the outputs to get a final result.
- **The `pipeline()` Function:** A high-level abstraction that simplifies the entire inference process into just two lines of code, automatically handling tokenization, inference, and decoding.

### Tasks Covered:
- **Text Classification** with `DistilBERT`
- **Text Generation** with `GPT-2` and `T5`
- **Fill-Mask** with `BERT`
- **Language Detection** with `XLM-Roberta`

---

