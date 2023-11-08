from transformers import AutoTokenizer

label_list = ['O', 'B-PRODUCT', 'I-PRODUCT']
label_encoding_dict = {'B-PRODUCT': 1, 'I-PRODUCT': 2, 'O': 0}

model_checkpoint = "bert-base-uncased"
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_dataset(dataset):
    return dataset.map(tokenize_and_align_labels, batched=True)


def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True,
                                 is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[
                    word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
