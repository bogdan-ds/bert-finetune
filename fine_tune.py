import numpy as np
from datasets import load_metric
from transformers import AutoModelForTokenClassification, TrainingArguments, \
    Trainer
from transformers import DataCollatorForTokenClassification

from dataset.dataset import get_scraped_token_dataset
from dataset.tokenize import tokenize_dataset, tokenizer, model_checkpoint, \
    label_list


train_dir = './dataset/annotated/train/'
test_dir = './dataset/annotated/test/'
batch_size = 4

train_dataset, test_dataset = get_scraped_token_dataset(train_dir, test_dir)
tokenized_train = tokenize_dataset(train_dataset)
tokenized_test = tokenize_dataset(test_dataset)

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint,
                                                        num_labels=len(
                                                            label_list))

args = TrainingArguments(
    f"test-ner",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=9,
    weight_decay=1e-5,
)

# data collator handles padding, attention masks, etc.
data_collator = DataCollatorForTokenClassification(tokenizer)
# seqeval is used because the entities span multiple tokens
metric = load_metric("seqeval")


def compute_metrics(p: tuple) -> dict:
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100] for
        prediction, label in zip(predictions, labels)]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100] for
        prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions,
                             references=true_labels)
    return {"precision": results["overall_precision"],
            "recall": results["overall_recall"], "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model('product-ner.model')
