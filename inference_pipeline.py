#!/usr/bin/env python
import click
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification

from dataset.tokenize import label_list
from scraper import WebScraper


@click.command()
@click.option('--url', required=True, help='URL to scrape')
def get_product_from_url(url: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained("./product-ner.model/")
    ws = WebScraper()
    response = ws.get(url)
    if response and response.content:
        text = ws.extract_text_from_response(response)
        # truncate in order to fit the 512 token limit
        inputs = tokenizer(text, truncation=True)
        # torch.tensor(inputs["input_ids"]).unsqueeze(0).size()

        model = AutoModelForTokenClassification.from_pretrained(
            "./product-ner.model/")
        # add another dimension to the input
        # and attn tensors to match BERT input shape
        predictions = model.forward(
            input_ids=torch.tensor(inputs['input_ids']).unsqueeze(0),
            attention_mask=torch.tensor(inputs['attention_mask']).unsqueeze(0))

        # get the index of the highest probability
        predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
        # convert the index to the label
        predictions = [label_list[i] for i in predictions]
        # convert the input_ids to words
        words = tokenizer.batch_decode(inputs["input_ids"])
        product = extract_product_from_predictions(predictions, words)
        print(f"Product: {product}")
    else:
        print(f"No valid response from {url}")


def extract_product_from_predictions(predictions: list, words: list) -> str:
    product = []
    for i, prediction in enumerate(predictions):
        if prediction == "B-PRODUCT":
            product.append(words[i])
        elif prediction == "I-PRODUCT":
            product.append(words[i])
        elif prediction == "O":
            if len(product) > 0:
                return " ".join(product)
    return " ".join(product)


if __name__ == '__main__':
    get_product_from_url()
