import os
import pandas as pd

from scraper import WebScraper


def generate() -> None:
    fetch_data("sources.csv")
    split_for_annotation("scraped-raw")


def fetch_data(source_file: str) -> None:
    sources_df = load_sources(source_file)
    # Get the text from each URL
    wscraper = WebScraper(save_locally=True)
    # Iterate over the rows in the dataframe
    text_counter = 0
    for index, row in sources_df.iterrows():
        if text_counter == 100:
            break
        url = row['url']
        response = wscraper.get(url)
        if response and response.content:
            wscraper.extract_text_from_response(response)
            text_counter += 1


def load_sources(csv: str) -> pd.DataFrame:
    return pd.read_csv(csv)


def split_for_annotation(directory: str, save_dir='annotated') -> None:
    files = os.listdir(directory)
    train = files[:int(len(files) * 0.8)]
    # create train and test directories if they don't exist
    if not os.path.exists(f"{save_dir}/train"):
        os.makedirs(f"{save_dir}/train")
    if not os.path.exists(f"{save_dir}/test"):
        os.makedirs(f"{save_dir}/test")
    for file in files:
        if file in train:
            sub_dir = 'train'
        else:
            sub_dir = 'test'
        with open(f"{directory}/{file}", 'r') as input_file:
            text = input_file.read()
            tokens = text.split()
            with open(f"{save_dir}/{sub_dir}/{file}", 'w', encoding='utf-8') as f:
                for token in tokens:
                    f.write(f"{token}\tO\n")
                f.write("\n")
