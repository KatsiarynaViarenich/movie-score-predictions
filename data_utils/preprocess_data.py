import ast
import os

import pandas as pd
import torch
import typer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertModel, BertTokenizer


def _transform_language_column(df, columns, desired_columns_list):
    desired_languages = ["en", "ja", "ko", "es", "fr"]
    for column_name, desired_columns in zip(columns, desired_columns_list):
        df[column_name] = df[column_name].apply(lambda x: x if x in desired_languages else "other")
        df[column_name] = df[column_name].apply(lambda x: x + f"_{column_name}")
        one_hot = pd.get_dummies(df[column_name])
        df = pd.concat([df, one_hot], axis=1)
        df.drop([column_name], axis=1, inplace=True)
    return df


def _transform_categorical_columns(df, categorical_columns):
    df = pd.get_dummies(df, columns=categorical_columns)
    return df


def _transform_list_columns(df, columns, desired_columns_list):
    df_encoded = df
    for column, desired_columns in zip(columns, desired_columns_list):
        if isinstance(df[column][0], str):
            df[column] = df[column].apply(ast.literal_eval)
        mlb = MultiLabelBinarizer()
        encoded_df = pd.DataFrame(mlb.fit_transform(df[column]), columns=mlb.classes_)
        if desired_columns is not None:
            encoded_df = encoded_df[desired_columns]
            encoded_df.columns = [f"{col}_{column}" for col in encoded_df.columns]
            encoded_df[f"other_{column}"] = encoded_df.sum(axis=1) == 0
        df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
    df_encoded = df_encoded.drop(columns, axis=1)
    return df_encoded


def _drop_unnecessary_columns(df, columns_to_drop):
    df = df.drop(columns_to_drop, axis=1)
    return df


def _transform_columns_to_bool(df, columns):
    for column in columns:
        df[str(column) + "_bool"] = df[column].notnull()
    df.drop(columns, axis=1, inplace=True)
    return df


def _transform_columns_to_year(df, columns):
    for column in columns:
        df[column] = pd.to_datetime(df[column], format="%Y-%m-%d")
        df[column] = df[column].apply(lambda x: x.year)
        df[column] = df[column].fillna(df[column].median())
    return df


def _extract_values_from_dict_list(dict_str, key):
    list_of_dicts = ast.literal_eval(dict_str)
    result_list = []
    for dictionary in list_of_dicts:
        result_list.append(dictionary[key])
    return result_list


def _transform_columns_with_dicts(df, columns, keys):
    for column, key in zip(columns, keys):
        df[column] = df[column].apply(lambda x: _extract_values_from_dict_list(x, key))
    return df


def _encode_text_bert(df, columns):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    for column in columns:
        encoded_text = []
        for text in df[column]:
            encoded_text.append(tokenizer.encode(text, add_special_tokens=True))
        encoded_text = torch.tensor(encoded_text)
        with torch.no_grad():
            last_hidden_states = model(encoded_text)[0]
        df[column] = last_hidden_states[:, 0, :].numpy().tolist()


def _encode_text_tf_idf(df):
    df["text_combined"] = df["title"] + " " + df["overview"] + " " + df["tagline"]
    df["text_combined"] = df["text_combined"].fillna("")
    vectorizer = TfidfVectorizer()
    text_combined_features = vectorizer.fit_transform(df["text_combined"])
    text_combined_df = pd.DataFrame(
        text_combined_features.toarray(), columns=vectorizer.get_feature_names()
    )
    df = pd.concat([df, text_combined_df], axis=1)
    df = df.drop(["text_combined", "title", "overview", "tagline"], axis=1)
    return df


def numeric_features_dataset_creation():
    dataset = pd.read_csv("../data/movies.csv", header=0, sep=",", lineterminator="\n")
    columns = ["budget", "popularity", "revenue", "runtime", "vote_average"]
    dataset = dataset[columns]
    dataset.to_csv("../data/preprocessed_movie_data_1.csv", index=False)


def some_features_creation():
    dataset = pd.read_csv("../data/movies.csv", header=0, sep=",", lineterminator="\n")
    columns = [
        "budget",
        "popularity",
        "revenue",
        "runtime",
        "vote_average",
        "vote_count",
        "genre_ids",
        "homepage",
        "release_date",
    ]
    dataset = dataset[columns]
    dataset = _transform_list_columns(dataset, ["genre_ids"])
    dataset = _transform_columns_to_bool(dataset, ["homepage"])
    dataset = _transform_columns_to_year(dataset, ["release_date"])
    dataset.to_csv("../data/preprocessed_movie_data_2.csv", index=False)


def all_features_dataset_creation():
    dataset = pd.read_csv("../data/movies.csv", header=0, sep=",", lineterminator="\n")
    columns_to_drop = [
        "backdrop_path",
        "belongs_to_collection",
        "id",
        "imdb_id",
        "poster_path",
        "genres",
        "original_title",
        "title",
        "overview",
        "tagline",
        "production_companies",
    ]
    dataset = _drop_unnecessary_columns(dataset, columns_to_drop)
    dataset = _transform_columns_with_dicts(
        dataset, ["production_countries", "spoken_languages"], ["iso_3166_1", "iso_639_1"]
    )
    dataset = _transform_list_columns(
        dataset,
        ["genre_ids", "production_countries", "spoken_languages"],
        [
            None,
            [
                "US",
                "GB",
                "JP",
                "FR",
                "CA",
                "DE",
                "KR",
            ],
            ["en", "es", "ja", "fr", "it", "de", "ko", "zh", "ru", "cn"],
        ],
    )
    dataset = _transform_language_column(
        dataset, ["original_language"], [["en", "ja", "ko", "es", "fr", "zh", "it"]]
    )
    dataset = _transform_categorical_columns(dataset, ["status"])
    dataset = _transform_columns_to_bool(dataset, ["homepage"])
    dataset = _transform_columns_to_year(dataset, ["release_date"])
    dataset.to_csv("../data/preprocessed_movie_data_3.csv", index=False)


def main(dataset_number: int = 3):
    if os.path.exists(f"../data/preprocessed_movie_data_{dataset_number}.csv"):
        print(f"Dataset preprocessed_movie_data_{dataset_number} already exists")
    elif dataset_number == 1:
        numeric_features_dataset_creation()
        print("Dataset preprocessed_movie_data_1 was created")
    elif dataset_number == 2:
        some_features_creation()
        print("Dataset preprocessed_movie_data_2 was created")
    elif dataset_number == 3:
        all_features_dataset_creation()
        print("Dataset preprocessed_movie_data_3 was created")
    else:
        print("Wrong dataset number")


typer.run(main)
