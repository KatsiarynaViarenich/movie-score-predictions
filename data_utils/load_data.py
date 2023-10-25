from time import sleep

import pandas as pd
import requests
from tqdm import tqdm, trange

from api_key import API_key


def fetch_movie_data(num_pages):
    movie_df = pd.DataFrame()
    for page in trange(1, num_pages + 1):
        url = f"https://api.themoviedb.org/3/discover/movie?page={page}&sort_by=popularity.desc&api_key={API_key}"
        response = requests.get(url).json()
        movie_df = pd.concat([movie_df, pd.DataFrame(response["results"])], ignore_index=True)
    return movie_df


def enrich_movie_data(movie_df):
    request_counter = 0
    columns = [
        "adult",
        "backdrop_path",
        "belongs_to_collection",
        "budget",
        "genres",
        "homepage",
        "id",
        "imdb_id",
        "original_language",
        "original_title",
        "overview",
        "popularity",
        "poster_path",
        "production_companies",
        "production_countries",
        "release_date",
        "revenue",
        "runtime",
        "spoken_languages",
        "status",
        "tagline",
        "title",
        "video",
        "vote_average",
        "vote_count",
    ]
    movie_info_df = pd.DataFrame(columns=columns)

    for movie_id in tqdm(movie_df["id"]):
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_key}"
        data = requests.get(url).json()
        new_row = pd.Series(data)
        movie_info_df = movie_info_df.append(new_row, ignore_index=True)

        request_counter += 1
        if request_counter % 10 == 0:
            sleep(1)

    exclude_columns = set(movie_df.columns) & set(movie_info_df.columns)
    enriched_df = pd.concat([movie_df.drop(columns=exclude_columns), movie_info_df], axis=1)
    return enriched_df


if __name__ == "__main__":
    movie_df = fetch_movie_data(500)
    enriched_df = enrich_movie_data(movie_df)
    enriched_df.to_csv("../data/movies.csv", index=False)
