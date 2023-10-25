import pandas as pd
import typer
from utils import get_model, get_score, split_data


def run(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def main(
    model_name: str = typer.Option(default="Linear Regression"),
    dataset_number: int = typer.Option(default=3),
):
    if dataset_number in [1, 2, 3]:
        dataset = pd.read_csv(
            f"../data/preprocessed_movie_data_{dataset_number}.csv",
            header=0,
            sep=",",
            lineterminator="\n",
        )
    else:
        raise ValueError("No such dataset at the moment")
    X_train, X_test, y_train, y_test = split_data(
        dataset.drop("vote_average", axis=1), dataset["vote_average"], percentage=0.2
    )
    model = get_model(model_name)
    y_pred = run(model, X_train, y_train, X_test)
    score = get_score(y_test, y_pred)
    print(f"Score for {model_name} is {score}")
    return score


typer.run(main)
