import pandas as pd
from tabulate import tabulate
from utils import get_model, get_score, split_data


def main():
    results = []

    for dataset_num in range(1, 4):
        dataset = pd.read_csv(f"../data/preprocessed_movie_data_{dataset_num}.csv")
        X_train, X_test, y_train, y_test = split_data(
            dataset.drop("vote_average", axis=1), dataset["vote_average"], percentage=0.2
        )

        for model_name in [
            "Linear Regression",
            "Ridge Regression",
            "Lasso Regression",
            "Elastic Net Regression",
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "LightGBM",
            "CatBoost",
            "KNN",
        ]:
            model = get_model(model_name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = get_score(y_test, y_pred)
            results.append([f"Dataset {dataset_num}", model_name, score])
            print(f"Dataset {dataset_num} - {model_name} - {score}")

    df_results = pd.DataFrame(results, columns=["Dataset", "Model", "Score"])
    pivot_table = df_results.pivot(index="Dataset", columns="Model", values="Score")
    table = tabulate(pivot_table, headers="keys", tablefmt="fancy_grid")
    print(table)


if __name__ == "__main__":
    main()
