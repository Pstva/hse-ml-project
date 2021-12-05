import click
from src.read_prepare_data import (
    read_cancer_dataset,
    read_spam_dataset,
    train_test_split,
    normalize
)
from src.metrics import get_precision_recall_accuracy, plot_precision_recall, plot_roc_curve
from src.knn import KNearest


@click.group()
def cli():
    pass
    

    
@cli.command() 
@click.argument("dataset_path", type=click.Path())
@click.argument("output_path_recall", type=click.Path())
@click.argument("output_path_precision", type=click.Path())
@click.argument("output_path_accuracy", type=click.Path())
def precision_recall(dataset_path, output_path_recall, output_path_precision, output_path_accuracy):
    if dataset_path.find("spam") > 0:
        X, y = read_spam_dataset(dataset_path)
    elif dataset_path.find("cancer") > 0:
        X, y = read_cancer_dataset(dataset_path)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    X_train, X_test = normalize(X_train, X_test)
    print('plotting')
    plot_precision_recall(X_train, y_train, X_test, y_test, [output_path_recall, output_path_precision, output_path_accuracy], max_k=20)
    

@cli.command() 
@click.argument("dataset_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def roc_curve(dataset_path, output_path):
    if dataset_path.find("spam") > 0:
        X, y = read_spam_dataset(dataset_path)
    elif dataset_path.find("cancer") > 0:
        X, y = read_cancer_dataset(dataset_path)
    X_train, y_train, X_test, y_test = train_test_split(X, y, 0.9)
    X_train, X_test = normalize(X_train, X_test)
    print('plotting')
    plot_roc_curve(X_train, y_train, X_test, y_test, output_path, max_k=30)

if __name__ == "__main__":
    cli()

