import csv


def save_predictions(pred, file):

    """

    Save predictions to CSV file

    Args:
        pred: list, of dictionaries
        file: str, filename + extension

    """

    with open(file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['Headline', 'Body ID', 'Stance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for prediction in pred:
            writer.writerow(prediction)