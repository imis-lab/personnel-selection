import csv
import pandas
import json


def convert_json_dataset_to_csv(input_filename: str, output_filename: str, unassigned_issues: bool = True):
    """"Convert a JSON file with Jira issues into a CSV file.
    :param input_filename: the path for the JSON input file
    :param output_filename: the path of the CSV output file
    :param unassigned_issues: if true then the issues without an assignee are also kept
    """
    with open(input_filename) as f:
        issues = json.load(f)['issues']
    with open(output_filename, 'w',
        encoding = 'utf-8-sig', errors = 'ignore') as f:
        fieldnames = ['label', 'text']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for issue in issues:
            label = issue['fields']['assignee'].get('key', None)
            if unassigned_issues is False and label is None:
                continue
            summary = issue['fields']['summary']
            description = issue['fields'].get('description', '')
            if description is None:
                description = ''
            text = summary + ' ' + description
            writer.writerow({
                'label': label,
                'text': text,
            })

def convert_csv_to_json_dataset(input_filename: str, output_filename: str):
    """"
    Convert a csv file with Jira issues to a json file.
    This is done by reading the csv into a pandas dataframe,
    dropping the index column, while using it to generate a dict
    representation of each row. This results in the creation of a
    json object which contains an array of issues. Each issue is
    a json-like dict object, which is written into a file.
    """

    csv = pandas.read_csv(r'C:\Users\USER\Desktop\issues.csv') \
                .drop('index', axis = 1).to_dict('index')

    json_object = {'issues': [val for val in csv.values()]}

    with open(r'C:\Users\USER\Desktop\issues.json', 'w',
        encoding = 'utf-8-sig', errors = 'ignore') as file:
        file.write(json.dumps(json_object, indent = 4, separators = (',', ': ')))

if __name__ == '__main__': convert_csv_to_json_dataset(
    r'C:\Users\USER\Desktop\issues.csv', 
    r'C:\Users\USER\Desktop\issues.json')