import csv
import json


def convert_json_dataset_to_csv(input_filename: str, output_filename: str, unassigned_issues: bool = True):
    """"Convert a JSON file with Jira issues into a CSV file.

    :param input_filename: the path for the JSON input file
    :param output_filename: the path of the CSV output file
    :param unassigned_issues: if true then the issues without an assignee are also kept
    """
    with open(input_filename) as f:
        issues = json.load(f)['issues']
    with open(output_filename, 'w') as f:
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


convert_json_dataset_to_csv('all_issues_minified.json', 'dataset.csv', False)
