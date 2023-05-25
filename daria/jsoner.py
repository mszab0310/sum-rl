xlsx_file = "C:/Users/mszab/Desktop/output_1684875991.xlsx"
json_file = "C:/Users/mszab/Desktop/intents.json"
# Call the function to convert XLSX to JSON

import pandas as pd
import json


def xlsx_to_json(xlsx_file, json_file):
    # Read the XLSX file into a pandas DataFrame
    df = pd.read_excel(xlsx_file)

    # Create a dictionary to store the intents
    intents = {}
    category_set = set()
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the text and category from the row
        text = str(row[0])
        category = str(row[1])

        category_set.add(category)
        print(category_set)

        # Check if the category already exists in the intents dictionary
        if category in intents:
            intents[category]["patterns"].append(text)
        else:
            intents[category] = {
                "tag": category,
                "patterns": [text],
                "responses": category
            }

    # Create a list to store the final intents
    final_intents = list(intents.values())

    # Create a dictionary to store the intents list
    data = {
        "intents": final_intents
    }

    # Convert the dictionary to JSON
    json_data = json.dumps(data, indent=4)

    # Save the JSON to a file
    with open(json_file, 'w') as f:
        f.write(json_data)
    print(json_data)

# Call the function to convert XLSX to JSON and save it to a file
xlsx_to_json(xlsx_file, json_file)