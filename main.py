import os

# Import packages
import pandas as pd
import logging
import warnings

from Model_types.Abstrative import text_gen_casual_llm
from Model_types.Extractive import Luhn, distill_large_text, lexrank

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.INFO)
pd.set_option('display.max_colwidth', None)


def select_model(model_type):
    """
    Asks the user to select a model based on the given model type and returns the selected model(s).
    """
    model_maps = {
        "1": {  # Casual LLM
            1: 'mistralai/Mistral-7B-Instruct-v0.2',
            2: 'Custom Model Path'  # Changed to a placeholder text
        },
        "2": {  # S2S
            1: 'MBZUAI/LaMini-Flan-T5-248M',
            2: 'google/flan-t5-base',
            3: 'google/flan-t5-xl',
            4: 'philschmid/bart-large-cnn-samsum',
            5: 'Custom Model Path'  # Changed to a placeholder text
        }
    }

    model_map = model_maps.get(model_type, {})
    selected_models = []

    if not model_map:
        print("Invalid model type selected.")
        return selected_models

    print("Which Model Would you Like to try?\n0 - Exit")
    for key, value in model_map.items():
        print(f"{key} - {value}")

    while True:
        selection = input("Enter model number(s), separated by commas or space (e.g., 1,3 or 2 4): ").strip()
        if selection == "0":
            break
        try:
            selected_indices = [int(num) for num in selection.replace(',', ' ').split()]
            for index in selected_indices:
                if index in model_map:
                    if model_map[index] == 'Custom Model Path':  # Check if the selection is for a custom model path
                        custom_path = input("Enter your own model path or Hugging Face model: ").strip()
                        selected_models.append(custom_path)
                    else:
                        selected_models.append(model_map[index])
                else:
                    print(f"Invalid selection: {index}. Please enter a valid model number.")
            if selected_models:
                break
        except ValueError:
            print("Invalid input. Please enter numbers only.")

    return selected_models


def handle_input_string():
    data = pd.read_csv(os.getcwd() + '/Resources/User.csv')

    question = []
    context = []
    # Collect inputs
    question_input = input("Please enter your instructions and/or question: ")
    context_input = input(
        "Please enter context you would like to use for your question, if no context just hit return: ")
    question.append(question_input)
    context.append(context_input)
    # Append the inputs as a new row in the DataFrame

    data['CONTEXT'] = context
    data['QUESTION'] = question

    data['data_dict_form'] = data.apply(create_data_dict, axis=1)
    print(data)

    return data


def list_csv_files(directory):
    """
    Lists all CSV files in the given directory.
    """
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    if not csv_files:
        print("No CSV files found in the directory.")
        return None
    else:
        print("Available CSV files:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}: {file}")
        return csv_files


def handle_csv_file():
    csv_files = list_csv_files(os.getcwd() + "/resources")
    if csv_files:
        file_index = input("Please enter the number of the CSV file you'd like to use: ")
        try:
            file_index = int(file_index) - 1  # Adjust for 0-based indexing
            if 0 <= file_index < len(csv_files):
                csv_file_path = os.path.join(os.getcwd() + "/resources", csv_files[file_index])
                print(f"Selected CSV file path: {csv_file_path}")
                data = pd.read_csv(csv_file_path)
                data['data_dict_form'] = data.apply(create_data_dict, axis=1)

            else:
                print("Invalid selection. Please enter a number corresponding to the CSV files listed.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return data


def main():
    models = []
    print("""Which Model Would you Like to try? 0 to exit\n
         1 - Abstractive\n
         2 - Summarization\n
         3 - Extractive""")
    selection = input("Type Number Only = ")
    models = select_model(selection)
    if models:
        print("You have selected the following model(s):")
        for model in models:
            print(model)
    else:
        print("No models selected.")

    print(f"""Please select the type of prompt you're working with:\n
    1: Just an input
    2: CSV file from DA/resources 'QUESTION' and 'CONTEXT' columns """)
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        prompt = handle_input_string()
    elif choice == '2':
        prompt = handle_csv_file()
    else:
        print("Invalid choice. Please enter either 1 or 2.")

    if int(selection) == 1:
        text_gen_casual_llm(prompt, models)

    data = pd.read_csv(os.getcwd() + "/resources/questions.csv")
    print(data.shape[0])

    filtered_df = data[['CONTEXT']]

    # Use a pipeline as a high-level helper
    from transformers import pipeline

    pipe = pipeline("text-generation", model="mistralai/Mixtral-8x7B-Instruct-v0.1")

    # Luhn()
    # print('\nLEX')
    # lexrank()
    # print('\nDISTBERT')
    # filtered_df = filtered_df.sample(3)
    # distill_large_text(filtered_df)

    filtered_df.to_csv(os.getcwd() + '/Resources/Summary.csv', index=True)


if __name__ == "__main__":
    main()
