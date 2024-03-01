import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from util.Prompcreator import generate_prompt


def generate_model_response_CLLM(model, tokenizer, message, model_name) -> str:
    """
    Generates response to answer the question given in the prompt based on the context FOR CASUAL LLM
    :param model: the LM used to generate the answer.
    :param tokenizer: the tokenizer to be used to tokenize the input context & question.
    :param message: either the full prompt str in expected template or the dictionary with the context and question keys.
    :return: generated response by the LM to the question in the prompt.
    """
    prompt = generate_prompt(message, model_name)
    inputs = tokenizer.encode("summarize: " + prompt, return_tensors="pt", truncation=True).to('mps')
    output = model.generate(inputs,
                            max_new_tokens=500,
                            min_length=65,
                            # The maximum numbers of new tokens to generate, ignoring the number of tokens in the prompt.
                            # temperature = 0.8,       # The value used to modulate the next token probabilities.
                            num_beams=5,  # Number of beams for beam search. Default = 1
                            no_repeat_ngram_size=4,  # All ngrams of that size can only occur once.
                            early_stopping=True,  # Controls the stopping condition for beam-based methods
                            length_penalty=1.0,
                            )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def text_gen_casual_llm(dict,CLLM):
    """
          MODEL IS CASUAL-LLM
          Generates the prompt based of context and question columns
          :param table: data frame with the rows `context`, `question`, and `response` and 'data_dict_form'.
          :return: appends table and adds new column based on the llm used and the results
    """

    # Load model directly
    for i in CLLM:
        cut_model = i.split("/")
        model_name = cut_model[0]
        logging.info(f"Loading {i} from Hugging Face")
        model, tokenizer = load_model_Casual_Llm(i)

        results = []
        for message in dict['data_dict_form']:

            logging.info(f"Generating Prompt Using {i} ")
            summary = generate_model_response_CLLM(model, tokenizer, message, model_name)
            parts = summary.rsplit('[/INST]]', 1)

            # Print everything after the last [/INST]]
            if len(parts) > 1:  # Check if the split was successful
                results.append(parts[1].strip())
                # Use strip() to remove any leading/trailing whitespace
            else:
                print("The tag [/INST]] does not exist or is not repeated.")
                results.append("-FAILED--FAILED--FAILED--FAILED--FAILED-")

        logging.info(f"Appending {i} Results To A New Column ")
        dict["Result"] = results
        dict.to_csv(os.getcwd() + '/Resources/User.csv', index=True)
        print(results)


def load_model_Casual_Llm(model_name: str, **dtype_kwargs: dict):
    """
    Load the tokenizer and the model from hugging face
    :param model_name: the model name in hugging face (str)
    :param dtype_kwargs: additional keyword args related to quantization etc.
    :return: the SEQ2SEQ LLM and tokenizer
    """
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.float16,
                                                 **dtype_kwargs
                                                 )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer
