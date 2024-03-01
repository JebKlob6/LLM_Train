import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from util.Prompcreator import generate_prompt


def generate_model_response(model, tokenizer, message, model_name) -> str:
    """
    Generates response to answer the question given in the prompt based on the context
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
                            num_beams=5,  # Number of beams for beam search. Default = 1
                            no_repeat_ngram_size=4,  # All ngrams of that size can only occur once.
                            early_stopping=True,  # Controls the stopping condition for beam-based methods
                            length_penalty=1.0,
                            )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def load_model_S2S(model_name: str, **dtype_kwargs: dict):
    """
    Load the tokenizer and the model from hugging face
    :param model_name: the model name in hugging face (str)
    :param dtype_kwargs: additional keyword args related to quantization etc.
    :return: the SEQ2SEQ LLM and tokenizer
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                  device_map="auto",
                                                  trust_remote_code=True,
                                                  torch_dtype=torch.float16,
                                                  **dtype_kwargs
                                                  )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def text_gen_seq2seq(table, dbf, models2s):
    """
      MODEL IS Seq2SeqLM-LLM
      Generates the prompt based of context and question columns
      :param table: data frame with the rows `context`, `question`, and `response` and 'data_dict_form'.
      :return: appends table and adds new column based on the llm used and the results
    """
    # os.path.expanduser('~/PycharmProjects/LLM_Train/trained_model')
    # 'MBZUAI/LaMini-Flan-T5-248M', 'google/flan-t5-base','google/flan-t5-xl','philschmid/bart-large-cnn-samsum'
    # print(tabulate(table, headers='keys', tablefmt='psql'))
    for i in models2s:

        logging.info(f"Loading {i} from Hugging Face")
        cut_model = i.split("/")
        model_name = cut_model[0]
        results = []
        model, tokenizer = load_model_S2S(i)
        for message in table['data_dict_form']:
            logging.info(f"Generating Prompt Using {i} ")

            summary = generate_model_response(model, tokenizer, message, model_name)
            results.append(summary)

        logging.info(f"Appending {i} Results To A New Column ")
        dbf[cut_model[-1] + " - summarization"] = results