def generate_prompt(example: dict, model_name):
    """
    Generates a standardized input prompt with an instruction, context, question and a 'response'.
    :param c:
    :param example: dictionary with keys "context", "question", and "response".
    :param with_response: specifies if the response should be included in the prompt or not.
    :return: prompt string with the {instruction}, {context}, {question}, (and) {answer}.
    """

    if model_name == "google":
        return (f"""Use the context to answer the following question below
        {example['support']}.Now use it to answer this question.\nQuestion: {example['question']}""")
    if model_name == "MBZUAI":
        return (f"""###You are a helpful code assistant. Your task is to summarize context and use it to answer the Question. The rules are as follows
                  Use the context below to accurately answer the question. Always use polite language. Only use information provided, and do not make information up.
                  Context:{example['support']}\nQuestion:{example['question']}""")
    if model_name == "philschmid":
        return (f"""###You are a helpful code assistant. Your task is to summarize context and use it to answer the Question. The rules are as follows
                  Use the context below to accurately answer the question. Always use polite language. Only use information provided, and do not make information up.
                  Context:{example['support']}\n\nQuestion:{example['question']}""")
    if model_name == "microsoft":
        return ("Instruct: <prompt>\nOutput:")
    if model_name == "mistralai":
        bos_token = "<s>"
        original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
        system_message = "Use the provided input to create an instruction that could have been used to generate the response with an LLM. "
        response = example['question']
        input = example['support']
        eos_token = "</s>"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += "[INST]]"
        full_prompt += original_system_message + "\n" + response + "\n" + input + "\n"
        full_prompt += "[/INST]]"
        full_prompt += eos_token
        return str(full_prompt)
    return ("FAILED DO NOT GENERATE ANSWER")