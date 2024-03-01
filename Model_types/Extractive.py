import os
import time

import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sumy.utils import get_stop_words
from lexrank.utils.text import tokenize
text = "Story time"
def lexrank():
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()

    summary = summarizer(parser.document, 5)  # Summarize the document with 5 sentences

    summary_text = ' '.join([str(sentence) for sentence in summary])
    print(summary_text)


def Luhn():
    from sumy.summarizers.luhn import LuhnSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LuhnSummarizer()

    # Generate the summary
    summary = summarizer(parser.document, 30)  # Summarize to 2 sentences
    summary_text = ' '.join([str(sentence) for sentence in summary])
    print(summary_text)


def split_into_chunks(text, max_length=1024):
    """
    Splits text into chunks of max_length, on sentence boundaries.
    """
    # Tokenize the text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)
    current_chunk = []
    chunks = []

    for sentence in sentences:
        if len(" ".join(current_chunk) + " " + sentence) <= max_length:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def distill_large_text(fdb):
    """
    Summarizes large text by processing chunks and then combining.
    """
    models = ["sshleifer/distilbart-cnn-12-6","sshleifer/distilbart-xsum-6-6",'Falconsai/text_summarization']

    # Load model directly


    for i in models:
        tokenizer = AutoTokenizer.from_pretrained(i)
        model = AutoModelForSeq2SeqLM.from_pretrained(i)
        results = []
        for message in fdb['CONTEXT']:

            chunks = split_into_chunks(message)

            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True,
                                   padding="max_length")
                start_time = time.time()
                summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0,
                                             num_beams=4, early_stopping=True)
                end_time = time.time()
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                text = ''.join(summary)
            results.append(text)
            execution_time = end_time - start_time
            print(execution_time)

        fdb['sum - ' + str(i)] = results
