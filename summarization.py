import argparse
from transformers import pipeline
import os

def main(input_file):
    """
    Summarize a crime story and answer key questions about the crime.

    Usage:
        python crime_story_summarizer.py --input_file inputs/crimeStory.txt
    """
    # Set environment variable for PyTorch MPS fallback
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Load the summarization pipeline
    print("Loading summarization model...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Read the text file
    with open(input_file, 'r') as f:
        text = f.read().strip()

    if not text:
        raise Exception("No text found in the input file.")

    print("=" * 80)
    print(f"Summarizing content from: {input_file}\n")
    
    # Generate summary
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    print("Summary:\n")
    print(summary[0]['summary_text'])

    print("=" * 80)

    # Load the question-answering pipeline
    print("Loading question-answering model...")
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

    questions = [
        "Who was the victim of the crime?",
        "What crime was committed?",
        "Who committed the crime?",
        "What evidence gave it away?",
    ]

    # Ask questions
    print("Answering questions:\n")
    for question in questions:
        answer = qa_model(question=question, context=text)
        print(f"Q: {question}")
        print(f"A: {answer['answer']} (Confidence: {answer['score']:.2%})")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize and analyze crime stories.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the crime story text file.")
    args = parser.parse_args()

    main(args.input_file)
