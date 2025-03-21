import argparse
from transformers import pipeline
import os

SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
QA_MODEL = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
FACT_EXTRACTION_MODEL = "facebook/bart-large-mnli"

def extract_facts(text, classifier):
    """
    Extract key facts from the text using zero-shot classification.
    """
    # Define categories we want to identify
    categories = [
        "crime details",
        "location information",
        "timeline of events",
        "people involved",
        "evidence found",
        "motive",
        "outcome"
    ]
    
    facts = []
    # Split text into smaller chunks for better processing
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    
    for chunk in chunks:
        # For each chunk, classify against our categories
        results = classifier(chunk, candidate_labels=categories, multi_label=True)
        
        # If any category has high confidence, consider it a fact
        for label, score in zip(results['labels'], results['scores']):
            if score > 0.5:  # Confidence threshold
                facts.append(f"{label}: {chunk[:200]}...")
    
    return facts

def main(input_file):
    """
    Summarize a crime story, extract key facts, and answer questions about the crime.

    Usage:
        python crime_story_summarizer.py --input_file inputs/crimeStory.txt
    """
    # Set environment variable for PyTorch MPS fallback
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Load the summarization pipeline
    print("Loading summarization model...")
    summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL)

    # Load the fact extraction pipeline
    print("Loading fact extraction model...")
    fact_classifier = pipeline("zero-shot-classification", model=FACT_EXTRACTION_MODEL)

    # Read the text file
    with open(input_file, 'r') as f:
        text = f.read().strip()

    if not text:
        raise Exception("No text found in the input file.")

    print("=" * 80)
    print(f"Analyzing content from: {input_file}\n")
    
    # Generate summary
    summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
    print("Summary:\n")
    print(summary[0]['summary_text'])

    print("\n" + "=" * 80)
    print("Key Facts and Important Points:\n")
    
    # Extract and display facts
    facts = extract_facts(text, fact_classifier)
    for i, fact in enumerate(facts, 1):
        print(f"{i}. {fact}")
        print("-" * 40)

    print("\n" + "=" * 80)

    # Load the question-answering pipeline
    print("Loading question-answering model...")
    qa_model = pipeline('question-answering', model=QA_MODEL, tokenizer=QA_MODEL)

    questions = [
        "Who was the victim of the crime?",
        "What crime was committed?",
        "Who committed the crime?",
        "What evidence gave it away?",
        "What was the motive?",
        "What was the weapon used?",
    ]

    # Ask questions
    print("Answering questions:\n")
    for question in questions:
        answer = qa_model(question=question, context=text)
        print(f"Q: {question}")
        print(f"A: {answer['answer']} (Confidence: {answer['score']:.2%})")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize, extract facts, and analyze crime stories.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the crime story text file.")
    args = parser.parse_args()

    main(args.input_file)
