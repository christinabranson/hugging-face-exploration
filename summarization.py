from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = None
with open('inputs/crimeStory2.txt', 'r') as f:
    text = f.read()

if not text:
    raise Exception("no text") 

print("="*80)

# Generate summary
summary = summarizer(text, max_length=200, min_length=50, do_sample=False)
print("Summary:", summary[0]['summary_text'])

print("="*80)

for question in [
    "Who was the victim of the crime?",
    "What crime was committed?",
    "Who committed the crime?",
    "What evidence gave it away?",
]:

    # QA
    qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
    model = qa_model(question = question, context = text)
    print(question, model)
