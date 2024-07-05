import spacy

# Load the saved pipeline
output_dir = "./textcat_model"
nlp = spacy.load(output_dir)

# Use the loaded model to make predictions
texts = ["This is a test text to classify", "Another example text"]
for text in texts:
    doc = nlp(text)
    print(text, doc.cats)
