import spacy

# Load the saved pipeline
output_dir = "./textcat_model"
nlp = spacy.load(output_dir)

# Use the loaded model to make predictions
texts = ["This is a test text to classify", 
         "Another example text",
         "Anglemyer A, Horvath T, Rutherford G (2014). The accessibility of ﬁrearms and risk for suicide and homicide victimization among household members: a systematic review and meta-analysis. Annals of Internal Medicine 160, 101–110. doi: 10.7326/M13-1301."]
for text in texts:
    doc = nlp(text)
    print(text, doc.cats)
