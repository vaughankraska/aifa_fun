import spacy
from spacy.training.example import Example
import os

# Initialize the blank pipeline
nlp = spacy.blank("en")
textcat = nlp.add_pipe("textcat")
textcat.add_label("REFERENCE")
textcat.add_label("NORMAL")

# Your training data should be defined here
train_data = [
    ("This is a reference text", {"cats": {"REFERENCE": 1.0, "NORMAL": 0.0}}),
    ("This is a normal text", {"cats": {"REFERENCE": 0.0, "NORMAL": 1.0}}),
    # Add more training examples here
]

# Training the pipeline
loss_list = []
optimizer = nlp.begin_training()
for i in range(20):
    losses = {}
    for text, annotations in train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], sgd=optimizer, losses=losses)
    print(losses)
    loss_list.append(losses)

# Save the trained pipeline
output_dir = "./textcat_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
nlp.to_disk(output_dir)
print(f"Saved model to {output_dir}")
