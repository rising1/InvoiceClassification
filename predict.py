import torch
import constants
from my_tokenizer import tokenize

model = torch.load(constants.PATH_TO_SAVED_MODEL + "invoice_model_" + ".pt")
model.eval()
model = model.to("cpu")
# print(model)

text = 'BROCKWELL Fuel - 1000L'


def predict(text):
    with torch.no_grad():
        text = torch.tensor(tokenize(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()

print(text, " allocates to ", constants.CLASSES[predict(text)])