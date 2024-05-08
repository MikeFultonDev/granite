from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cpu" # or "cuda"
model_path = "ibm-granite/granite-3b-code-instruct" # pick anyone from above list

tokenizer = AutoTokenizer.from_pretrained(model_path)

# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()

# change input text as desired
input_text = "As an expert Python programmer, what package is used for terminal output?"
# tokenize the text
input_tokens = tokenizer(input_text, return_tensors="pt")

# transfer tokenized inputs to the device
for i in input_tokens:
    input_tokens[i] = input_tokens[i].to(device)

# generate output tokens
output = model.generate(**input_tokens, max_length=40)
# decode output tokens into text
output = tokenizer.batch_decode(output)

# loop over the batch to print, in this example the batch size is 1
for i in output:
    print(i)
