from transformers import GPT2Tokenizer

def tokenize(s, type='gpt2-medium'):
	try:
		tokenizer = GPT2Tokenizer.from_pretrained(type)
	except OSError:
		return None
	return tokenizer.tokenize(s)
	