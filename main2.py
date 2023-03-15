import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import MT5ForConditionalGeneration, T5Tokenizer
# input_text = "They were there to enjoy us and they were there to pray for us."

def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences,num_beams,temperature,do_sample):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=1000,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=temperature,do_sample=do_sample,top_k=50,top_p=0.95)
  tgt_text = set()
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text


model_name1 = 'milyiyo/paraphraser-spanish-mt5-small'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer1 = T5Tokenizer.from_pretrained(model_name1)
model1 = MT5ForConditionalGeneration.from_pretrained(model_name1).to(device)

def get_response1(input_text,num_return_sequences,num_beams,temperature,do_sample):
    batch = tokenizer1([input_text], padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    translated = model1.generate(input_ids=batch['input_ids'],max_length=1000, attention_mask=batch['attention_mask'], num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=temperature,do_sample=do_sample,top_k=50,top_p=0.95, early_stopping=True)
    tgt_text = tokenizer1.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


# def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
#   # tokenize the text to be form of a list of token IDs
#   inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
#   # generate the paraphrased sentences
#   outputs = model.generate(
#     **inputs,
#     num_beams=num_beams,
#     num_return_sequences=num_return_sequences,
#   )
#   # decode the generated sentences using the tokenizer to get them back to text
#   return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# num_beams = 5
# num_return_sequences = 5
# context = "The use of technology has revolutionized the way we communicate with each other. With just a few clicks, we can now easily connect with people from all over the world and exchange ideas and information instantly"
# x = get_response(context,num_return_sequences,num_beams)
# print(x)