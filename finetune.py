import re
import pickle
import torch
from GPT2.utils import load_weight
from GPT2.encoder import get_encoder
from GPT2.model import GPT2LMHeadModel
from GPT2.config import GPT2Config
from matplotlib import pyplot as plt

def remove_urls_and_entities(text):
    """
    Removes URLs and HTML entities from a string using regular expressions.
    
    Args:
        text (str): The input string to remove URLs and HTML entities from.
        
    Returns:
        str: The input string with any URLs and HTML entities removed.
    """
    # Define regular expressions to match URLs and HTML entities
    url_pattern = re.compile(r'https?://(?:www\.\S+|(?!www)\S+)')
    entity_pattern = re.compile(r'&\w+;')
    
    # Use the sub() method to replace URLs and HTML entities with an empty string
    text_without_urls_and_entities = url_pattern.sub('', text)
    text_without_urls_and_entities = entity_pattern.sub('', text_without_urls_and_entities)
    
    return text_without_urls_and_entities.replace('\n', ' ').strip()


# Read the dictionary of tweet objects from the pickle object
data_path = 'D:\\5005-Data\\tweet_combined_with_sentiment.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print('Data read complete')

# Concatenate the rawContent attribute of each dictionary into a list
tweets_list = []
enc = get_encoder()
for k, tweet in data.items():
    content = remove_urls_and_entities(tweet.rawContent)
    content = '<T-Begin>' + content + '<T-End>'
    encoded = enc.encode(content)
    tweets_list.append(encoded)

max_len = max(len(t) for t in tweets_list)
print('Total number of tweets:', len(tweets_list))
print('Longest tweet by tokens number:', max_len)
del data

current = []
new_tweets_list = []
for t in tweets_list:
    current.extend(t)
    if len(current) >= max_len:
        current = current[:max_len]
        new_tweets_list.append(current)
        current = t

num_sample = len(new_tweets_list)
tweets_tensor = torch.tensor(new_tweets_list, dtype=torch.int64)
tweets_tensor = tweets_tensor.cuda()
print('Data tensor size:', tweets_tensor.size())

config = GPT2Config()
model = GPT2LMHeadModel(config)
model = model.cuda()

# Comment the two line below to train from scratch
state_dict = torch.load('gpt2-117m-pretrain.pt')
model = load_weight(model, state_dict)

optim = torch.optim.AdamW(model.parameters(), lr=0.00005)
train_len = max_len - 1
batch_size = 6
epoch_no = 2
step_no = 0

loss_history = []
scaler = torch.cuda.amp.GradScaler()

for _ in range(epoch_no):
    for i in range(0, num_sample, batch_size):
        step_no += 1
        if i + batch_size < num_sample:
            sample = tweets_tensor[i:i + batch_size]
        else:
            sample = tweets_tensor[i:]
        input_ids = sample[:, :train_len]
        label_ids = sample[:, 1:]

        optim.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model(input_ids, lm_labels=label_ids)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        if step_no % 30 == 0:
            loss = float(loss)
            loss_history.append((step_no, loss))
            print(f'Model loss at step {step_no}:', loss)

torch.save(model.state_dict(), 'tweet-gen-pretrain.pt')
print('Successfully saved check point')

x, y = zip(*loss_history)

plt.plot(x, y)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss over time with pretraining')

plt.show()