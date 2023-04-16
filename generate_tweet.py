import torch
import random
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder


def generate_tweet(n=10, params_location='tweet-gen-with-pretrain.pt'):

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(params_location, map_location='cpu' if not torch.cuda.is_available() else None)

    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    context_tokens = enc.encode('<T-Begin>')
    tweets = []

    while True:
        out = sample_sequence(
            model, length=200,
            context=context_tokens, batch_size=1, 
            temperature=1.2, top_k=30, device=device
        )
        out = out[0].tolist()
        text = enc.decode(out)
        text = text.split('<T-End>')
        text.pop() # Remove incomplete last tweet
        tweets.extend([t.replace('<T-Begin>', '') for t in text])
        if len(tweets) > n:
            break
    
    for i in range(n):
        print(f'{i + 1}:', tweets[i])

    return tweets


if __name__ == '__main__':
    generate_tweet(10, params_location='tweet-gen-pretrain.pt')