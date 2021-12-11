import discord
import difflib

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

import pandas as pd
import numpy as np
import os

#-----
#model
#-----

print('get the model ready...')

label_dict = {'bye': 0,
              'cloud': 1,
              'general_weather': 2,
              'good_bad': 3,
              'greeting': 4,
              'rain': 5,
              'sun_hours': 6,
              'sunny': 7,
              'temperature': 8,
              'unknown': 9}

label_dict_inverse = {v: k for k, v in label_dict.items()}

bert = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
dir_path = os.path.dirname(os.path.realpath(__file__))

print(dir_path)
bert.load_state_dict(torch.load(f'{dir_path}/BERT.model', map_location=torch.device('cpu')))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#---------
#functions
#---------

def similarity(word, pattern):
    return difflib.SequenceMatcher(a=word.lower(), b=pattern.lower()).ratio()

def get_intent(string):
    question = [string]

    data = {'Question': question}
    df = pd.DataFrame(data=data)
    value_to_predict = df.Question.values

    encoded_data = tokenizer.batch_encode_plus(
        value_to_predict,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_id = encoded_data['input_ids']
    attention_mask = encoded_data['attention_mask']
    value_to_predict_tensor = TensorDataset(input_id, attention_mask)

    dataloader_prediction = DataLoader(value_to_predict_tensor,
                                       sampler=SequentialSampler(value_to_predict_tensor),
                                       batch_size=1)
    bert.eval()

    prediction = []

    for batch in dataloader_prediction:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1]
                  }

        with torch.no_grad():
            outputs = bert(**inputs)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        prediction.append(logits)

    prediction = np.concatenate(prediction, axis=0)

    prediction_flat = np.argmax(prediction, axis=1).flatten()

    intent = prediction_flat[0]
    #print(prediction_flat[0])
    #print(type(prediction_flat[0]))
    return intent

#-----------
#discord bot
#-----------

print('starting the bot...')


# usually implemented in an .env file
discord_token = 'OTE1OTI1ODgyMzc0MzQwNjM4.Yaiscw.dDy4ShZak3_-L2WQZCtIO077Xoc'
discord_server = '915925218504106004'
# end of .env file

client = discord.Client()

@client.event
async def on_ready():
    print(f'{client.user.name} has connected to Discord!')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    content = message.content
    print(content)
    print(type(content))
    if content != '':
        channel = getattr(message, 'channel')

        intent = get_intent(content)

        response = label_dict_inverse.get(intent)

        await channel.send(response)

client.run(discord_token)