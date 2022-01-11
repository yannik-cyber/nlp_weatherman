import discord
import difflib

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

import pandas as pd
import numpy as np
import os
import json
import requests
import random
from datetime import datetime
import spacy

from geopy.geocoders import Nominatim

#-----
#model
#-----

print('get the model ready...')

label_dict = {'bye': 0, 'cloud': 1, 'general_weather': 2, 'greeting': 3, 'rain': 4, 'sun_hours': 5, 'temperature': 6, 'unknown': 7, 'wind': 8}



label_dict_inverse = {v: k for k, v in label_dict.items()}

bert = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
dir_path = os.path.dirname(os.path.realpath(__file__))

#print(dir_path)
device_location = 'cuda' if torch.cuda.is_available() else 'cpu'

bert.load_state_dict(torch.load(f'{dir_path}/BERT.model', map_location=torch.device(device_location)))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)
device = torch.device(device_location)

#-----------
#entity recognition
#-----------

nlp = spacy.load("en_core_web_lg")

#---------
#functions
#---------


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


def entity_recognition(string):
    doc = nlp(string)
    location = ''
    time = ''
    for ent in doc.ents:
        label = ent.label_
        content = ent.text
        if label == 'GPE':
            location = content
        elif label == 'DATE':
            time = content
    if location == '':
        location = 'Mannheim'

    return location, time


def get_weather(city_name):
  api_key = "1977d0c01a9cead345257efcc246e9ae"
  part = 'minutely'

  address = city_name
  geolocator = Nominatim(user_agent="Your_Name")
  location = geolocator.geocode(address)
  lat = location.latitude
  lon = location.longitude

  response = requests.get(f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude={part}&appid={api_key}')
  data = json.loads(response.text)
  return data


def get_general_weather(location, time):
  if location == '':
    city_name = 'Mannheim'
  else:
    city_name = location

  weather = get_weather(city_name)
  try:
    alerts = weather ['alerts']
  except KeyError:
    alerts = ''

  if time == '' or time == 'today':
    data = weather['daily'][0]

  elif time == 'tomorrow':
    data = weather['daily'][1]

  weather_data = {'temp_morn' : int(float(data['feels_like']['morn']) - 273.15),
                  'temp_eve' : int(float(data['feels_like']['eve']) - 273.15),
                  'temp_night' : int(float(data['feels_like']['night']) - 273.15),
                  'temp_day' : int(float(data['temp']['day']) - 273.15),
                  'temp_min' : int(float(data['temp']['min']) - 273.15),
                  'temp_max' : int(float(data['temp']['max']) - 273.15),
                  'temp_now': int(float(weather['current']['feels_like']) - 273.15),
                  'rain_prob' : int(float(data['pop']) * 100),
                  'clouds' : data['clouds'],
                  'wind' : data['wind_speed']}

  return weather_data, alerts


def get_rain(location, time):
    if location == '':
        city_name = 'Mannheim'
    else:
        city_name = location

    weather = get_weather(city_name)

    if time == '' or time == 'current':
        data = weather['current']
        if "rain" in data:
            rain = 'yes'
            rain_prob = 100
        else:
            rain = 'no'
            rain_prob = 0

    elif time == 'today':
        data = weather['hourly']

        prob_list = []
        hours = 7
        for hour in range(hours):
            prob = data[hour]['pop']
            prob_list.append(int(prob * 100))

        if (100 in prob_list):
            rain = 'yes'
            rain_prob = 100
        else:
            mean_prob = sum(prob_list) / len(prob_list)

            if mean_prob < 33:
                rain = 'no'
                rain_prob = mean_prob
            elif mean_prob >= 33 and mean_prob < 66:
                rain = 'maybe'
                rain_prob = mean_prob
            elif mean_prob >= 66:
                rain = 'yes'
                rain_prob = mean_prob

    elif time == 'tomorrow':
        prob = weather['daily'][1]['pop']
        if prob < 33:
            rain = 'no'
            rain_prob = prob
        elif prob >= 33 and prob < 66:
            rain = 'maybe'
            rain_prob = prob
        elif prob >= 66:
            rain = 'yes'
            rain_prob = prob

    return rain, rain_prob


def get_temperature(location, time):
    if location == '':
        city_name = 'Mannheim'

    else:
        city_name = location

    weather = get_weather(city_name)

    if time == '':
        temp = int(weather['current']['temp'] - 273.15)

    elif time == 'today':
        temp = weather['daily'][0]['feels_like']

    elif time == 'tomorrow':
        temp = weather['daily'][1]['feels_like']

    return temp

def get_sun_hours(location, time):
    if location == '':
        city_name = 'Mannheim'
    else:
        city_name = location

    weather = get_weather(city_name)
    if time == '' or time == 'current':
        sunrise = pd.to_datetime(int(weather['current']['sunrise']), unit='s')
        sunset = pd.to_datetime(int(weather['current']['sunset']), unit='s')
    elif time == 'today':
        sunrise = pd.to_datetime(int(weather['daily'][0]['sunrise']), unit='s')
        sunset = pd.to_datetime(int(weather['daily'][0]['sunset']), unit='s')
    elif time == 'tomorrow':
        sunrise = pd.to_datetime(int(weather['daily'][1]['sunrise']), unit='s')
        sunset = pd.to_datetime(int(weather['daily'][1]['sunset']), unit='s')
    return sunrise, sunset, city_name
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

# intent: Bye
        if intent == 0:
            responses = ['Bye, have a nice day.',
                         'Bye bye',
                         'Farewell traveler.']
            response = random.choice(responses)

# intent cloud
        elif intent == 1:
            location, time = entity_recognition(content)

            data, alerts = get_general_weather(location, time)
            if data['clouds'] < 33:
                responses = [f'It is sunny',
                             f'Take you sunglasses out. It is blue sky',
                             f'The sky is blue.'
                             ]
            elif data['clouds'] >= 33 and data['clouds'] < 66:
                responses = [f'A little bit from both sides.',
                             f'Partly sunny, partly cloudy.',
                             f'Not really sunny but not really cloudy either.'
                             ]
            else:
                responses = [f'No sun today.',
                             f'Clouds, clouds and more clouds.',
                             f'It is super cloudy.'
                             ]

            response = random.choice(responses)

# intent  general_weather
        elif intent == 2:
            location, time = entity_recognition(content)
            data, alerts = get_general_weather(location, time)
            #response beginning
            #temperature
            string = f"{time} in {location} the temperature ranges between {data['temp_min']}°C and {data['temp_max']}°C."

            if time == 'today':
                string += f"Right now the temperature is {data['temp_now']}C°."

            # cloudyness
            string += f"\n{time} "
            if data['clouds'] < 33:
                string += f"it is sunny"
            elif data['clouds'] >= 33 and data['clouds'] < 66:
                string += f"it is partly sunny and cloudy"
            else:
                string += f"it is very cloudy"

            # rain
            string += f"\n Also the probability for rain is {data['rain_prob']}%, so "
            if data['rain_prob'] < 33:
                string += f"it will not rain {time}"
            elif data['rain_prob'] >= 33 and data['rain_prob'] < 66:
                string += f"it might rain {time}"
            else:
                string += f"it will rain {time}"
            # wind
            string += f"\nThe wind is {data['wind']} km/h fast."

            response = string

# intent greeting
        elif intent == 3:
            responses = ['Hi there',
                         'Hello',
                         'Good day, sir.',
                         'Hello, what can I do for you?',
                         'Hey',
                         'Salut',
                         'Hello, my friend']
            response = random.choice(responses)

# intent rain
        elif intent == 4:
            location, time = entity_recognition(content)
            rain, rain_prob = get_rain(location, time)

            if rain == 'yes':
                responses = [f'Yes, if you want to go outside {time}, you should take an umbrella.',
                             f'Yes',
                             f'Yes, the probability is {rain_prob}%'
                             ]


            elif rain == 'no':
                responses = [f'No, it is not very likely. The probability for {time} is only {rain_prob}%',
                             f'No',
                             f'No, the probability is {rain_prob}%'
                             ]

            elif rain == 'maybe':
                responses = [f'Maybe yes, maybe no. Could be both.',
                             f'Maybe',
                             f'It might rain {time}, the probability is {rain_prob}%'
                            ]

            response = random.choice(responses)

# intent sun_hours
        elif intent == 5:
            location, time = entity_recognition(content)
            sunrise, sunset, city_name = get_sun_hours(location, time)
            #print(sunrise)
            #print(sunset)
            sunrise = str(sunrise)[11:]
            sunset = str(sunset)[11:]
            response = f'Sunrise: {sunrise} & Sunset: {sunset}'

# intent temperature
        elif intent == 6:
            location, time = entity_recognition(content)
            temp = get_temperature(location, time)

            if time == '':
                responses = [f'Right now the temperature is {temp}C°',
                             f'In {location} the temperature is {temp}C°',
                             f'Outside it is {temp}C°']

            elif time == 'today':
                avg_temperature = int(((temp['morn'] - 273.15) + (temp['day'] - 273.15) + (temp['eve'] - 273.15) +
                                       (temp['night'] - 273.15))/4)
                temp_day =  int(temp['day'] - 273.15)
                temp_morn =  int(temp['morn'] - 273.15)
                temp_eve =  int(temp['eve'] - 273.15)
                temp_night =  int(temp['night'] - 273.15)
                responses = [f"Today the average temperature is {avg_temperature}C°",
                             f"In the morning it is {temp_morn}C° \nDuring the day it is {temp_day}C° \n In the evening it is {temp_eve}C° \n In the night it is {temp_night}C°",
                             f"Today the temperature will rise up to {temp_day}°C"]

            elif time == 'tomorrow':
                avg_temperature = int(((temp['morn'] - 273.15) + (temp['day'] - 273.15) + (temp['eve'] - 273.15) +
                                       (temp['night'] - 273.15)) / 4)
                temp_day = int(temp['day'] - 273.15)
                temp_morn = int(temp['morn'] - 273.15)
                temp_eve = int(temp['eve'] - 273.15)
                temp_night = int(temp['night'] - 273.15)
                responses = [f"Tomorrow the average temperature is {avg_temperature}C°",
                             f"Tomorrow in the morning it will be {temp_morn}C° \nDuring the day it will be {temp_day}C° \n In the evening it will be {temp_eve}C° \n In the night it will be {temp_night}C°",
                             f"Tomorrow the temperature will rise up to {temp_day}°C"]

            response = random.choice(responses)

# intent unknown
        elif intent == 7:
            responses = [f"{label_dict_inverse.get(intent)}, sorry can't help you!",
                         f"Sorry, I don't know what you mean, please try again",
                         f"BEEP BOOP, I am a robot and can tell you only about the weather",
                         f"Help = No, Try again soon.",
                         f"I don't really feel like answering you right now",
                         f"Me and Google know everything, for this question you should consult Google."]
            response = random.choice(responses)

# intent windy
        elif intent == 8:
            location, time = entity_recognition(content)
            data, alerts = get_general_weather(location, time)
            wind = data['wind']
            print(wind)
            if wind <= 3:
                responses = [f'There is no wind. The wind speed is {wind}km/h',
                             f'No wind.',
                             f'There is no wind.']
            elif wind <= 10:
                responses = [f'It is a soft breeze. The wind speed is {wind}km/h',
                             f'There is a soft breeze.',
                             f'A nice chilly breeze outside']

            elif wind > 10:
                responses = [f'It is windy. The wind speed is {wind}km/h.',
                             f'It is a little windy',
                             f'Maybe a hurrican.']

            response = random.choice(responses)

        await channel.send(response)




client.run(discord_token)