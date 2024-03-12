import os
import openai
import getpass

# pricing: https://openai.com/pricing

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

#PROMPT = "An eco-friendly computer from the 90s in the style of vaporwave"
#PROMPT = "steelmaking plant with a small fire that doesn't look frightening"
PROMPT = "create a realistic looking baby with age of 3 months that looks like us"

client = openai.OpenAI()

response = client.images.generate(
    prompt=PROMPT,
    model="dall-e-2",
    n=1,
    size="512x512", # 256x256, 512x512, 1024x1024
)

print(response.data[0].url)