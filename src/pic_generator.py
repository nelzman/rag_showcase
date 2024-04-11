import os
import openai
import getpass

# pricing: https://openai.com/pricing

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

#PROMPT = "An eco-friendly computer from the 90s in the style of vaporwave"
#PROMPT = "steelmaking plant with a small fire that doesn't look frightening"
PROMPT = "create a male programmer from mid-europe who does python code reviews in a cyberpunk style"

client = openai.OpenAI()

response = client.images.generate(
    prompt=PROMPT,
    model="dall-e-3",
    n=1,
    size="1024x1024", # 256x256, 512x512, 1024x1024
)

print(response.data[0].url)