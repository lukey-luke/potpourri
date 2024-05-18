from langchain.chains import Chain
from langchain.prompts import Prompt
from googs import google_search
import os

GOOGLE_API_KEY = os.environ.get("YOUR_GOOGLE_API_KEY")
CSE_ID = os.environ.get("YOUR_CSE_ID")

class GoogleSearchChain(Chain):
    def __init__(self, api_key, cse_id):
        self.api_key = api_key
        self.cse_id = cse_id

    def _call(self, inputs):
        query = inputs['query']
        search_results = google_search(query, self.api_key, self.cse_id)
        return search_results

class ImageGenerationChain(Chain):
    def _call(self, inputs):
        prompt = inputs['prompt']
        image = generate_image(prompt)
        return image

class LangChainApp(Chain):
    def __init__(self, google_search_chain, image_generation_chain):
        self.google_search_chain = google_search_chain
        self.image_generation_chain = image_generation_chain

    def _call(self, inputs):
        query = inputs['query']
        search_results = self.google_search_chain({'query': query})
        snippet = search_results[0]['snippet']  # Using the first search result snippet
        image = self.image_generation_chain({'prompt': snippet})
        return image

# Initialize chains
google_search_chain = GoogleSearchChain(api_key=GOOGLE_API_KEY, cse_id=CSE_ID)
image_generation_chain = ImageGenerationChain()

# Create the LangChain app
langchain_app = LangChainApp(google_search_chain, image_generation_chain)

# Use the LangChain app
query = "beautiful sunset over a mountain range"
image = langchain_app({'query': query})

# Save the generated image
image.save("generated_image.png")

