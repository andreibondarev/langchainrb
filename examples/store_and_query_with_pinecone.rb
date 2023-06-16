require "langchain"

# gem install pinecone
# or add `gem "pinecone"` to your Gemfile

# Instantiate the Pinecone client
pinecone = Langchain::Vectorsearch::Pinecone.new(
  environment: ENV["PINECONE_ENVIRONMENT"],
  api_key: ENV["PINECONE_API_KEY"],
  index_name: "recipes",
  llm: Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
)

# Create the default schema.
# If you are using the free Pinecone tier, ensure there is not an existing schema/index
pinecone.create_default_schema

# Set up an array of text strings
recipes = [
  "Preheat oven to 400 degrees F (200 degrees C). Cut the top off the head of garlic. Arrange the garlic, carrots, celery, onion, pepper, and tomato on a large baking sheet in a single layer. Drizzle the olive oil over the vegetables; season with salt and pepper. Roast the vegetables in the preheated oven, turning every 20 minutes, until tender and browned, about 1 hour. Combine the water, thyme, parsley, and bay leaves in a large stock pot over medium-high heat. Squeeze the head of garlic into the stock pot, and discard the outer husk. Place the carrots, celery, onion, pepper, and tomato in the stock pot. Bring the water to a boil; reduce heat to low and simmer for 1 1/2 hours; strain and cool.",
  "Heat oven to 190C/fan 170C/gas 5. Heat 1 tbsp oil and the butter in a frying pan, then add the onion and fry for 5 mins until softened. Cool slightly. Tip the sausagemeat, lemon zest, breadcrumbs, apricots, chestnuts and thyme into a bowl. Add the onion and cranberries, and mix everything together with your hands, adding plenty of pepper and a little salt. Cut each chicken breast into three fillets lengthwise and season all over with salt and pepper. Heat the remaining oil in the frying pan, and fry the chicken fillets quickly until browned, about 6-8 mins. Roll out two-thirds of the pastry to line a 20-23cm springform or deep loose-based tart tin. Press in half the sausage mix and spread to level. Then add the chicken pieces in one layer and cover with the rest of the sausage. Press down lightly. Roll out the remaining pastry. Brush the edges of the pastry with beaten egg and cover with the pastry lid. Pinch the edges to seal, then trim. Brush the top of the pie with egg, then roll out the trimmings to make holly leaf shapes and berries. Decorate the pie and brush again with egg. Set the tin on a baking sheet and bake for 50-60 mins, then cool in the tin for 15 mins. Remove and leave to cool completely. Serve with a winter salad and pickles."
]

# Add data to the index. Pinecone will use OpenAI to generate embeddings behind the scene.
pinecone.add_texts(
  texts: recipes
)

# Query your data
pinecone.similarity_search(
  query: "chicken",
  k: 1
)

# Interact with your index through Q&A
pinecone.ask(
  question: "What is a good recipe for chicken?"
)

# Generate an embedding and search by it
openai = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
embedding = openai.embed(text: "veggie")

pinecone.similarity_search_by_vector(
  embedding: embedding
)
