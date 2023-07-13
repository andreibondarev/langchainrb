require "langchain"
require "dotenv/load"

functions = [
  {
    name: "create_rails_controller",
    description: "gives a command to create a rails controller",
    parameters: {
      type: :object,
      properties: {
        controller_name: {
          type: :string,
          description: "the controller name, e.g. users_controller"
        },
        unit: {
          type: "string",
          enum: %w[celsius fahrenheit]
        }
      },
      required: ["controller_name"]
    }
  }
]

openai = Langchain::LLM::OpenAI.new(
  api_key: ENV["OPENAI_API_KEY"],
  default_options: {
    chat_completion_model_name: "gpt-3.5-turbo-16k"
  }
)

client = Langchain::Vectorsearch::Qdrant.new(
  url: ENV["QDRANT_URL"],
  api_key: ENV["QDRANT_API_KEY"],
  index_name: ENV["QDRANT_INDEX"],
  llm: openai
)

client.llm.functions = functions
client.llm.complete_response = true
chat = client.ask(question: "create a users_controller")

puts chat
