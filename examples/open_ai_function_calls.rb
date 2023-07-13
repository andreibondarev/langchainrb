require "langchain"
require "dotenv/load"

functions = [
  {
    name: "get_current_weather",
    description: "Get the current weather in a given location",
    parameters: {
      type: :object,
      properties: {
        location: {
          type: :string,
          description: "The city and state, e.g. San Francisco, CA"
        },
        unit: {
          type: "string",
          enum: %w[celsius fahrenheit]
        }
      },
      required: ["location"]
    }
  }
]

openai = Langchain::LLM::OpenAI.new(
  api_key: ENV["OPENAI_API_KEY"],
  default_options: {
    chat_completion_model_name: "gpt-3.5-turbo-16k"
  }
)

chat = Langchain::Conversation.new(llm: openai)

chat.set_context("You are the climate bot")
chat.set_functions(functions)

DONE = %w[done end eof exit].freeze

user_message = "what's the weather in NYC?"

puts chat.message(user_message)
