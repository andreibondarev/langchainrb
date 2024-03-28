💎🔗 Langchain.rb
---
⚡ Building LLM-powered applications in Ruby ⚡

For deep Rails integration see: [langchainrb_rails](https://github.com/andreibondarev/langchainrb_rails) gem.

Available for paid consulting engagements! [Email me](mailto:andrei@sourcelabs.io).

![Tests status](https://github.com/andreibondarev/langchainrb/actions/workflows/ci.yml/badge.svg?branch=main)
[![Gem Version](https://badge.fury.io/rb/langchainrb.svg)](https://badge.fury.io/rb/langchainrb)
[![Docs](http://img.shields.io/badge/yard-docs-blue.svg)](http://rubydoc.info/gems/langchainrb)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/andreibondarev/langchainrb/blob/main/LICENSE.txt)
[![](https://dcbadge.vercel.app/api/server/WDARp7J2n8?compact=true&style=flat)](https://discord.gg/WDARp7J2n8)
[![X](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40rushing_andrei)](https://twitter.com/rushing_andrei)

## Use Cases
* Retrieval Augmented Generation (RAG) and vector search
* [Assistants](#assistants) (chat bots) & [AI Agents](https://github.com/andreibondarev/langchainrb/tree/main/lib/langchain/agent/agents.md)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Large Language Models (LLMs)](#large-language-models-llms)
- [Prompt Management](#prompt-management)
- [Output Parsers](#output-parsers)
- [Building RAG](#building-retrieval-augment-generation-rag-system)
- [Assistants](#assistants)
- [Evaluations](#evaluations-evals)
- [Examples](#examples)
- [Logging](#logging)
- [Problems](#problems)
- [Development](#development)
- [Discord](#discord)

## Installation

Install the gem and add to the application's Gemfile by executing:

    bundle add langchainrb

If bundler is not being used to manage dependencies, install the gem by executing:

    gem install langchainrb

Additional gems may be required. They're not included by default so you can include only what you need.

## Usage

```ruby
require "langchain"
```

## Large Language Models (LLMs)
Langchain.rb wraps supported LLMs in a unified interface allowing you to easily swap out and test out different models.

#### Supported LLMs and features:
| LLM providers                                                                                   | `embed()`            | `complete()`         | `chat()`              | `summarize()`        | Notes              |
| --------                                                                                        |:------------------:| :-------:          | :-----------------: | :-------:          | :----------------- |
| [OpenAI](https://openai.com/?utm_source=langchainrb&utm_medium=github)                          | ✅                 | ✅                 | ✅                  | ❌                 | Including Azure OpenAI |
| [AI21](https://ai21.com/?utm_source=langchainrb&utm_medium=github)                              | ❌                 | ✅                 | ❌                  | ✅                 |                    |
| [Anthropic](https://anthropic.com/?utm_source=langchainrb&utm_medium=github)                        | ❌                 | ✅                 | ❌                  | ❌                 |                    |
| [AWS Bedrock](https://aws.amazon.com/bedrock?utm_source=langchainrb&utm_medium=github)          | ✅                 | ✅                 | ❌                  | ❌                 | Provides AWS, Cohere, AI21, Antropic and Stability AI models |
| [Cohere](https://cohere.com/?utm_source=langchainrb&utm_medium=github)                     | ✅                 | ✅                 | ✅                  | ✅                 |                    |
| [GooglePalm](https://ai.google/discover/palm2?utm_source=langchainrb&utm_medium=github)         | ✅                 | ✅                 | ✅                  | ✅                 |                    |
| [Google Vertex AI](https://cloud.google.com/vertex-ai?utm_source=langchainrb&utm_medium=github) | ✅                 | ✅                 | ❌                  | ✅                 |                    |
| [HuggingFace](https://huggingface.co/?utm_source=langchainrb&utm_medium=github)                 | ✅                 | ❌                 | ❌                  | ❌                 |                    |
| [Ollama](https://ollama.ai/?utm_source=langchainrb&utm_medium=github)                           | ✅                 | ✅                 | ✅                  | ✅                 |                    |
| [Replicate](https://replicate.com/?utm_source=langchainrb&utm_medium=github)                    | ✅                 | ✅                 | ✅                  | ✅                 |                    |

#### Using standalone LLMs:

#### OpenAI

Add `gem "ruby-openai", "~> 6.3.0"` to your Gemfile.

```ruby
llm = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
```
You can pass additional parameters to the constructor, it will be passed to the OpenAI client:
```ruby
llm = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"], llm_options: { ... })
```

Generate vector embeddings:
```ruby
llm.embed(text: "foo bar").embedding
```

Generate a chat completion:
```ruby
llm.chat(messages: [{role: "user", content: "What is the meaning of life?"}]).completion
```

Summarize the text:
```ruby
llm.summarize(text: "...").completion
```

You can use any other LLM by invoking the same interface:
```ruby
llm = Langchain::LLM::GooglePalm.new(api_key: ENV["GOOGLE_PALM_API_KEY"], default_options: { ... })
```

### Prompt Management

#### Prompt Templates

Create a prompt with input variables:

```ruby
prompt = Langchain::Prompt::PromptTemplate.new(template: "Tell me a {adjective} joke about {content}.", input_variables: ["adjective", "content"])
prompt.format(adjective: "funny", content: "chickens") # "Tell me a funny joke about chickens."
```

Creating a PromptTemplate using just a prompt and no input_variables:

```ruby
prompt = Langchain::Prompt::PromptTemplate.from_template("Tell me a funny joke about chickens.")
prompt.input_variables # []
prompt.format # "Tell me a funny joke about chickens."
```

Save prompt template to JSON file:

```ruby
prompt.save(file_path: "spec/fixtures/prompt/prompt_template.json")
```

Loading a new prompt template using a JSON file:

```ruby
prompt = Langchain::Prompt.load_from_path(file_path: "spec/fixtures/prompt/prompt_template.json")
prompt.input_variables # ["adjective", "content"]
```

#### Few Shot Prompt Templates

Create a prompt with a few shot examples:

```ruby
prompt = Langchain::Prompt::FewShotPromptTemplate.new(
  prefix: "Write antonyms for the following words.",
  suffix: "Input: {adjective}\nOutput:",
  example_prompt: Langchain::Prompt::PromptTemplate.new(
    input_variables: ["input", "output"],
    template: "Input: {input}\nOutput: {output}"
  ),
  examples: [
    { "input": "happy", "output": "sad" },
    { "input": "tall", "output": "short" }
  ],
   input_variables: ["adjective"]
)

prompt.format(adjective: "good")

# Write antonyms for the following words.
#
# Input: happy
# Output: sad
#
# Input: tall
# Output: short
#
# Input: good
# Output:
```

Save prompt template to JSON file:

```ruby
prompt.save(file_path: "spec/fixtures/prompt/few_shot_prompt_template.json")
```

Loading a new prompt template using a JSON file:

```ruby
prompt = Langchain::Prompt.load_from_path(file_path: "spec/fixtures/prompt/few_shot_prompt_template.json")
prompt.prefix # "Write antonyms for the following words."
```

Loading a new prompt template using a YAML file:

```ruby
prompt = Langchain::Prompt.load_from_path(file_path: "spec/fixtures/prompt/prompt_template.yaml")
prompt.input_variables #=> ["adjective", "content"]
```


### Output Parsers

Parse LLM text responses into structured output, such as JSON.

#### Structured Output Parser

You can use the `StructuredOutputParser` to generate a prompt that instructs the LLM to provide a JSON response adhering to a specific JSON schema:

```ruby
json_schema = {
  type: "object",
  properties: {
    name: {
      type: "string",
      description: "Persons name"
    },
    age: {
      type: "number",
      description: "Persons age"
    },
    interests: {
      type: "array",
      items: {
        type: "object",
        properties: {
          interest: {
            type: "string",
            description: "A topic of interest"
          },
          levelOfInterest: {
            type: "number",
            description: "A value between 0 and 100 of how interested the person is in this interest"
          }
        },
        required: ["interest", "levelOfInterest"],
        additionalProperties: false
      },
      minItems: 1,
      maxItems: 3,
      description: "A list of the person's interests"
    }
  },
  required: ["name", "age", "interests"],
  additionalProperties: false
}
parser = Langchain::OutputParsers::StructuredOutputParser.from_json_schema(json_schema)
prompt = Langchain::Prompt::PromptTemplate.new(template: "Generate details of a fictional character.\n{format_instructions}\nCharacter description: {description}", input_variables: ["description", "format_instructions"])
prompt_text = prompt.format(description: "Korean chemistry student", format_instructions: parser.get_format_instructions)
# Generate details of a fictional character.
# You must format your output as a JSON value that adheres to a given "JSON Schema" instance.
# ...
```

Then parse the llm response:

```ruby
llm = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
llm_response = llm.chat(messages: [{role: "user", content: prompt_text}]).completion
parser.parse(llm_response)
# {
#   "name" => "Kim Ji-hyun",
#   "age" => 22,
#   "interests" => [
#     {
#       "interest" => "Organic Chemistry",
#       "levelOfInterest" => 85
#     },
#     ...
#   ]
# }
```

If the parser fails to parse the LLM response, you can use the `OutputFixingParser`. It sends an error message, prior output, and the original prompt text to the LLM, asking for a "fixed" response:

```ruby
begin
  parser.parse(llm_response)
rescue Langchain::OutputParsers::OutputParserException => e
  fix_parser = Langchain::OutputParsers::OutputFixingParser.from_llm(
    llm: llm,
    parser: parser
  )
  fix_parser.parse(llm_response)
end
```

Alternatively, if you don't need to handle the `OutputParserException`, you can simplify the code:

```ruby
# we already have the `OutputFixingParser`:
# parser = Langchain::OutputParsers::StructuredOutputParser.from_json_schema(json_schema)
fix_parser = Langchain::OutputParsers::OutputFixingParser.from_llm(
  llm: llm,
  parser: parser
)
fix_parser.parse(llm_response)
```

See [here](https://github.com/andreibondarev/langchainrb/tree/main/examples/create_and_manage_prompt_templates_using_structured_output_parser.rb) for a concrete example

## Building Retrieval Augment Generation (RAG) system
RAG is a methodology that assists LLMs generate accurate and up-to-date information.
A typical RAG workflow follows the 3 steps below:
1. Relevant knowledge (or data) is retrieved from the knowledge base (typically a vector search DB)
2. A prompt, containing retrieved knowledge above, is constructed.
3. LLM receives the prompt above to generate a text completion.
Most common use-case for a RAG system is powering Q&A systems where users pose natural language questions and receive answers in natural language.

### Vector search databases
Langchain.rb provides a convenient unified interface on top of supported vectorsearch databases that make it easy to configure your index, add data, query and retrieve from it.

#### Supported vector search databases and features:

| Database                                                                                   | Open-source        | Cloud offering     |
| --------                                                                                   |:------------------:| :------------:     |
| [Chroma](https://trychroma.com/?utm_source=langchainrb&utm_medium=github)                  | ✅                 | ✅                 |
| [Epsilla](https://epsilla.com/?utm_source=langchainrb&utm_medium=github)                   | ✅                 | ✅                 |
| [Hnswlib](https://github.com/nmslib/hnswlib/?utm_source=langchainrb&utm_medium=github)     | ✅                 | ❌                 |
| [Milvus](https://milvus.io/?utm_source=langchainrb&utm_medium=github)                      | ✅                 | ✅ Zilliz Cloud    |
| [Pinecone](https://www.pinecone.io/?utm_source=langchainrb&utm_medium=github)              | ❌                 | ✅                 |
| [Pgvector](https://github.com/pgvector/pgvector/?utm_source=langchainrb&utm_medium=github) | ✅                 | ✅                 |
| [Qdrant](https://qdrant.tech/?utm_source=langchainrb&utm_medium=github)                    | ✅                 | ✅                 |
| [Weaviate](https://weaviate.io/?utm_source=langchainrb&utm_medium=github)                  | ✅                 | ✅                 |
| [Elasticsearch](https://www.elastic.co/?utm_source=langchainrb&utm_medium=github)          | ✅                 | ✅                 |

### Using Vector Search Databases 🔍

Pick the vector search database you'll be using, add the gem dependency and instantiate the client:
```ruby
gem "weaviate-ruby", "~> 0.8.9"
```

Choose and instantiate the LLM provider you'll be using to generate embeddings
```ruby
llm = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
```

```ruby
client = Langchain::Vectorsearch::Weaviate.new(
    url: ENV["WEAVIATE_URL"],
    api_key: ENV["WEAVIATE_API_KEY"],
    index_name: "Documents",
    llm: llm
)
```

You can instantiate any other supported vector search database:
```ruby
client = Langchain::Vectorsearch::Chroma.new(...)   # `gem "chroma-db", "~> 0.6.0"`
client = Langchain::Vectorsearch::Epsilla.new(...)  # `gem "epsilla-ruby", "~> 0.0.3"`
client = Langchain::Vectorsearch::Hnswlib.new(...)  # `gem "hnswlib", "~> 0.8.1"`
client = Langchain::Vectorsearch::Milvus.new(...)   # `gem "milvus", "~> 0.9.2"`
client = Langchain::Vectorsearch::Pinecone.new(...) # `gem "pinecone", "~> 0.1.6"`
client = Langchain::Vectorsearch::Pgvector.new(...) # `gem "pgvector", "~> 0.2"`
client = Langchain::Vectorsearch::Qdrant.new(...)   # `gem "qdrant-ruby", "~> 0.9.3"`
client = Langchain::Vectorsearch::Elasticsearch.new(...)   # `gem "elasticsearch", "~> 8.2.0"`
```

Create the default schema:
```ruby
client.create_default_schema
```

Add plain text data to your vector search database:
```ruby
client.add_texts(
  texts: [
    "Begin by preheating your oven to 375°F (190°C). Prepare four boneless, skinless chicken breasts by cutting a pocket into the side of each breast, being careful not to cut all the way through. Season the chicken with salt and pepper to taste. In a large skillet, melt 2 tablespoons of unsalted butter over medium heat. Add 1 small diced onion and 2 minced garlic cloves, and cook until softened, about 3-4 minutes. Add 8 ounces of fresh spinach and cook until wilted, about 3 minutes. Remove the skillet from heat and let the mixture cool slightly.",
      "In a bowl, combine the spinach mixture with 4 ounces of softened cream cheese, 1/4 cup of grated Parmesan cheese, 1/4 cup of shredded mozzarella cheese, and 1/4 teaspoon of red pepper flakes. Mix until well combined. Stuff each chicken breast pocket with an equal amount of the spinach mixture. Seal the pocket with a toothpick if necessary. In the same skillet, heat 1 tablespoon of olive oil over medium-high heat. Add the stuffed chicken breasts and sear on each side for 3-4 minutes, or until golden brown."
  ]
)
```

Or use the file parsers to load, parse and index data into your database:
```ruby
my_pdf = Langchain.root.join("path/to/my.pdf")
my_text = Langchain.root.join("path/to/my.txt")
my_docx = Langchain.root.join("path/to/my.docx")

client.add_data(paths: [my_pdf, my_text, my_docx])
```
Supported file formats: docx, html, pdf, text, json, jsonl, csv, xlsx, eml.

Retrieve similar documents based on the query string passed in:
```ruby
client.similarity_search(
  query:,
  k:       # number of results to be retrieved
)
```

Retrieve similar documents based on the query string passed in via the [HyDE technique](https://arxiv.org/abs/2212.10496):
```ruby
client.similarity_search_with_hyde()
```

Retrieve similar documents based on the embedding passed in:
```ruby
client.similarity_search_by_vector(
  embedding:,
  k:       # number of results to be retrieved
)
```

RAG-based querying
```ruby
client.ask(question: "...")
```

## Assistants
Assistants are Agent-like objects that leverage helpful instructions, LLMs, tools and knowledge to respond to user queries. Assistants can be configured with an LLM of your choice (currently only OpenAI), any vector search database and easily extended with additional tools.

### Creating an Assistant
1. Instantiate an LLM of your choice
```ruby
llm = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
```
2. Instantiate a Thread. Threads keep track of the messages in the Assistant conversation.
```ruby
thread = Langchain::Thread.new
```
You can pass old message from previously using the Assistant:
```ruby
thread.messages = messages
```
Messages contain the conversation history and the whole message history is sent to the LLM every time. A Message belongs to 1 of the 4 roles:
* `Message(role: "system")` message usually contains the instructions.
* `Message(role: "user")` messages come from the user.
* `Message(role: "assistant")` messages are produced by the LLM.
* `Message(role: "tool")` messages are sent in response to tool calls with tool outputs.

3. Instantiate an Assistant
```ruby
assistant = Langchain::Assistant.new(
  llm: llm,
  thread: thread,
  instructions: "You are a Meteorologist Assistant that is able to pull the weather for any location",
  tools: [
    Langchain::Tool::Weather.new(api_key: ENV["OPEN_WEATHER_API_KEY"])
  ]
)
```
### Using an Assistant
You can now add your message to an Assistant.
```ruby
assistant.add_message content: "What's the weather in New York, New York?"
```

Run the Assistant to generate a response. 
```ruby
assistant.run
```

If a Tool is invoked you can manually submit an output.
```ruby
assistant.submit_tool_output tool_call_id: "...", output: "It's 70 degrees and sunny in New York City"
```

Or run the assistant with `auto_tool_execution: tool` to call Tools automatically.
```ruby
assistant.add_message content: "How about San Diego, CA?"
assistant.run(auto_tool_execution: true)
```
You can also combine the two by calling:
```ruby
assistant.add_message_and_run content: "What about Sacramento, CA?", auto_tool_execution: true
```

### Accessing Thread messages
You can access the messages in a Thread by calling `assistant.thread.messages`.
```ruby
assistant.thread.messages
```

The Assistant checks the context window limits before every request to the LLM and remove oldest thread messages one by one if the context window is exceeded.

## Evaluations (Evals)
The Evaluations module is a collection of tools that can be used to evaluate and track the performance of the output products by LLM and your RAG (Retrieval Augmented Generation) pipelines.

### RAGAS
Ragas helps you evaluate your Retrieval Augmented Generation (RAG) pipelines. The implementation is based on this [paper](https://arxiv.org/abs/2309.15217) and the original Python [repo](https://github.com/explodinggradients/ragas). Ragas tracks the following 3 metrics and assigns the 0.0 - 1.0 scores:
* Faithfulness - the answer is grounded in the given context.
* Context Relevance - the retrieved context is focused, containing little to no irrelevant information.
* Answer Relevance - the generated answer addresses the actual question that was provided.

```ruby
# We recommend using Langchain::LLM::OpenAI as your llm for Ragas
ragas = Langchain::Evals::Ragas::Main.new(llm: llm)

# The answer that the LLM generated
# The question (or the original prompt) that was asked
# The context that was retrieved (usually from a vectorsearch database)
ragas.score(answer: "", question: "", context: "")
# =>
# {
#   ragas_score: 0.6601257446503674,
#   answer_relevance_score: 0.9573145866787608,
#   context_relevance_score: 0.6666666666666666,
#   faithfulness_score: 0.5
# }
```

## Examples
Additional examples available: [/examples](https://github.com/andreibondarev/langchainrb/tree/main/examples)

## Logging

Langchain.rb uses standard logging mechanisms and defaults to `:warn` level. Most messages are at info level, but we will add debug or warn statements as needed.
To show all log messages:

```ruby
Langchain.logger.level = :debug
```

## Problems
If you're having issues installing `unicode` gem required by `pragmatic_segmenter`, try running:
```bash
gem install unicode -- --with-cflags="-Wno-incompatible-function-pointer-types"
```

## Development

1. `git clone https://github.com/andreibondarev/langchainrb.git`
2. `cp .env.example .env`, then fill out the environment variables in `.env`
3. `bundle exec rake` to ensure that the tests pass and to run standardrb
4. `bin/console` to load the gem in a REPL session. Feel free to add your own instances of LLMs, Tools, Agents, etc. and experiment with them.
5. Optionally, install lefthook git hooks for pre-commit to auto lint: `gem install lefthook && lefthook install -f`

## Discord
Join us in the [Langchain.rb](https://discord.gg/WDARp7J2n8) Discord server.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=andreibondarev/langchainrb&type=Date)](https://star-history.com/#andreibondarev/langchainrb&Date)

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/andreibondarev/langchainrb.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
