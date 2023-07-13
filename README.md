💎🔗 LangChain.rb
---
⚡ Building applications with LLMs through composability ⚡

👨‍💻👩‍💻 CURRENTLY SEEKING PEOPLE TO FORM THE CORE GROUP OF MAINTAINERS WITH

:warning: UNDER ACTIVE AND RAPID DEVELOPMENT (MAY BE BUGGY AND UNTESTED)

![Tests status](https://github.com/andreibondarev/langchainrb/actions/workflows/ci.yml/badge.svg)
[![Gem Version](https://badge.fury.io/rb/langchainrb.svg)](https://badge.fury.io/rb/langchainrb)
[![Docs](http://img.shields.io/badge/yard-docs-blue.svg)](http://rubydoc.info/gems/langchainrb)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/andreibondarev/langchainrb/blob/main/LICENSE.txt)
[![](https://dcbadge.vercel.app/api/server/WDARp7J2n8?compact=true&style=flat)](https://discord.gg/WDARp7J2n8)


Langchain.rb is a library that's an abstraction layer on top many emergent AI, ML and other DS tools. The goal is to abstract complexity and difficult concepts to make building AI/ML-supercharged applications approachable for traditional software engineers.

## Installation

Install the gem and add to the application's Gemfile by executing:

    $ bundle add langchainrb

If bundler is not being used to manage dependencies, install the gem by executing:

    $ gem install langchainrb

## Usage

```ruby
require "langchain"
```

#### Supported vector search databases and features:

| Database | Querying           | Storage | Schema Management | Backups | Rails Integration |
| -------- |:------------------:| -------:| -----------------:| -------:| -----------------:|
| [Chroma](https://trychroma.com/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | WIP     | :white_check_mark: |
| [Hnswlib](https://github.com/nmslib/hnswlib/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | WIP     | WIP               |
| [Milvus](https://milvus.io/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | WIP     | WIP               |
| [Pinecone](https://www.pinecone.io/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | WIP     | :white_check_mark: |
| [Pgvector](https://github.com/pgvector/pgvector) | :white_check_mark: | :white_check_mark: | :white_check_mark: | WIP     | :white_check_mark: |
| [Qdrant](https://qdrant.tech/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | WIP     | :white_check_mark: |
| [Weaviate](https://weaviate.io/) | :white_check_mark: | :white_check_mark: | :white_check_mark: | WIP     | :white_check_mark: |

### Using Vector Search Databases 🔍

Choose the LLM provider you'll be using (OpenAI or Cohere) and retrieve the API key.

Add `gem "weaviate-ruby", "~> 0.8.3"`  to your Gemfile.

Pick the vector search database you'll be using and instantiate the client:
```ruby
client = Langchain::Vectorsearch::Weaviate.new(
    url: ENV["WEAVIATE_URL"],
    api_key: ENV["WEAVIATE_API_KEY"],
    index_name: "",
    llm: Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
)

# You can instantiate any other supported vector search database:
client = Langchain::Vectorsearch::Chroma.new(...) # `gem "chroma-db", "~> 0.3.0"`
client = Langchain::Vectorsearch::Hnswlib.new(...) # `gem "hnswlib", "~> 0.8.1"`
client = Langchain::Vectorsearch::Milvus.new(...) # `gem "milvus", "~> 0.9.0"`
client = Langchain::Vectorsearch::Pinecone.new(...) # `gem "pinecone", "~> 0.1.6"`
client = Langchain::Vectorsearch::Pgvector.new(...) # `gem "pgvector", "~> 0.2"`
client = Langchain::Vectorsearch::Qdrant.new(...) # `gem"qdrant-ruby", "~> 0.9.0"`
```

```ruby
# Creating the default schema
client.create_default_schema
```

```ruby
# Store plain texts in your vector search database
client.add_texts(
    texts: [
        "Begin by preheating your oven to 375°F (190°C). Prepare four boneless, skinless chicken breasts by cutting a pocket into the side of each breast, being careful not to cut all the way through. Season the chicken with salt and pepper to taste. In a large skillet, melt 2 tablespoons of unsalted butter over medium heat. Add 1 small diced onion and 2 minced garlic cloves, and cook until softened, about 3-4 minutes. Add 8 ounces of fresh spinach and cook until wilted, about 3 minutes. Remove the skillet from heat and let the mixture cool slightly.",
        "In a bowl, combine the spinach mixture with 4 ounces of softened cream cheese, 1/4 cup of grated Parmesan cheese, 1/4 cup of shredded mozzarella cheese, and 1/4 teaspoon of red pepper flakes. Mix until well combined. Stuff each chicken breast pocket with an equal amount of the spinach mixture. Seal the pocket with a toothpick if necessary. In the same skillet, heat 1 tablespoon of olive oil over medium-high heat. Add the stuffed chicken breasts and sear on each side for 3-4 minutes, or until golden brown."
    ]
)
```
```ruby
# Store the contents of your files in your vector search database
my_pdf = Langchain.root.join("path/to/my.pdf")
my_text = Langchain.root.join("path/to/my.txt")
my_docx = Langchain.root.join("path/to/my.docx")

client.add_data(paths: [my_pdf, my_text, my_docx])
```
```ruby
# Retrieve similar documents based on the query string passed in
client.similarity_search(
    query:,
    k:       # number of results to be retrieved
)
```
```ruby
# Retrieve similar documents based on the embedding passed in
client.similarity_search_by_vector(
    embedding:,
    k:       # number of results to be retrieved
)
```
```ruby
# Q&A-style querying based on the question passed in
client.ask(
    question:
)
```

## Integrating Vector Search into ActiveRecord models
```ruby
class Product < ActiveRecord::Base
  vectorsearch provider: Langchain::Vectorsearch::Qdrant.new(
                 api_key: ENV["QDRANT_API_KEY"],
                 url: ENV["QDRANT_URL"],
                 index_name: "Products",
                 llm: Langchain::LLM::GooglePalm.new(api_key: ENV["GOOGLE_PALM_API_KEY"])
               )

  after_save :upsert_to_vectorsearch
end
```

Additional info [here](https://github.com/andreibondarev/langchainrb/blob/main/lib/langchain/active_record/hooks.rb#L10-L38).

### Using Standalone LLMs 🗣️

Add `gem "ruby-openai", "~> 4.0.0"` to your Gemfile.

#### OpenAI
```ruby
openai = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])
```
You can pass additional parameters to the constructor, it will be passed to the OpenAI client:
```ruby
openai = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"], llm_options: {uri_base: "http://localhost:1234"}) )
```
```ruby
openai.embed(text: "foo bar")
```
```ruby
openai.complete(prompt: "What is the meaning of life?")
```

#### Cohere
Add `gem "cohere-ruby", "~> 0.9.3"` to your Gemfile.

```ruby
cohere = Langchain::LLM::Cohere.new(api_key: ENV["COHERE_API_KEY"])
```
```ruby
cohere.embed(text: "foo bar")
```
```ruby
cohere.complete(prompt: "What is the meaning of life?")
```

#### HuggingFace
Add `gem "hugging-face", "~> 0.3.2"` to your Gemfile.
```ruby
hugging_face = Langchain::LLM::HuggingFace.new(api_key: ENV["HUGGING_FACE_API_KEY"])
```

#### Replicate
Add `gem "replicate-ruby", "~> 0.2.2"` to your Gemfile.
```ruby
replicate = Langchain::LLM::Replicate.new(api_key: ENV["REPLICATE_API_KEY"])
```

#### Google PaLM (Pathways Language Model)
Add `"google_palm_api", "~> 0.1.2"` to your Gemfile.
```ruby
google_palm = Langchain::LLM::GooglePalm.new(api_key: ENV["GOOGLE_PALM_API_KEY"])
```

#### AI21
Add `gem "ai21", "~> 0.2.1"` to your Gemfile.
```ruby
ai21 = Langchain::LLM::AI21.new(api_key: ENV["AI21_API_KEY"])
```

### Using Prompts 📋

#### Prompt Templates

Create a prompt with one input variable:

```ruby
prompt = Langchain::Prompt::PromptTemplate.new(template: "Tell me a {adjective} joke.", input_variables: ["adjective"])
prompt.format(adjective: "funny") # "Tell me a funny joke."
```

Create a prompt with multiple input variables:

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

### Using Output Parsers 

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
llm_response = llm.chat(prompt: prompt_text)
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

### Using Agents 🤖
Agents are semi-autonomous bots that can respond to user questions and use available to them Tools to provide informed replies. They break down problems into series of steps and define Actions (and Action Inputs) along the way that are executed and fed back to them as additional information. Once an Agent decides that it has the Final Answer it responds with it.

#### ReAct Agent

Add `gem "ruby-openai"`, `gem "eqn"`, and `gem "google_search_results"` to your Gemfile

```ruby
search_tool = Langchain::Tool::GoogleSearch.new(api_key: ENV["SERPAPI_API_KEY"])
calculator = Langchain::Tool::Calculator.new

openai = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])

agent = Langchain::Agent::ReActAgent.new(
  llm: openai,
  tools: [search_tool, calculator]
)

agent.tools
# => ["google_search", "calculator"]
```
```ruby
agent.run(question: "How many full soccer fields would be needed to cover the distance between NYC and DC in a straight line?")
#=> "Approximately 2,945 soccer fields would be needed to cover the distance between NYC and DC in a straight line."
```

#### SQL-Query Agent

Add `gem "sequel"` to your Gemfile

```ruby
database = Langchain::Tool::Database.new(connection_string: "postgres://user:password@localhost:5432/db_name")

agent = Langchain::Agent::SQLQueryAgent.new(llm: Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"]), db: database)
```
```ruby
agent.run(question: "How many users have a name with length greater than 5 in the users table?")
#=> "14 users have a name with length greater than 5 in the users table."
```

#### Demo
![May-12-2023 13-09-13](https://github.com/andreibondarev/langchainrb/assets/541665/6bad4cd9-976c-420f-9cf9-b85bf84f7eaf)

![May-12-2023 13-07-45](https://github.com/andreibondarev/langchainrb/assets/541665/9aacdcc7-4225-4ea0-ab96-7ee48826eb9b)

#### Available Tools 🛠️

| Name         | Description                                        | ENV Requirements                                              | Gem Requirements                          |
| ------------ | :------------------------------------------------: | :-----------------------------------------------------------: | :---------------------------------------: |
| "calculator" | Useful for getting the result of a math expression |                                                               | `gem "eqn", "~> 1.6.5"`                   |
| "database"   | Useful for querying a SQL database |                                                               | `gem "sequel", "~> 5.68.0"`                   |
| "ruby_code_interpreter" | Interprets Ruby expressions             |                                                               | `gem "safe_ruby", "~> 1.0.4"`             |
| "google_search"     | A wrapper around Google Search                     | `ENV["SERPAPI_API_KEY"]` (https://serpapi.com/manage-api-key) | `gem "google_search_results", "~> 2.0.0"` |
| "weather"  | Calls Open Weather API to retrieve the current weather        |      `ENV["OPEN_WEATHER_API_KEY"]` (https://home.openweathermap.org/api_keys)               | `gem "open-weather-ruby-client", "~> 0.3.0"`    |
| "wikipedia"  | Calls Wikipedia API to retrieve the summary        |                                                               | `gem "wikipedia-client", "~> 1.17.0"`     |

#### Loaders 🚚

Need to read data from various sources? Load it up.

##### Usage

Just call `Langchan::Loader.load` with the path to the file or a URL you want to load.

```ruby
Langchain::Loader.load('/path/to/file.pdf')
```

or

```ruby
Langchain::Loader.load('https://www.example.com/file.pdf')
```

##### Supported Formats


| Format | Pocessor                     |       Gem Requirements       |
| ------ | ---------------------------- | :--------------------------: |
| docx   | Langchain::Processors::Docx  |   `gem "docx", "~> 0.8.0"`   |
| html   | Langchain::Processors::HTML  | `gem "nokogiri", "~> 1.13"`  |
| pdf    | Langchain::Processors::PDF   | `gem "pdf-reader", "~> 1.4"` |
| text   | Langchain::Processors::Text  |                              |
| JSON   | Langchain::Processors::JSON  |                              |
| JSONL  | Langchain::Processors::JSONL |                              |
| csv    | Langchain::Processors::CSV   |                              |
| xlsx   | Langchain::Processors::Xlsx  |   `gem "roo", "~> 2.10.0"`   |

## Examples
Additional examples available: [/examples](https://github.com/andreibondarev/langchainrb/tree/main/examples)

## Logging

LangChain.rb uses standard logging mechanisms and defaults to `:warn` level. Most messages are at info level, but we will add debug or warn statements as needed.
To show all log messages:

```ruby
Langchain.logger.level = :info
```

## Development

1. `git clone https://github.com/andreibondarev/langchainrb.git`
2. `cp .env.example .env`, then fill out the environment variables in `.env`
3. `bundle exec rake` to ensure that the tests pass and to run standardrb
4. `bin/console` to load the gem in a REPL session. Feel free to add your own instances of LLMs, Tools, Agents, etc. and experiment with them.
5. Optionally, install lefthook git hooks for pre-commit to auto lint: `gem install lefthook && lefthook install -f`

## Discord
Join us in the [Langchain.rb](https://discord.gg/WDARp7J2n8) Discord server.

## Core Contributors
[<img style="border-radius:50%" alt="Andrei Bondarev" src="https://avatars.githubusercontent.com/u/541665?v=4" width="80" height="80" class="avatar">](https://twitter.com/rushing_andrei)

## Contributors
[<img style="border-radius:50%" alt="Alex Chaplinsky" src="https://avatars.githubusercontent.com/u/695947?v=4" width="80" height="80" class="avatar">](https://github.com/alchaplinsky)
[<img style="border-radius:50%" alt="Josh Nichols" src="https://avatars.githubusercontent.com/u/159?v=4" width="80" height="80" class="avatar">](https://github.com/technicalpickles)
[<img style="border-radius:50%" alt="Matt Lindsey" src="https://avatars.githubusercontent.com/u/5638339?v=4" width="80" height="80" class="avatar">](https://github.com/mattlindsey)
[<img style="border-radius:50%" alt="Ricky Chilcott" src="https://avatars.githubusercontent.com/u/445759?v=4" width="80" height="80" class="avatar">](https://github.com/rickychilcott)
[<img style="border-radius:50%" alt="Moeki Kawakami" src="https://avatars.githubusercontent.com/u/72325947?v=4" width="80" height="80" class="avatar">](https://github.com/moekidev)
[<img style="border-radius:50%" alt="Jens Stmrs" src="https://avatars.githubusercontent.com/u/3492669?v=4" width="80" height="80" class="avatar">](https://github.com/faustus7)
[<img style="border-radius:50%" alt="Rafael Figueiredo" src="https://avatars.githubusercontent.com/u/35845775?v=4" width="80" height="80" class="avatar">](https://github.com/rafaelqfigueiredo)
[<img style="border-radius:50%" alt="Piero Dotti" src="https://avatars.githubusercontent.com/u/5167659?v=4" width="80" height="80" class="avatar">](https://github.com/ProGM)
[<img style="border-radius:50%" alt="Michał Ciemięga" src="https://avatars.githubusercontent.com/u/389828?v=4" width="80" height="80" class="avatar">](https://github.com/zewelor)
[<img style="border-radius:50%" alt="Bruno Bornsztein" src="https://avatars.githubusercontent.com/u/3760?v=4" width="80" height="80" class="avatar">](https://github.com/bborn)
[<img style="border-radius:50%" alt="Tim Williams" src="https://avatars.githubusercontent.com/u/1192351?v=4" width="80" height="80" class="avatar">](https://github.com/timrwilliams)
[<img style="border-radius:50%" alt="Zhenhang Tung" src="https://avatars.githubusercontent.com/u/8170159?v=4" width="80" height="80" class="avatar">](https://github.com/ZhenhangTung)
[<img style="border-radius:50%" alt="Hama" src="https://avatars.githubusercontent.com/u/38002468?v=4" width="80" height="80" class="avatar">](https://github.com/akmhmgc)
[<img style="border-radius:50%" alt="Josh Weir" src="https://avatars.githubusercontent.com/u/10720337?v=4" width="80" height="80" class="avatar">](https://github.com/joshweir)
[<img style="border-radius:50%" alt="Arthur Hess" src="https://avatars.githubusercontent.com/u/446035?v=4" width="80" height="80" class="avatar">](https://github.com/arthurhess)
[<img style="border-radius:50%" alt="Jin Shen" src="https://avatars.githubusercontent.com/u/54917718?v=4" width="80" height="80" class="avatar">](https://github.com/jacshen-ebay)
[<img style="border-radius:50%" alt="Earle Bunao" src="https://avatars.githubusercontent.com/u/4653624?v=4" width="80" height="80" class="avatar">](https://github.com/erbunao)
[<img style="border-radius:50%" alt="Maël H." src="https://avatars.githubusercontent.com/u/61985678?v=4" width="80" height="80" class="avatar">](https://github.com/mael-ha)
[<img style="border-radius:50%" alt="Chris O. Adebiyi" src="https://avatars.githubusercontent.com/u/62605573?v=4" width="80" height="80" class="avatar">](https://github.com/oluvvafemi)
[<img style="border-radius:50%" alt="Aaron Breckenridge" src="https://avatars.githubusercontent.com/u/201360?v=4" width="80" height="80" class="avatar">](https://github.com/breckenedge)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=andreibondarev/langchainrb&type=Date)](https://star-history.com/#andreibondarev/langchainrb&Date)

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/andreibondarev/langchain.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
