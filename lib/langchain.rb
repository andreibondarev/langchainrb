# frozen_string_literal: true

require "logger"
require "pathname"
require "colorize"

require_relative "./langchain/version"

# Langchain.rb a is library for building LLM-backed Ruby applications. It is an abstraction layer that sits on top of the emerging AI-related tools that makes it easy for developers to consume and string those services together.
#
# = Installation
# Install the gem and add to the application's Gemfile by executing:
#
#     $ bundle add langchainrb
#
# If bundler is not being used to manage dependencies, install the gem by executing:
#
#     $ gem install langchainrb
#
# Require the gem to start using it:
#
#     require "langchain"
#
# = Concepts
#
# == Processors
# Processors load and parse/process various data types such as CSVs, PDFs, Word documents, HTML pages, and others.
#
# == Chunkers
# Chunkers split data based on various available options such as delimeters, chunk sizes or custom-defined functions. Chunkers are used when data needs to be split up before being imported in vector databases.
#
# == Prompts
# Prompts are structured inputs to the LLMs. Prompts provide instructions, context and other user input that LLMs use to generate responses.
#
# == Large Language Models (LLMs)
# LLM is a language model consisting of a neural network with many parameters (typically billions of weights or more), trained on large quantities of unlabeled text using self-supervised learning or semi-supervised learning.
#
# == Vectorsearch Databases
# Vector database is a type of database that stores data as high-dimensional vectors, which are mathematical representations of features or attributes. Each vector has a certain number of dimensions, which can range from tens to thousands, depending on the complexity and granularity of the data.
#
# == Embedding
# Word embedding or word vector is an approach with which we represent documents and words. It is defined as a numeric vector input that allows words with similar meanings to have the same representation. It can approximate meaning and represent a word in a lower dimensional space.
#
#
# = Logging
#
# LangChain.rb uses standard logging mechanisms and defaults to :debug level. Most messages are at info level, but we will add debug or warn statements as needed. To show all log messages:
#
# Langchain.logger.level = :info
module Langchain
  autoload :Loader, "langchain/loader"
  autoload :Data, "langchain/data"
  autoload :Conversation, "langchain/conversation"
  autoload :ConversationMemory, "langchain/conversation_memory"
  autoload :DependencyHelper, "langchain/dependency_helper"
  autoload :ContextualLogger, "langchain/contextual_logger"

  class << self
    # @return [ContextualLogger]
    attr_reader :logger

    # @param logger [Logger]
    # @return [ContextualLogger]
    def logger=(logger)
      @logger = ContextualLogger.new(logger)
    end

    # @return [Pathname]
    attr_reader :root
  end

  self.logger ||= ::Logger.new($stdout, level: :warn)

  @root = Pathname.new(__dir__)

  module Agent
    autoload :Base, "langchain/agent/base"
    autoload :ReActAgent, "langchain/agent/react_agent/react_agent.rb"
    autoload :SQLQueryAgent, "langchain/agent/sql_query_agent/sql_query_agent.rb"
  end

  module Chunker
    autoload :Base, "langchain/chunker/base"
    autoload :Text, "langchain/chunker/text"
    autoload :RecursiveText, "langchain/chunker/recursive_text"
  end

  module Tool
    autoload :Base, "langchain/tool/base"
    autoload :Calculator, "langchain/tool/calculator"
    autoload :RubyCodeInterpreter, "langchain/tool/ruby_code_interpreter"
    autoload :GoogleSearch, "langchain/tool/google_search"
    autoload :Weather, "langchain/tool/weather"
    autoload :Wikipedia, "langchain/tool/wikipedia"
    autoload :Database, "langchain/tool/database"
  end

  module Processors
    autoload :Base, "langchain/processors/base"
    autoload :CSV, "langchain/processors/csv"
    autoload :Docx, "langchain/processors/docx"
    autoload :HTML, "langchain/processors/html"
    autoload :JSON, "langchain/processors/json"
    autoload :JSONL, "langchain/processors/jsonl"
    autoload :PDF, "langchain/processors/pdf"
    autoload :Text, "langchain/processors/text"
    autoload :Xlsx, "langchain/processors/xlsx"
  end

  module Utils
    module TokenLength
      autoload :BaseValidator, "langchain/utils/token_length/base_validator"
      autoload :AI21Validator, "langchain/utils/token_length/ai21_validator"
      autoload :CohereValidator, "langchain/utils/token_length/cohere_validator"
      autoload :GooglePalmValidator, "langchain/utils/token_length/google_palm_validator"
      autoload :OpenAIValidator, "langchain/utils/token_length/openai_validator"
      autoload :TokenLimitExceeded, "langchain/utils/token_length/token_limit_exceeded"
    end
  end

  module Vectorsearch
    autoload :Base, "langchain/vectorsearch/base"
    autoload :Chroma, "langchain/vectorsearch/chroma"
    autoload :Hnswlib, "langchain/vectorsearch/hnswlib"
    autoload :Milvus, "langchain/vectorsearch/milvus"
    autoload :Pinecone, "langchain/vectorsearch/pinecone"
    autoload :Pgvector, "langchain/vectorsearch/pgvector"
    autoload :Qdrant, "langchain/vectorsearch/qdrant"
    autoload :Weaviate, "langchain/vectorsearch/weaviate"
  end

  module LLM
    autoload :AI21, "langchain/llm/ai21"
    autoload :Base, "langchain/llm/base"
    autoload :Cohere, "langchain/llm/cohere"
    autoload :GooglePalm, "langchain/llm/google_palm"
    autoload :HuggingFace, "langchain/llm/hugging_face"
    autoload :LlamaCpp, "langchain/llm/llama_cpp"
    autoload :OpenAI, "langchain/llm/openai"
    autoload :Replicate, "langchain/llm/replicate"
  end

  module Prompt
    require_relative "langchain/prompt/loading"

    autoload :Base, "langchain/prompt/base"
    autoload :PromptTemplate, "langchain/prompt/prompt_template"
    autoload :FewShotPromptTemplate, "langchain/prompt/few_shot_prompt_template"
  end

  module ActiveRecord
    autoload :Hooks, "langchain/active_record/hooks"
  end

  module OutputParsers
    autoload :Base, "langchain/output_parsers/base"
    autoload :StructuredOutputParser, "langchain/output_parsers/structured"
    autoload :OutputFixingParser, "langchain/output_parsers/fix"
  end

  module Errors
    class BaseError < StandardError; end
  end
end

require "langchain/railtie" if defined?(Rails)
