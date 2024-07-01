# frozen_string_literal: true

require "tool_tailor"

module Langchain::Tool
  # = Tools
  #
  # Tools are used by Agents to perform specific tasks. A 'Tool' is a collection of functions ("methods").
  #
  # == Available Tools
  #
  # - {Langchain::Tool::Calculator}: calculate the result of a math expression
  # - {Langchain::Tool::Database}: executes SQL queries
  # - {Langchain::Tool::FileSystem}: interacts with the file system
  # - {Langchain::Tool::GoogleSearch}: search on Google (via SerpAPI)
  # - {Langchain::Tool::RubyCodeInterpreter}: runs ruby code
  # - {Langchain::Tool::Weather}: gets current weather data
  # - {Langchain::Tool::Wikipedia}: search on Wikipedia
  #
  # == Usage
  #
  # 1. Pick the tools you'd like to pass to an Agent and install the gems listed under **Gem Requirements**
  #
  #     # For example to use the Calculator, GoogleSearch, and Wikipedia:
  #     gem install eqn
  #     gem install google_search_results
  #     gem install wikipedia-client
  #
  # 2. Set the environment variables listed under **ENV Requirements**
  #
  #     export SERPAPI_API_KEY=paste-your-serpapi-api-key-here
  #
  # 3. Pass the tools when Agent is instantiated.
  #
  #     agent = Langchain::Assistant.new(
  #       llm: Langchain::LLM::OpenAI.new(api_key: "YOUR_API_KEY"), # or other LLM that supports function calling (coming soon)
  #       thread: Langchain::Thread.new,
  #       tools: [
  #         Langchain::Tool::GoogleSearch.new(api_key: "YOUR_API_KEY"),
  #         Langchain::Tool::Calculator.new,
  #         Langchain::Tool::Wikipedia.new
  #       ]
  #     )
  #
  # == Adding Tools
  #
  # 1. Inside lib/langchain/tool/ folder create a file with a class YourToolName that inherits from {Langchain::Tool::Base}
  # 2. Add `NAME=` and `FUNCTIONS=` constants in your Tool class
  # 4. Implement various public methods in your tool class
  # 5. Add your tool to the {file:README.md}
  class Base
    include Langchain::DependencyHelper

    def initialize
      validate_constants
      validate_functions
    end

    # Returns the NAME constant of the tool
    #
    # @return [String] tool name
    def name
      self.class::NAME
    end

    # Returns the FUNCTIONS constant of the tool
    #
    # @return [String] tool functions
    def functions
      self.class::FUNCTIONS
    end

    def self.logger_options
      {
        color: :light_blue
      }
    end

    # Returns the tool as a list of OpenAI formatted functions
    #
    # @return [Array<Hash>] List of hashes representing the tool as OpenAI formatted functions
    def to_openai_tools
      method_annotations
    end

    # Returns the tool as a list of Anthropic formatted functions
    #
    # @return [Array<Hash>] List of hashes representing the tool as Anthropic formatted functions
    def to_anthropic_tools
      method_annotations.map do |annotation|
        # Slice out only the content of the "function" key
        annotation["function"]
          # Rename "parameters" to "input_schema" key
          .transform_keys("parameters" => "input_schema")
      end
    end

    # Returns the tool as a list of Google Gemini formatted functions
    #
    # @return [Array<Hash>] List of hashes representing the tool as Google Gemini formatted functions
    def to_google_gemini_tools
      method_annotations.map do |annotation|
        # Slice out only the content of the "function" key
        annotation["function"]
      end
    end

    # Return tool's method annotations as JSON
    #
    # @return [Hash] Tool's method annotations
    def method_annotations
      functions.map do |function|
        annotations = JSON.parse(ToolTailor.convert(self.class.instance_method(function)))
        annotations["function"]["name"].prepend("#{name}__")

        annotations
      end
    end

    private

    def validate_functions
      functions.each do |function|
        raise "Method #{function} not found for #{self.class}" unless respond_to?(function)
      end
    end

    def validate_constants
      validate_constant(:NAME)
      validate_constant(:FUNCTIONS)
    end

    def validate_constant(constant)
      raise "#{self.class} must define a #{constant} constant" unless self.class.const_defined?(constant)
    end
  end
end
