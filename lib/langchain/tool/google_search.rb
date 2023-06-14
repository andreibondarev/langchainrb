# frozen_string_literal: true

module Langchain::Tool
  class GoogleSearch < Base
    #
    # Wrapper around Google Serp SPI
    #
    # Gem requirements: gem "google_search_results", "~> 2.0.0"
    #
    # Usage:
    # search = Langchain::Tool::GoogleSearch.new(api_key: "YOUR_API_KEY")
    # search.execute(input: "What is the capital of France?")
    #

    NAME = "google_search"

    description <<~DESC
      A wrapper around Google Search.

      Useful for when you need to answer questions about current events.
      Always one of the first options when you need to find information on internet.

      Input should be a search query.
    DESC

    attr_reader :api_key

    #
    # Initializes the Google Search tool
    #
    # @param api_key [String] Search API key
    # @return [Langchain::Tool::GoogleSearch] Google search tool
    #
    def initialize(api_key:)
      depends_on "google_search_results"
      require "google_search_results"
      @api_key = api_key
    end

    #
    # Executes Google Search and returns hash_results JSON
    #
    # @param input [String] search query
    # @return [Hash] hash_results JSON
    #
    def self.execute_search(input:)
      new.execute_search(input: input)
    end

    #
    # Executes Google Search and returns the result
    #
    # @param input [String] search query
    # @return [String] Answer
    #
    def execute(input:)
      Langchain.logger.info("Executing \"#{input}\"", for: self.class)

      hash_results = execute_search(input: input)

      # TODO: Glance at all of the fields that langchain Python looks through: https://github.com/hwchase17/langchain/blob/v0.0.166/langchain/utilities/serpapi.py#L128-L156
      # We may need to do the same thing here.
      hash_results.dig(:answer_box, :answer) ||
        hash_results.dig(:answer_box, :snippet) ||
        hash_results.dig(:organic_results, 0, :snippet)
    end

    #
    # Executes Google Search and returns hash_results JSON
    #
    # @param input [String] search query
    # @return [Hash] hash_results JSON
    #
    def execute_search(input:)
      ::GoogleSearch
        .new(q: input, serp_api_key: api_key)
        .get_hash
    end
  end
end
