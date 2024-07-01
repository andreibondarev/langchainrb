# frozen_string_literal: true

module Langchain::Tool
  class Wikipedia < Base
    #
    # Tool that adds the capability to search using the Wikipedia API
    #
    # Gem requirements:
    #     gem "wikipedia-client", "~> 1.17.0"
    #
    # Usage:
    #     wikipedia = Langchain::Tool::Wikipedia.new
    #     wikipedia.execute(input: "The Roman Empire")
    #
    NAME = "wikipedia"
    FUNCTIONS = [:execute]

    # Initializes the Wikipedia tool
    def initialize
      super()

      depends_on "wikipedia-client", req: "wikipedia"
    end

    # Executes Wikipedia API search and returns the answer
    #
    # @param input [String] search query
    # @return [String] Answer
    def execute(input:)
      Langchain.logger.info("Executing \"#{input}\"", for: self.class)

      page = ::Wikipedia.find(input)
      # It would be nice to figure out a way to provide page.content but the LLM token limit is an issue
      page.summary
    end
  end
end
