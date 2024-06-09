# frozen_string_literal: true

module Langchain
  class Chunk
    # The chunking process is the process of splitting a document into smaller chunks and creating instances of Langchain::Chunk class

    attr_reader :text, :source

    # Initialize a new chunk
    # @param [String] text
    # @param [String] source
    # @return [Langchain::Chunk]
    def initialize(text:, source: nil)
      @text = text
      @source = source
    end
  end
end
