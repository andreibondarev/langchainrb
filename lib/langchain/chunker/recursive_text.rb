# frozen_string_literal: true

require "baran"

module Langchain
  module Chunker
    # Recursive text chunker. Preferentially splits on separators.
    #
    # Usage:
    #     Langchain::Chunker::RecursiveText.new(text).chunks
    class RecursiveText < Base
      attr_reader :text, :chunk_size, :chunk_overlap, :separators

      # @param [String] text
      # @param [Integer] chunk_size
      # @param [Integer] chunk_overlap
      # @param [Array<String>] separators
      def initialize(text, chunk_size: 1000, chunk_overlap: 200, separators: ["\n\n"])
        @text = text
        @chunk_size = chunk_size
        @chunk_overlap = chunk_overlap
        @separators = separators
      end

      # Split the text into chunks
      #
      # @param [String] source
      # @return [Array<Langchain::Chunk>]
      def chunks(source: nil)
        splitter = Baran::RecursiveCharacterTextSplitter.new(
          chunk_size: chunk_size,
          chunk_overlap: chunk_overlap,
          separators: separators
        )

        splitter.chunks(text).map do |chunk|
          Langchain::Chunk.new(text: chunk[:text], source: source)
        end
      end
    end
  end
end
