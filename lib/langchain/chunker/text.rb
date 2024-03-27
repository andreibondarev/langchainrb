# frozen_string_literal: true

require "baran"

module Langchain
  module Chunker
    # Simple text chunker
    #
    # Usage:
    #     Langchain::Chunker::Text.new(text).chunks
    class Text < Base
      attr_reader :text, :chunk_size, :chunk_overlap, :separator

      # @param [String] text
      # @param [Integer] chunk_size
      # @param [Integer] chunk_overlap
      # @param [String] separator
      def initialize(text, chunk_size: 1000, chunk_overlap: 200, separator: "\n\n")
        @text = text
        @chunk_size = chunk_size
        @chunk_overlap = chunk_overlap
        @separator = separator
      end

      # Split text into chunks
      #
      # @param [String] source
      # @return [Array<Langchain::Chunk>]
      def chunks(source: nil)
        splitter = Baran::CharacterTextSplitter.new(
          chunk_size: chunk_size,
          chunk_overlap: chunk_overlap,
          separator: separator
        )

        splitter.chunks(text).map do |chunk|
          Langchain::Chunk.new(text: chunk[:text], source: source)
        end
      end
    end
  end
end
