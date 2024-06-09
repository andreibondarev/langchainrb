# frozen_string_literal: true

require "pragmatic_segmenter"

module Langchain
  module Chunker
    # This chunker splits text by sentences.
    #
    # Usage:
    #     Langchain::Chunker::Sentence.new(text).chunks
    class Sentence < Base
      attr_reader :text

      # @param text [String]
      # @return [Langchain::Chunker::Sentence]
      def initialize(text)
        @text = text
      end

      # @return [Array<Langchain::Chunk>]
      def chunks(source: nil)
        ps = PragmaticSegmenter::Segmenter.new(text: text)
        ps.segment.map do |chunk|
          Langchain::Chunk.new(text: chunk, source: source)
        end
      end
    end
  end
end
