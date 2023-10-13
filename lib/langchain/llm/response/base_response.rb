# frozen_string_literal: true

module Langchain
  module LLM
    class BaseResponse
      attr_reader :raw_response, :model

      def initialize(raw_response, model: nil)
        @raw_response = raw_response
        @model = model
      end

      def completion
        raise NotImplementedError
      end

      def chat_completion
        raise NotImplementedError
      end

      def embedding
        raise NotImplementedError
      end

      def completions
        raise NotImplementedError
      end

      def chat_completions
        raise NotImplementedError
      end

      def embeddings
        raise NotImplementedError
      end

      def prompt_tokens
        raise NotImplementedError
      end

      def completion_tokens
        raise NotImplementedError
      end

      def total_tokens
        raise NotImplementedError
      end
    end
  end
end
