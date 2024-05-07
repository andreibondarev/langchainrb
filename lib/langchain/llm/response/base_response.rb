# frozen_string_literal: true

module Langchain
  module LLM
    class BaseResponse
      attr_reader :raw_response, :model

      # Save context in the response when doing RAG workflow vectorsearch#ask()
      attr_accessor :context

      def initialize(raw_response, model: nil)
        @raw_response = raw_response
        @model = model
      end

      # Returns the timestamp when the response was created
      #
      # @return [Time]
      def created_at
        raise NotImplementedError
      end

      # Returns the completion text
      #
      # @return [String]
      #
      def completion
        raise NotImplementedError
      end

      # Returns the chat completion text
      #
      # @return [String]
      #
      def chat_completion
        raise NotImplementedError
      end

      # Returns tool calls from the model
      #
      # @return [Array]
      def tool_calls
        raise NotImplementedError
      end

      # Returns the JSON schema structured response
      #
      # @return [Hash] JSON schema structured response
      def response_schema
        raise NotImplementedError
      end

      # Return the first embedding
      #
      # @return [Array<Float>]
      def embedding
        raise NotImplementedError
      end

      # Return the completion candidates
      #
      # @return [Array<String>]
      def completions
        raise NotImplementedError
      end

      # Return the chat completion candidates
      #
      # @return [Array<String>]
      def chat_completions
        raise NotImplementedError
      end

      # Return the embeddings
      #
      # @return [Array<Array>]
      def embeddings
        raise NotImplementedError
      end

      # Number of tokens utilized in the prompt
      #
      # @return [Integer]
      def prompt_tokens
        raise NotImplementedError
      end

      # Number of tokens utilized to generate the completion
      #
      # @return [Integer]
      def completion_tokens
        raise NotImplementedError
      end

      # Total number of tokens utilized
      #
      # @return [Integer]
      def total_tokens
        raise NotImplementedError
      end

      # Response's role parameter
      #
      # @return [String]
      def role
        raise NotImplementedError
      end
    end
  end
end
