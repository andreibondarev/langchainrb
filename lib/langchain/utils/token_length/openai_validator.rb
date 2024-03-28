# frozen_string_literal: true

# require "tiktoken_ruby"

module Langchain
  module Utils
    module TokenLength
      #
      # This class is meant to validate the length of the text passed in to OpenAI's API.
      # It is used to validate the token length before the API call is made
      #
      class OpenAIValidator < BaseValidator
        COMPLETION_TOKEN_LIMITS = {
          # GPT-4 Turbo has a separate token limit for completion
          # Source:
          # https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
          "gpt-4-1106-preview" => 4096,
          "gpt-4-vision-preview" => 4096,
          "gpt-3.5-turbo-1106" => 4096
        }

        # NOTE: The gpt-4-turbo-preview is an alias that will always point to the latest GPT 4 Turbo preview
        #   the future previews may have a different token limit!
        TOKEN_LIMITS = {
          # Source:
          # https://platform.openai.com/docs/api-reference/embeddings
          # https://platform.openai.com/docs/models/gpt-4
          "text-embedding-3-large" => 8191,
          "text-embedding-3-small" => 8191,
          "text-embedding-ada-002" => 8191,
          "gpt-3.5-turbo" => 4096,
          "gpt-3.5-turbo-0301" => 4096,
          "gpt-3.5-turbo-0613" => 4096,
          "gpt-3.5-turbo-1106" => 16385,
          "gpt-3.5-turbo-16k" => 16384,
          "gpt-3.5-turbo-16k-0613" => 16384,
          "text-davinci-003" => 4097,
          "text-davinci-002" => 4097,
          "code-davinci-002" => 8001,
          "gpt-4" => 8192,
          "gpt-4-0314" => 8192,
          "gpt-4-0613" => 8192,
          "gpt-4-32k" => 32768,
          "gpt-4-32k-0314" => 32768,
          "gpt-4-32k-0613" => 32768,
          "gpt-4-1106-preview" => 128000,
          "gpt-4-turbo-preview" => 128000,
          "gpt-4-0125-preview" => 128000,
          "gpt-4-vision-preview" => 128000,
          "text-curie-001" => 2049,
          "text-babbage-001" => 2049,
          "text-ada-001" => 2049,
          "davinci" => 2049,
          "curie" => 2049,
          "babbage" => 2049,
          "ada" => 2049
        }.freeze

        #
        # Calculate token length for a given text and model name
        #
        # @param text [String] The text to calculate the token length for
        # @param model_name [String] The model name to validate against
        # @return [Integer] The token length of the text
        #
        def self.token_length(text, model_name, options = {})
          # tiktoken-ruby doesn't support text-embedding-3-large or text-embedding-3-small yet
          if ["text-embedding-3-large", "text-embedding-3-small"].include?(model_name)
            model_name = "text-embedding-ada-002"
          end

          encoder = Tiktoken.encoding_for_model(model_name)
          encoder.encode(text).length
        end

        def self.token_limit(model_name)
          TOKEN_LIMITS[model_name]
        end

        def self.completion_token_limit(model_name)
          COMPLETION_TOKEN_LIMITS[model_name] || token_limit(model_name)
        end

        # If :max_tokens is passed in, take the lower of it and the calculated max_tokens
        def self.validate_max_tokens!(content, model_name, options = {})
          max_tokens = super(content, model_name, options)
          [options[:max_tokens], max_tokens].reject(&:nil?).min
        end

        # Copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        # Return the number of tokens used by a list of messages
        #
        # @param messages [Array<Hash>] The messages to calculate the token length for
        # @param model [String] The model name to validate against
        # @return [Integer] The token length of the messages
        #
        def self.token_length_from_messages(messages, model_name, options = {})
          encoding = Tiktoken.encoding_for_model(model_name)

          if ["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4-0314", "gpt-4-32k-0314", "gpt-4-0613", "gpt-4-32k-0613"].include?(model_name)
            tokens_per_message = 3
            tokens_per_name = 1
          elsif model_name == "gpt-3.5-turbo-0301"
            tokens_per_message = 4  # every message follows {role/name}\n{content}\n
            tokens_per_name = -1  # if there's a name, the role is omitted
          elsif model_name.include?("gpt-3.5-turbo")
            # puts "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
            return token_length_from_messages(messages, "gpt-3.5-turbo-0613", options)
          elsif model_name.include?("gpt-4")
            # puts "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
            return token_length_from_messages(messages, "gpt-4-0613", options)
          else
            raise NotImplementedError.new(
              "token_length_from_messages() is not implemented for model #{model_name}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."
            )
          end

          num_tokens = 0
          messages.each do |message|
            num_tokens += tokens_per_message
            message.each do |key, value|
              num_tokens += encoding.encode(value).length
              num_tokens += tokens_per_name if ["name", :name].include?(key)
            end
          end

          num_tokens += 3  # every reply is primed with assistant
          num_tokens
        end
      end
    end
  end
end
