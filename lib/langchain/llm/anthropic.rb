# frozen_string_literal: true

module Langchain::LLM
  #
  # Wrapper around Anthropic APIs.
  #
  # Gem requirements:
  #   gem "anthropic", "~> 0.1.0"
  #
  # Usage:
  #     anthorpic = Langchain::LLM::Anthropic.new(
  #       api_key: ENV["ANTHROPIC_API_KEY"]),
  #       llm_options: { extra_headers: { anthropic_beta: "tools-2024-04-04"} }
  #     )
  #
  class Anthropic < Base
    DEFAULTS = {
      temperature: 0.0,
      completion_model_name: "claude-2.1",
      chat_completion_model_name: "claude-3-sonnet-20240229",
      max_tokens_to_sample: 256
    }.freeze

    # Initialize an Anthropic LLM instance
    #
    # @param api_key [String] The API key to use
    # @param llm_options [Hash] Options to pass to the Anthropic client
    # @param default_options [Hash] Default options to use on every call to LLM, e.g.: { temperature:, completion_model_name:, chat_completion_model_name:, max_tokens_to_sample: }
    # @return [Langchain::LLM::Anthropic] Langchain::LLM::Anthropic instance
    def initialize(api_key:, llm_options: {}, default_options: {})
      depends_on "anthropic"

      @client = ::Anthropic::Client.new(access_token: api_key, **llm_options)
      @defaults = DEFAULTS.merge(default_options)
    end

    # Generate a completion for a given prompt
    #
    # @param prompt [String] The prompt that you want Claude to complete
    # @param model [String] The model that will complete your prompt.
    # @param max_tokens_to_sample [Integer] The maximum number of tokens to generate before stopping
    # @param stop_sequences [Array<String>] Sequences that will cause the model to stop generating
    # @param temperature [Float] Amount of randomness injected into the response
    # @param top_p [Integer] Use nucleus sampling
    # @param top_k [Integer] Only sample from the top K options for each subsequent token
    # @param metadata [Hash] An object describing metadata about the request.
    # @param stream [Boolean] Whether to stream the response
    # @return [Langchain::LLM::AnthropicResponse] The completion
    def complete(
      prompt:,
      model: @defaults[:completion_model_name],
      max_tokens_to_sample: @defaults[:max_tokens_to_sample],
      stop_sequences: nil,
      temperature: @defaults[:temperature],
      top_p: nil,
      top_k: nil,
      metadata: nil,
      stream: nil
    )
      raise ArgumentError.new("model argument is required") if model.empty?
      raise ArgumentError.new("max_tokens_to_sample argument is required") if max_tokens_to_sample.nil?

      parameters = {
        model: model,
        prompt: prompt,
        max_tokens_to_sample: max_tokens_to_sample,
        temperature: temperature
      }
      parameters[:stop_sequences] = stop_sequences if stop_sequences
      parameters[:top_p] = top_p if top_p
      parameters[:top_k] = top_k if top_k
      parameters[:metadata] = metadata if metadata
      parameters[:stream] = stream if stream

      response = client.complete(parameters: parameters)
      Langchain::LLM::AnthropicResponse.new(response)
    end

    # Generate a chat completion for given messages
    #
    # @param messages [Array<Hash>] Input messages
    # @param model [String] Model that will complete your prompt
    # @param max_tokens [Integer] Maximum number of tokens to generate before stopping
    # @param metadata [Hash] Object describing metadata about the request
    # @param stop_sequences [Array<String>] Custom text sequences that will cause the model to stop generating
    # @param stream [Boolean] Whether to incrementally stream the response using server-sent events
    # @param system [String] System prompt
    # @param temperature [Float] Amount of randomness injected into the response
    # @param tools [Array<Hash>] Definitions of tools that the model may use
    # @param top_k [Integer] Only sample from the top K options for each subsequent token
    # @param top_p [Float] Use nucleus sampling.
    # @return [Langchain::LLM::AnthropicResponse] The chat completion
    def chat(
      messages: [],
      model: @defaults[:chat_completion_model_name],
      max_tokens: @defaults[:max_tokens_to_sample],
      metadata: nil,
      stop_sequences: nil,
      stream: nil,
      system: nil,
      temperature: @defaults[:temperature],
      tools: [],
      response_schema: {},
      top_k: nil,
      top_p: nil
    )
      raise ArgumentError.new("messages argument is required") if messages.empty?
      raise ArgumentError.new("model argument is required") if model.empty?
      raise ArgumentError.new("max_tokens argument is required") if max_tokens.nil?
      raise ArgumentError.new("response_schema must be of type JSON") if response_schema && !response_schema.is_a?(Hash)

      parameters = {
        messages: messages,
        model: model,
        max_tokens: max_tokens,
        temperature: temperature
      }
      parameters[:metadata] = metadata if metadata
      parameters[:stop_sequences] = stop_sequences if stop_sequences
      parameters[:stream] = stream if stream
      parameters[:system] = system if system

      if response_schema.any?
        parameters[:tools] = [
          {
            name: "response_schema",
            description: "Correctly extracted data with all the required parameters with correct types",
            input_schema: response_schema
          }
        ]
      elsif tools.any?
        parameters[:tools] = tools
      end

      parameters[:top_k] = top_k if top_k
      parameters[:top_p] = top_p if top_p

      response = client.messages(parameters: parameters)

      Langchain::LLM::AnthropicResponse.new(response)
    end
  end
end
