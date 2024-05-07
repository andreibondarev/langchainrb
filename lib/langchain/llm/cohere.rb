# frozen_string_literal: true

module Langchain::LLM
  #
  # Wrapper around the Cohere API.
  #
  # Gem requirements:
  #     gem "cohere-ruby", "~> 0.9.6"
  #
  # Usage:
  #     cohere = Langchain::LLM::Cohere.new(api_key: ENV["COHERE_API_KEY"])
  #
  class Cohere < Base
    DEFAULTS = {
      temperature: 0.0,
      completion_model_name: "command",
      embeddings_model_name: "small",
      chat_completion_model_name: "command-r",
      dimension: 1024,
      truncate: "START"
    }.freeze

    def initialize(api_key, default_options = {})
      depends_on "cohere-ruby", req: "cohere"

      @client = ::Cohere::Client.new(api_key)
      @defaults = DEFAULTS.merge(default_options)
    end

    # Generate an embedding for a given text
    #
    # @param texts [Array<String>] An array of strings for the model to embed
    # @param model [String] The identifier of the model
    # @param input_type [String] Specifies the type of input passed to the model. Required for embedding models v3 and higher
    # @param embedding_types [Array] Specifies the types of embeddings you want to get back
    # @param truncate [String] One of NONE|START|END to specify how the API will handle inputs longer than the maximum token length
    # @return [Langchain::LLM::CohereResponse] Response object
    def embed(
      texts: [],
      model: @defaults[:embeddings_model_name],
      input_type: nil,
      embedding_types: [],
      truncate: nil
    )
      parameters = {
        texts: texts,
        model: @defaults[:embeddings_model_name]
      }
      parameters[:input_type] = input_type if input_type
      parameters[:embedding_types] = embedding_types if embedding_types.any?
      parameters[:truncate] = truncate if truncate.any?

      response = client.embed(parameters)

      Langchain::LLM::CohereResponse.new response, model: @defaults[:embeddings_model_name]
    end

    # Generate a completion for a given prompt
    #
    # @param prompt [String] The prompt to generate a completion for
    # @param params[:stop_sequences]
    # @return [Langchain::LLM::CohereResponse] Response object
    def complete(prompt:, **params)
      default_params = {
        prompt: prompt,
        temperature: @defaults[:temperature],
        model: @defaults[:completion_model_name],
        truncate: @defaults[:truncate]
      }

      if params[:stop_sequences]
        default_params[:stop_sequences] = params.delete(:stop_sequences)
      end

      default_params.merge!(params)

      default_params[:max_tokens] = Langchain::Utils::TokenLength::CohereValidator.validate_max_tokens!(prompt, default_params[:model], llm: client)

      response = client.generate(**default_params)
      Langchain::LLM::CohereResponse.new response, model: @defaults[:completion_model_name]
    end

    def chat(
      messages: [],
      model: @defaults[:chat_completion_model_name],
      stream: nil,
      preamble: nil,
      chat_history: nil,
      conversation_id: nil,
      prompt_truncation: nil,
      connectors: [],
      search_queries_only: nil,
      documents: [],
      temperature: @defaults[:temperature],
      max_tokens: nil,
      max_input_tokens: nil,
      k: nil,
      p: nil,
      seed: nil,
      stop_sequences: [],
      frequency_penalty: nil,
      presence_penalty: nil,
      tools: [],
      tool_results: [],
      response_schema: {}
    )
      raise ArgumentError.new("messages argument is required") if messages.empty?
      raise ArgumentError.new("response_schema must be of type JSON") if response_schema && !response_schema.is_a?(Hash)

      parameters = {
        # Pop the last message from the array and send as expected `message:` parameter
        message: messages.pop
      }
      # Send the remaining messages as `chat_history:` parameter
      parameters[:chat_history] = messages if messages.any?
      parameters[:model] = model if model
      parameters[:stream] = stream if stream
      parameters[:preamble] = preamble if preamble
      parameters[:conversation_id] = conversation_id if conversation_id
      parameters[:prompt_truncation] = prompt_truncation if prompt_truncation
      parameters[:connectors] = connectors if connectors.any?
      parameters[:search_queries_only] = search_queries_only if search_queries_only
      parameters[:documents] = documents if documents.any?
      parameters[:temperature] = temperature if temperature
      parameters[:max_tokens] = max_tokens if max_tokens
      parameters[:max_input_tokens] = max_input_tokens if max_input_tokens
      parameters[:k] = k if k
      parameters[:p] = p if p
      parameters[:seed] = seed if seed
      parameters[:stop_sequences] = stop_sequences if stop_sequences.any?
      parameters[:frequency_penalty] = frequency_penalty if frequency_penalty
      parameters[:presence_penalty] = presence_penalty if presence_penalty

      if response_schema.any?
        parameters[:tools] = [
          {
            type: "function",
            function: {
              name: "response_schema",
              description: "Correctly extracted data with all the required parameters with correct types",
              parameters: response_schema
            }
          }
        ]
      elsif tools.any?
        parameters[:tools] = tools
      end

      parameters[:tool_results] = tool_results if tool_results.any?

      response = client.chat(**parameters)
      Langchain::LLM::CohereResponse.new(response)
    end

    # Generate a summary in English for a given text
    #
    # More parameters available to extend this method with: https://github.com/andreibondarev/cohere-ruby/blob/0.9.4/lib/cohere/client.rb#L107-L115
    #
    # @param text [String] The text to generate a summary for
    # @return [String] The summary
    def summarize(text:)
      response = client.summarize(text: text)
      response.dig("summary")
    end
  end
end
