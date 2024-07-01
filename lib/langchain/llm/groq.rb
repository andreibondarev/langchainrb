# frozen_string_literal: true

module Langchain::LLM
  # LLM interface for Groq APIs: https://console.groq.com/playground
  #
  # Gem requirements:
  #    gem "groq", "~> 0.3.1"
  #
  # Usage:
  #    groq = Langchain::LLM::Groq.new(
  #      api_key: ENV["GROQ_API_KEY"],
  #      llm_options: {}, # Available options: https://github.com/drnic/groq-ruby/blob/develop/lib/groq/client.rb#L4-L10
  #      default_options: {},
  #      embedding_client: -> do
  #        Langchain::LLM::HuggingFace.new(
  #          api_key: ENV["HUGGING_FACE_API_KEY"],
  #          default_options: {
  #            temperature: 0.0,
  #            embeddings_model_name: "mixedbread-ai/mxbai-embed-large-v1",
  #            dimensions: 1_024
  #          }
  #        )
  #      end
  #    )
  class Groq < Base
    DEFAULTS = {
      n: 1,
      temperature: 0.0,
      chat_completion_model_name: "llama3-70b-8192",
      dimensions: 4_096
    }.freeze

    attr_reader :defaults

    # Initialize an Groq LLM instance
    #
    # @param api_key [String] The API key to use
    # @param client_options [Hash] Options to pass to the Groq::Client constructor
    def initialize(api_key:, llm_options: {}, default_options: {}, embedding_llm: nil)
      depends_on "groq", req: "groq"

      @client = ::Groq::Client.new(api_key: api_key, **llm_options)
      @defaults = DEFAULTS.merge(default_options)

      @embedding_client = embedding_llm.call if embedding_llm

      chat_parameters.update(
        model: {default: @defaults[:chat_completion_model_name]},
        n: {default: @defaults[:n]},
        temperature: {default: @defaults[:temperature]}
      )
      chat_parameters.ignore(:n)
      chat_parameters.remap(model: :model_id)
    end

    # Generate an embedding for a given text
    #
    # @param text [String] The text to generate an embedding for
    def embed(text:)
      @embedding_client.embed(text: text)
    end

    # Generate a chat completion for given messages.
    #
    # @param [Hash] params unified chat parmeters from [Langchain::LLM::Parameters::Chat::SCHEMA]
    # @option params [Array<Hash>] :messages List of messages comprising the conversation so far
    # @option params [String] :model ID of the model to use
    def chat(params = {}, &block)
      parameters = chat_parameters.to_params(params)

      raise ArgumentError.new("messages argument is required") if Array(parameters[:messages]).empty?
      raise ArgumentError.new("model argument is required") if parameters[:model_id].to_s.empty?
      if parameters[:tool_choice] && Array(parameters[:tools]).empty?
        raise ArgumentError.new("'tool_choice' is only allowed when 'tools' are specified.")
      end

      # TODO: Clean this part up
      if block
        @response_chunks = []
        parameters[:stream] = proc do |chunk, _bytesize|
          chunk_content = chunk.dig("choices", 0)
          @response_chunks << chunk
          yield chunk_content
        end
      end

      response = with_api_error_handling do
        client.chat(parameters.delete(:messages), **parameters)
      end

      response = response_from_chunks if block
      reset_response_chunks

      Langchain::LLM::GroqResponse.new(response)
    end

    def default_dimensions
      @defaults[:dimensions] || EMBEDDING_SIZES.fetch(defaults[:embeddings_model_name])
    end

    private

    attr_reader :response_chunks

    def reset_response_chunks
      @response_chunks = []
    end

    def with_api_error_handling
      response = yield
      return if response.empty?

      raise Langchain::LLM::ApiError.new "Groq API error: #{response.dig("error", "message")}" if response&.dig("error")

      response
    end

    def response_from_chunks
      grouped_chunks = @response_chunks.group_by { |chunk| chunk.dig("choices", 0, "index") }
      final_choices = grouped_chunks.map do |index, chunks|
        {
          "index" => index,
          "message" => {
            "role" => "assistant",
            "content" => chunks.map { |chunk| chunk.dig("choices", 0, "delta", "content") }.join
          },
          "finish_reason" => chunks.last.dig("choices", 0, "finish_reason")
        }
      end
      @response_chunks.first&.slice("id", "object", "created", "model")&.merge({"choices" => final_choices})
    end
  end
end
