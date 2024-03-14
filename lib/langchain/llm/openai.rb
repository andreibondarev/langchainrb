# frozen_string_literal: true

module Langchain::LLM
  # LLM interface for OpenAI APIs: https://platform.openai.com/overview
  #
  # Gem requirements:
  #    gem "ruby-openai", "~> 6.3.0"
  #
  # Usage:
  #    openai = Langchain::LLM::OpenAI.new(
  #      api_key: ENV["OPENAI_API_KEY"],
  #      llm_options: {}, # Available options: https://github.com/alexrudall/ruby-openai/blob/main/lib/openai/client.rb#L5-L13
  #      default_options: {}
  #    )
  class OpenAI < Base
    DEFAULTS = {
      n: 1,
      temperature: 0.0,
      chat_completion_model_name: "gpt-3.5-turbo",
      embeddings_model_name: "text-embedding-ada-002",
      token_counter: :tiktoken
    }.freeze

    EMBEDDING_SIZES = {
      "text-embedding-ada-002": 1536,
      "text-embedding-3-large": 3072,
      "text-embedding-3-small": 1536
    }.freeze

    LENGTH_VALIDATOR = Langchain::Utils::TokenLength::OpenAIValidator

    attr_reader :defaults

    # Initialize an OpenAI LLM instance
    #
    # @param api_key [String] The API key to use
    # @param llm_options [Hash] Options to pass to the OpenAI::Client constructor
    # @param default_options [Hash] Options to customize the default behavior of the LLM
    def initialize(api_key:, llm_options: {}, default_options: {})
      depends_on "ruby-openai", req: "openai"

      @client = ::OpenAI::Client.new(access_token: api_key, **llm_options)

      @defaults = DEFAULTS.merge(default_options)
    end

    # Generate an embedding for a given text
    #
    # @param text [String] The text to generate an embedding for
    # @param model [String] ID of the model to use
    # @param encoding_format [String] The format to return the embeddings in. Can be either float or base64.
    # @param user [String] A unique identifier representing your end-user
    # @return [Langchain::LLM::OpenAIResponse] Response object
    def embed(
      text:,
      model: defaults[:embeddings_model_name],
      encoding_format: nil,
      user: nil
    )
      raise ArgumentError.new("text argument is required") if text.empty?
      raise ArgumentError.new("model argument is required") if model.empty?
      raise ArgumentError.new("encoding_format must be either float or base64") if encoding_format && %w[float base64].include?(encoding_format)

      parameters = {
        input: text,
        model: model,
        dimensions: default_dimension
      }
      parameters[:encoding_format] = encoding_format if encoding_format
      parameters[:user] = user if user

      validate_max_tokens(text, parameters[:model])

      response = with_api_error_handling do
        client.embeddings(parameters: parameters)
      end

      Langchain::LLM::OpenAIResponse.new(response)
    end

    # rubocop:disable Style/ArgumentsForwarding
    # Generate a completion for a given prompt
    #
    # @param prompt [String] The prompt to generate a completion for
    # @param params [Hash] The parameters to pass to the `chat()` method
    # @return [Langchain::LLM::OpenAIResponse] Response object
    def complete(prompt:, **params)
      warn "DEPRECATED: `Langchain::LLM::OpenAI#complete` is deprecated, and will be removed in the next major version. Use `Langchain::LLM::OpenAI#chat` instead."

      if params[:stop_sequences]
        params[:stop] = params.delete(:stop_sequences)
      end
      # Should we still accept the `messages: []` parameter here?
      messages = [{role: "user", content: prompt}]
      chat(messages: messages, **params)
    end
    # rubocop:enable Style/ArgumentsForwarding

    # Generate a chat completion for a given prompt or messages.
    #
    # @param messages [Array<Hash>] List of messages comprising the conversation so far
    # @param model [String] ID of the model to use
    def chat(
      messages: [],
      model: defaults[:chat_completion_model_name],
      frequency_penalty: nil,
      logit_bias: nil,
      logprobs: nil,
      top_logprobs: nil,
      max_tokens: nil,
      n: defaults[:n],
      presence_penalty: nil,
      response_format: nil,
      seed: nil,
      stop: nil,
      stream: nil,
      temperature: defaults[:temperature],
      top_p: nil,
      tools: [],
      tool_choice: nil,
      user: nil,
      &block
    )
      raise ArgumentError.new("messages argument is required") if messages.empty?
      raise ArgumentError.new("model argument is required") if model.empty?
      raise ArgumentError.new("'tool_choice' is only allowed when 'tools' are specified.") if tool_choice && tools.empty?

      parameters = {
        messages: messages,
        model: model
      }
      parameters[:frequency_penalty] = frequency_penalty if frequency_penalty
      parameters[:logit_bias] = logit_bias if logit_bias
      parameters[:logprobs] = logprobs if logprobs
      parameters[:top_logprobs] = top_logprobs if top_logprobs
      # TODO: Fix max_tokens validation to account for tools/functions
      parameters[:max_tokens] = max_tokens if max_tokens # || validate_max_tokens(parameters[:messages], parameters[:model])
      parameters[:n] = n if n
      parameters[:presence_penalty] = presence_penalty if presence_penalty
      parameters[:response_format] = response_format if response_format
      parameters[:seed] = seed if seed
      parameters[:stop] = stop if stop
      parameters[:stream] = stream if stream
      parameters[:temperature] = temperature if temperature
      parameters[:top_p] = top_p if top_p
      parameters[:tools] = tools if tools.any?
      parameters[:tool_choice] = tool_choice if tool_choice
      parameters[:user] = user if user

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
        client.chat(parameters: parameters)
      end

      response = response_from_chunks if block
      reset_response_chunks

      Langchain::LLM::OpenAIResponse.new(response)
    end

    # Generate a summary for a given text
    #
    # @param text [String] The text to generate a summary for
    # @return [String] The summary
    def summarize(text:)
      prompt_template = Langchain::Prompt.load_from_path(
        file_path: Langchain.root.join("langchain/llm/prompts/summarize_template.yaml")
      )
      prompt = prompt_template.format(text: text)

      complete(prompt: prompt)
    end

    def default_dimension
      @defaults[:dimension] || EMBEDDING_SIZES.fetch(defaults[:embeddings_model_name].to_sym)
    end

    private

    attr_reader :response_chunks

    def reset_response_chunks
      @response_chunks = []
    end

    def with_api_error_handling
      response = yield
      return if response.empty?

      raise Langchain::LLM::ApiError.new "OpenAI API error: #{response.dig("error", "message")}" if response&.dig("error")

      response
    end

    def validate_max_tokens(messages, model, max_tokens = nil)
      LENGTH_VALIDATOR.validate_max_tokens!(
        messages, model,
        max_tokens: max_tokens, llm: self,
        token_counter: defaults[:token_counter]
      )
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
