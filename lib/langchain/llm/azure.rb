# frozen_string_literal: true

module Langchain::LLM
  # LLM interface for Azure OpenAI Service APIs: https://learn.microsoft.com/en-us/azure/ai-services/openai/
  #
  # Gem requirements:
  #    gem "ruby-openai", "~> 6.1.0"
  #
  # Usage:
  #    openai = Langchain::LLM::Azure.new(api_key:, llm_options: {}, embedding_deployment_url: chat_deployment_url:)
  #
  class Azure < OpenAI
    attr_reader :embed_client
    attr_reader :chat_client

    def initialize(
      api_key:,
      llm_options: {},
      default_options: {},
      embedding_deployment_url: nil,
      chat_deployment_url: nil
    )
      depends_on "ruby-openai", req: "openai"
      @embed_client = ::OpenAI::Client.new(
        access_token: api_key,
        uri_base: embedding_deployment_url,
        **llm_options
      )
      @chat_client = ::OpenAI::Client.new(
        access_token: api_key,
        uri_base: chat_deployment_url,
        **llm_options
      )
      @defaults = DEFAULTS.merge(default_options)
    end

    #
    # Generate an embedding for a given text
    #
    # @param text [String] The text to generate an embedding for
    # @param params extra parameters passed to OpenAI::Client#embeddings
    # @return [Langchain::LLM::OpenAIResponse] Response object
    #
    def embed(text:, **params)
      parameters = {model: @defaults[:embeddings_model_name], input: text}

      validate_max_tokens(text, parameters[:model])

      response = with_api_error_handling do
        embed_client.embeddings(parameters: parameters.merge(params))
      end

      Langchain::LLM::OpenAIResponse.new(response)
    end

    #
    # Generate a completion for a given prompt
    #
    # @param prompt [String] The prompt to generate a completion for
    # @param params  extra parameters passed to OpenAI::Client#complete
    # @return [Langchain::LLM::Response::OpenaAI] Response object
    #
    def complete(prompt:, **params)
      parameters = compose_parameters @defaults[:completion_model_name], params

      parameters[:messages] = compose_chat_messages(prompt: prompt)
      parameters[:max_tokens] = params[:max_tokens] || validate_max_tokens(parameters[:messages], parameters[:model])

      response = with_api_error_handling do
        chat_client.chat(parameters: parameters)
      end

      Langchain::LLM::OpenAIResponse.new(response)
    end

    #
    # Generate a chat completion for a given prompt or messages.
    #
    # == Examples
    #
    #     # simplest case, just give a prompt
    #     openai.chat prompt: "When was Ruby first released?"
    #
    #     # prompt plus some context about how to respond
    #     openai.chat context: "You are RubyGPT, a helpful chat bot for helping people learn Ruby", prompt: "Does Ruby have a REPL like IPython?"
    #
    #     # full control over messages that get sent, equivilent to the above
    #     openai.chat messages: [
    #       {
    #         role: "system",
    #         content: "You are RubyGPT, a helpful chat bot for helping people learn Ruby", prompt: "Does Ruby have a REPL like IPython?"
    #       },
    #       {
    #         role: "user",
    #         content: "When was Ruby first released?"
    #       }
    #     ]
    #
    #     # few-short prompting with examples
    #     openai.chat prompt: "When was factory_bot released?",
    #       examples: [
    #         {
    #           role: "user",
    #           content: "When was Ruby on Rails released?"
    #         }
    #         {
    #           role: "assistant",
    #           content: "2004"
    #         },
    #       ]
    #
    # @param prompt [String] The prompt to generate a chat completion for
    # @param messages [Array<Hash>] The messages that have been sent in the conversation
    # @param context [String] An initial context to provide as a system message, ie "You are RubyGPT, a helpful chat bot for helping people learn Ruby"
    # @param examples [Array<Hash>] Examples of messages to provide to the model. Useful for Few-Shot Prompting
    # @param options [Hash] extra parameters passed to OpenAI::Client#chat
    # @yield [Hash] Stream responses back one token at a time
    # @return [Langchain::LLM::OpenAIResponse] Response object
    #
    def chat(prompt: "", messages: [], context: "", examples: [], **options, &block)
      raise ArgumentError.new(":prompt or :messages argument is expected") if prompt.empty? && messages.empty?

      parameters = compose_parameters @defaults[:chat_completion_model_name], options, &block
      parameters[:messages] = compose_chat_messages(prompt: prompt, messages: messages, context: context, examples: examples)

      if functions
        parameters[:functions] = functions
      else
        parameters[:max_tokens] = validate_max_tokens(parameters[:messages], parameters[:model])
      end

      response = with_api_error_handling { chat_client.chat(parameters: parameters) }

      return if block

      Langchain::LLM::OpenAIResponse.new(response)
    end
  end
end
