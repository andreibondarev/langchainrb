# frozen_string_literal: true

module Langchain::LLM
  #
  # Wrapper around the Google PaLM (Pathways Language Model) APIs: https://ai.google/build/machine-learning/
  #
  # Gem requirements:
  #     gem "google_palm_api", "~> 0.1.0"
  #
  # Usage:
  #     google_palm = Langchain::LLM::GooglePalm.new(api_key: "YOUR_API_KEY")
  #
  class GooglePalm < Base
    #
    # Wrapper around the Google PaLM (Pathways Language Model) APIs.
    #
    # Gem requirements: gem "google_palm_api", "~> 0.1.1"
    #
    # Usage:
    # google_palm = Langchain::LLM::GooglePalm.new(api_key: "YOUR_API_KEY")
    #

    DEFAULTS = {
      temperature: 0.0,
      dimension: 768 # This is what the `embedding-gecko-001` model generates
    }.freeze

    def initialize(api_key:)
      depends_on "google_palm_api"
      require "google_palm_api"

      @client = ::GooglePalmApi::Client.new(api_key: api_key)
    end

    #
    # Generate an embedding for a given text
    #
    # @param text [String] The text to generate an embedding for
    # @return [Array] The embedding
    #
    def embed(text:)
      response = client.embed(
        text: text
      )
      response.dig("embedding", "value")
    end

    #
    # Generate a completion for a given prompt
    #
    # @param prompt [String] The prompt to generate a completion for
    # @param params extra parameters passed to GooglePalmAPI::Client#generate_text
    # @return [String] The completion
    #
    def complete(prompt:, **params)
      default_params = {
        prompt: prompt,
        temperature: DEFAULTS[:temperature]
      }

      if params[:stop_sequences]
        default_params[:stop_sequences] = params.delete(:stop_sequences)
      end

      if params[:max_tokens]
        default_params[:max_output_tokens] = params.delete(:max_tokens)
      end

      default_params.merge!(params)

      response = client.generate_text(**default_params)
      response.dig("candidates", 0, "output")
    end

    #
    # Generate a chat completion for a given prompt
    #
    # @param prompt [String] The prompt to generate a chat completion for
    # @param messages [Array] The messages that have been sent in the conversation
    # @param params extra parameters passed to GooglePalmAPI::Client#generate_chat_message
    # @return [String] The chat completion
    #
    def chat(prompt: "", messages: [], context: "", examples: [], **options)
      raise ArgumentError.new(":prompt or :messages argument is expected") if prompt.empty? && messages.empty?

      default_params = {
        temperature: DEFAULTS[:temperature],
        context: context,
        messages: compose_chat_messages(prompt: prompt, messages: messages),
        examples: compose_examples(examples)
      }

      Langchain::Utils::TokenLength::GooglePalmValidator.validate_max_tokens!(self, default_params[:messages], "chat-bison-001")

      if options[:stop_sequences]
        default_params[:stop] = options.delete(:stop_sequences)
      end

      if options[:max_tokens]
        default_params[:max_output_tokens] = options.delete(:max_tokens)
      end

      default_params.merge!(options)

      response = client.generate_chat_message(**default_params)
      raise "GooglePalm API returned an error: #{response}" if response.dig("error")

      response.dig("candidates", 0, "content")
    end

    #
    # Generate a summarization for a given text
    #
    # @param text [String] The text to generate a summarization for
    # @return [String] The summarization
    #
    def summarize(text:)
      prompt_template = Langchain::Prompt.load_from_path(
        file_path: Langchain.root.join("langchain/llm/prompts/summarize_template.yaml")
      )
      prompt = prompt_template.format(text: text)

      complete(
        prompt: prompt,
        temperature: DEFAULTS[:temperature],
        # Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
        max_tokens: 2048
      )
    end

    private

    def compose_chat_messages(prompt:, messages:)
      history = []
      history.concat transform_messages(messages) unless messages.empty?

      unless prompt.empty?
        if history.last && history.last[:role] == "user"
          history.last[:content] += "\n#{prompt}"
        else
          history.append({author: "user", content: prompt})
        end
      end
      history
    end

    def compose_examples(examples)
      examples.each_slice(2).map do |example|
        {
          input: {content: example.first[:content]},
          output: {content: example.last[:content]}
        }
      end
    end

    def transform_messages(messages)
      messages.map do |message|
        {
          author: message[:role],
          content: message[:content]
        }
      end
    end
  end
end
