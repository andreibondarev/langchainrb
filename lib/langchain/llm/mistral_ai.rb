# frozen_string_literal: true

module Langchain::LLM
  # Gem requirements:
  #    gem "mistral-ai"
  #
  # Usage:
  #    llm = Langchain::LLM::MistralAI.new(api_key: ENV["MISTRAL_AI_API_KEY"])
  class MistralAI < Base
    DEFAULTS = {
      chat_completion_model_name: "mistral-medium",
      embeddings_model_name: "mistral-embed"
    }.freeze

    attr_reader :defaults

    def initialize(api_key:, default_options: {})
      depends_on "mistral-ai"

      @client = Mistral.new(
        credentials: {api_key: api_key},
        options: {server_sent_events: true}
      )

      @defaults = DEFAULTS.merge(default_options)
    end

    def chat(
      messages:,
      model: defaults[:chat_completion_model_name],
      temperature: nil,
      top_p: nil,
      max_tokens: nil,
      safe_prompt: nil,
      random_seed: nil
    )
      params = {
        messages: messages,
        model: model
      }
      params[:temperature] = temperature if temperature
      params[:top_p] = top_p if top_p
      params[:max_tokens] = max_tokens if max_tokens
      params[:safe_prompt] = safe_prompt if safe_prompt
      params[:random_seed] = random_seed if random_seed

      response = client.chat_completions(params)

      Langchain::LLM::MistralAIResponse.new(response.to_h)
    end

    def embed(
      text:,
      model: defaults[:embeddings_model_name],
      encoding_format: nil
    )
      params = {
        input: text,
        model: model
      }
      params[:encoding_format] = encoding_format if encoding_format

      response = client.embeddings(params)

      Langchain::LLM::MistralAIResponse.new(response.to_h)
    end
  end
end
