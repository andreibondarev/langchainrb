# frozen_string_literal: true

module Langchain::LLM
  # LLM interface for Aws Bedrock APIs: https://docs.aws.amazon.com/bedrock/
  #
  # Gem requirements:
  #    gem 'aws-sdk-bedrockruntime', '~> 1.1'
  #
  # Usage:
  #    bedrock = Langchain::LLM::AwsBedrock.new(llm_options: {})
  #
  class AwsBedrock < Base
    DEFAULTS = {
      completion_model_name: "anthropic.claude-v2",
      max_tokens_to_sample: 300,
      temperature: 1,
      top_k: 250,
      top_p: 0.999,
      stop_sequences: ["\n\nHuman:"],
      anthropic_version: "bedrock-2023-05-31",
      return_likelihoods: "NONE",
      count_penalty: {
        scale: 0,
        apply_to_whitespaces: false,
        apply_to_punctuations: false,
        apply_to_numbers: false,
        apply_to_stopwords: false,
        apply_to_emojis: false
      },
      presence_penalty: {
        scale: 0,
        apply_to_whitespaces: false,
        apply_to_punctuations: false,
        apply_to_numbers: false,
        apply_to_stopwords: false,
        apply_to_emojis: false
      },
      frequency_penalty: {
        scale: 0,
        apply_to_whitespaces: false,
        apply_to_punctuations: false,
        apply_to_numbers: false,
        apply_to_stopwords: false,
        apply_to_emojis: false
      }
    }.freeze

    SUPPORTED_COMPLETION_PROVIDERS = %i[anthropic cohere ai21].freeze

    def initialize(llm_options: {}, default_options: {})
      depends_on "aws-sdk-bedrockruntime", req: "aws-sdk-bedrockruntime"

      @client = ::Aws::BedrockRuntime::Client.new(**llm_options)
      @defaults = DEFAULTS.merge(default_options)
    end

    #
    # Generate a completion for a given prompt
    #
    # @param prompt [String] The prompt to generate a completion for
    # @param params  extra parameters passed to Aws::BedrockRuntime::Client#invoke_model
    # @return [Langchain::LLM::AnthropicResponse], [Langchain::LLM::CohereResponse] or [Langchain::LLM::AI21Response] Response object
    #
    def complete(prompt:, **params)
      raise "Completion provider #{completion_provider} is not supported." unless SUPPORTED_COMPLETION_PROVIDERS.include?(completion_provider)

      parameters = compose_parameters params

      parameters[:prompt] = wrap_prompt prompt
      # TODO Validate Max Tokens
      # parameters[max_tokens_key] = validate_max_tokens(prompt, parameters[:completion_model_name])

      response = client.invoke_model({
        model_id: @defaults[:completion_model_name],
        body: parameters.to_json,
        content_type: "application/json",
        accept: "application/json"
      })

      parse_response response
    end

    private

    def completion_provider
      @defaults[:completion_model_name].split(".").first.to_sym
    end

    def wrap_prompt(prompt)
      if completion_provider == :anthropic
        "\n\nHuman: #{prompt}\n\nAssistant:"
      else
        prompt
      end
    end

    def max_tokens_key
      if completion_provider == :anthropic
        :max_tokens_to_sample
      elsif completion_provider == :cohere
        :max_tokens
      elsif completion_provider == :ai21
        :maxTokens
      end
    end

    def compose_parameters(params)
      if completion_provider == :anthropic
        compose_parameters_anthropic params
      elsif completion_provider == :cohere
        compose_parameters_cohere params
      elsif completion_provider == :ai21
        compose_parameters_ai21 params
      end
    end

    def parse_response(response)
      if completion_provider == :anthropic
        Langchain::LLM::AnthropicResponse.new(JSON.parse(response.body.string))
      elsif completion_provider == :cohere
        Langchain::LLM::CohereResponse.new(JSON.parse(response.body.string))
      elsif completion_provider == :ai21
        Langchain::LLM::AI21Response.new(JSON.parse(response.body.string, symbolize_names: true))
      end
    end

    def validate_max_tokens(prompt, model_name)
      if completion_provider == :anthropic
        # TODO: Implement token length validator for Anthropic
        # Langchain::Utils::TokenLength::AnthropicValidator.validate_max_tokens!(prompt, model_name, client)
      elsif completion_provider == :cohere
        cohere_client = ::Cohere::Client.new("")
        Langchain::Utils::TokenLength::CohereValidator.validate_max_tokens!(prompt, model_name, cohere_client)
      elsif completion_provider == :ai21
        ai21_client = ::AI21::Client.new("")
        Langchain::Utils::TokenLength::AI21Validator.validate_max_tokens!(prompt, model_name, ai21_client)
      else
        raise NotImplementedError, "Completion provider #{completion_provider} is not supported."
      end
    end

    def compose_parameters_cohere(params)
      default_params = @defaults.merge(params)

      {
        max_tokens: default_params[:max_tokens_to_sample],
        temperature: default_params[:temperature],
        p: default_params[:top_p],
        k: default_params[:top_k],
        stop_sequences: default_params[:stop_sequences]
      }
    end

    def compose_parameters_anthropic(params)
      default_params = @defaults.merge(params)

      {
        max_tokens_to_sample: default_params[:max_tokens_to_sample],
        temperature: default_params[:temperature],
        top_k: default_params[:top_k],
        top_p: default_params[:top_p],
        stop_sequences: default_params[:stop_sequences],
        anthropic_version: default_params[:anthropic_version]
      }
    end

    def compose_parameters_ai21(params)
      default_params = @defaults.merge(params)

      {
        maxTokens: default_params[:max_tokens_to_sample],
        temperature: default_params[:temperature],
        topP: default_params[:top_p],
        stopSequences: default_params[:stop_sequences],
        countPenalty: {
          scale: default_params[:count_penalty][:scale],
          applyToWhitespaces: default_params[:count_penalty][:apply_to_whitespaces],
          applyToPunctuations: default_params[:count_penalty][:apply_to_punctuations],
          applyToNumbers: default_params[:count_penalty][:apply_to_numbers],
          applyToStopwords: default_params[:count_penalty][:apply_to_stopwords],
          applyToEmojis: default_params[:count_penalty][:apply_to_emojis]
        },
        presencePenalty: {
          scale: default_params[:presence_penalty][:scale],
          applyToWhitespaces: default_params[:presence_penalty][:apply_to_whitespaces],
          applyToPunctuations: default_params[:presence_penalty][:apply_to_punctuations],
          applyToNumbers: default_params[:presence_penalty][:apply_to_numbers],
          applyToStopwords: default_params[:presence_penalty][:apply_to_stopwords],
          applyToEmojis: default_params[:presence_penalty][:apply_to_emojis]
        },
        frequencyPenalty: {
          scale: default_params[:frequency_penalty][:scale],
          applyToWhitespaces: default_params[:frequency_penalty][:apply_to_whitespaces],
          applyToPunctuations: default_params[:frequency_penalty][:apply_to_punctuations],
          applyToNumbers: default_params[:frequency_penalty][:apply_to_numbers],
          applyToStopwords: default_params[:frequency_penalty][:apply_to_stopwords],
          applyToEmojis: default_params[:frequency_penalty][:apply_to_emojis]
        }
      }
    end
  end
end