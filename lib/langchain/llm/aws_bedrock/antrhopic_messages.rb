# frozen_string_literal: true

module Langchain::LLM
  # LLM interface for Aws Bedrock Claude Messaging APIs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
  #
  # Gem requirements:
  #    gem 'aws-sdk-bedrockruntime', '~> 1.1'
  #
  # Usage:
  #    claude3 = Langchain::LLM::AwsBedrock::AnthropicMessages.new(llm_options: {})
  #
  class AwsBedrock::AnthropicMessages < AwsBedrock
    DEFAULTS = {
      completion_model_name: "anthropic.claude-3-sonnet-20240229-v1:0",
      embedding_model_name: "",
      max_tokens: 300,
      temperature: 1,
      top_k: 250,
      top_p: 0.999,
      stop_sequences: ["AI"],
      anthropic_version: "bedrock-2023-05-31"
    }.freeze

    def initialize(completion_model: DEFAULTS[:completion_model_name], embedding_model: DEFAULTS[:embedding_model_name], aws_client_options: {}, default_options: {})
      depends_on "aws-sdk-bedrockruntime", req: "aws-sdk-bedrockruntime"

      @client = ::Aws::BedrockRuntime::Client.new(**aws_client_options)
      @defaults = DEFAULTS.merge(default_options)
        .merge(completion_model_name: completion_model)
        .merge(embedding_model_name: embedding_model)
    end

    def chat(system: nil, messages: [], **params)
      raise ArgumentError.new("messages argument is required") if messages.empty?

      parameters = compose_parameters_anthropic(params)
      puts("parameters: #{parameters}")
      parameters[:system] = system if system
      parameters[:messages] = messages

      response = client.invoke_model({
        model_id: @defaults[:completion_model_name],
        body: parameters.to_json,
        content_type: "application/json",
        accept: "application/json"
      })

      AnthropicMessagesResponse.new(JSON.parse(response.body.string))
    end

    def compose_parameters_anthropic(params)
      default_params = @defaults.merge(params)

      {
        max_tokens: default_params[:max_tokens
        ],
        temperature: default_params[:temperature],
        top_k: default_params[:top_k],
        top_p: default_params[:top_p],
        stop_sequences: default_params[:stop_sequences],
        anthropic_version: default_params[:anthropic_version]
      }
    end
  end
end
