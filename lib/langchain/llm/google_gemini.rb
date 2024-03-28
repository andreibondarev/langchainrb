# frozen_string_literal: true

module Langchain::LLM
  # Usage:
  #     google_gemini = Langchain::LLM::GoogleGemini.new(project_id: ENV['GOOGLE_VERTEX_AI_PROJECT_ID'])
  class GoogleGemini < Base

    DEFAULTS = {
      chat_completion_model_name: "gemini-pro",
    }

    attr_reader :defaults

    def initialize(project_id: nil, default_options: {})
      depends_on "gemini-ai"

      @defaults = DEFAULTS.merge(default_options)

      @client = Gemini.new(
        credentials: {
          service: 'vertex-ai-api',
          region: 'us-east4',
          project_id: project_id
        },
        options: { model: defaults[:chat_completion_model_name], server_sent_events: true }
      )
    end

    def chat(
      messages: [],
      tools: [],
      tool_choice: nil # Do we need this param?
    )
      params = {contents: messages}
      params.merge!(tools: { function_declarations: tools }) if tools.any?

      response = client.generate_content(params)

      Langchain::LLM::GoogleGeminiResponse.new(response.to_h, model: @defaults[:chat_completion_model_name])
    end
  end
end
