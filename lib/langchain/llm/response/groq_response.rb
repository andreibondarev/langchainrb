# frozen_string_literal: true

module Langchain::LLM
  class GroqResponse < BaseResponse
    def chat_completions
      raw_response.dig("content")
    end
  end
end
