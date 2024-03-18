# frozen_string_literal: true

module Langchain::LLM
  class AnthropicMessagesResponse < BaseResponse
    def model
      raw_response.dig("model")
    end

    def chat_completions
      raw_response.dig("content")
    end

    def chat_completion
      chat_completions&.dig(0, "text")
    end

    def completion
      completions&.dig(0, "text")
    end

    def completions
      [raw_response.dig("content")]
    end

    def stop_reason
      raw_response.dig("stop_reason")
    end

    def stop
      raw_response.dig("stop_sequence")
    end

    def log_id
      raw_response.dig("id")
    end

    def prompt_tokens
      raw_response.dig("usage", "input_tokens")
    end

    def completion_tokens
      raw_response.dig("usage", "output_tokens")
    end

    def total_tokens
      prompt_tokens + completion_tokens
    end
  end
end
