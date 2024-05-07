# frozen_string_literal: true

module Langchain::LLM
  class AnthropicResponse < BaseResponse
    def model
      raw_response.dig("model")
    end

    def completion
      completions.first
    end

    def chat_completion
      raw_response.dig("content", 0, "text")
    end

    def tool_calls
      raw_response.dig("content").select { |element| element["type"] == "tool_use" }
    end

    # @return [Hash] JSON schema structured response
    def response_schema
      if tool_calls.any?
        tool_calls.first.dig("input", "properties")
      end
    end

    def completions
      [raw_response.dig("completion")]
    end

    def stop_reason
      raw_response.dig("stop_reason")
    end

    def stop
      raw_response.dig("stop")
    end

    def log_id
      raw_response.dig("log_id")
    end

    def prompt_tokens
      raw_response.dig("usage", "input_tokens").to_i
    end

    def completion_tokens
      raw_response.dig("usage", "output_tokens").to_i
    end

    def total_tokens
      prompt_tokens + completion_tokens
    end

    def role
      raw_response.dig("role")
    end
  end
end
