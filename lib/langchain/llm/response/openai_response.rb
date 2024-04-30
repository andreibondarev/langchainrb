# frozen_string_literal: true

module Langchain::LLM
  class OpenAIResponse < BaseResponse
    def model
      raw_response["model"]
    end

    def created_at
      if raw_response.dig("created")
        Time.at(raw_response.dig("created"))
      end
    end

    def completion
      completions&.dig(0, "message", "content")
    end

    def role
      completions&.dig(0, "message", "role")
    end

    def chat_completion
      completion
    end

    def tool_calls
      chat_completions&.dig(0, "message", "tool_calls")
    end

    def embedding
      embeddings&.first
    end

    def completions
      raw_response.dig("choices")
    end

    def chat_completions
      raw_response.dig("choices")
    end

    def embeddings
      raw_response.dig("data")&.map { |datum| datum.dig("embedding") }
    end

    def prompt_tokens
      raw_response.dig("usage", "prompt_tokens")
    end

    def completion_tokens
      raw_response.dig("usage", "completion_tokens")
    end

    def total_tokens
      raw_response.dig("usage", "total_tokens")
    end

    def function_responses
      tool_calls&.map { |tool_call| tool_call.dig("function") }
    end

    def by_function_name(function_name)
      function_responses&.find { |res| res.dig("name") == function_name }&.dig("arguments")
    end
  end
end
