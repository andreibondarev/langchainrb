# frozen_string_literal: true

module Langchain::LLM
  class CohereResponse < BaseResponse
    def embedding
      embeddings.first
    end

    def embeddings
      raw_response.dig("embeddings")
    end

    def completions
      raw_response.dig("generations")
    end

    def completion
      completions&.dig(0, "text")
    end

    def tool_calls
    end

    def response_schema
    end
  end
end
