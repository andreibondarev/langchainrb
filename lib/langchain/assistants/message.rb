# frozen_string_literal: true

module Langchain
  # Langchain::Message are the messages that are sent to LLM chat methods
  class Message
    attr_reader :role, :content, :tool_calls, :tool_call_id

    ROLES = [
      # OpenAI uses the following roles:
      "system",
      "assistant",
      "user",
      "tool",
      # Google Gemini uses the following roles:
      "model",
      "function"
    ].freeze

    # @param role [String] The role of the message
    # @param content [String] The content of the message
    # @param tool_calls [Array<Hash>] Tool calls to be made
    # @param tool_call_id [String] The ID of the tool call to be made
    def initialize(role:, content: nil, tool_calls: [], tool_call_id: nil) # TODO: Implement image_file: reference (https://platform.openai.com/docs/api-reference/messages/object#messages/object-content)
      raise ArgumentError, "Role must be one of #{ROLES.join(", ")}" unless ROLES.include?(role)
      raise ArgumentError, "Tool calls must be an array of hashes" unless tool_calls.is_a?(Array) && tool_calls.all? { |tool_call| tool_call.is_a?(Hash) }

      @role = role
      # Some Tools return content as a JSON hence `.to_s`
      @content = content.to_s
      @tool_calls = tool_calls
      @tool_call_id = tool_call_id
    end

    # Convert the message to an OpenAI API-compatible hash
    #
    # @return [Hash] The message as an OpenAI API-compatible hash
    def to_openai_format
      {}.tap do |h|
        h[:role] = role
        h[:content] = content if content # Content is nil for tool calls
        h[:tool_calls] = tool_calls if tool_calls.any?
        h[:tool_call_id] = tool_call_id if tool_call_id
      end
    end

    def to_google_gemini_format
      {}.tap do |h|
        h[:role] = role
        h[:parts] = if function?
          [{
            functionResponse: { 
              name: tool_call_id,
              response: {
                name: tool_call_id,
                content: content
              }
            }
          }]
        elsif tool_calls.any?
          tool_calls
        else
          [{ text: content }]
        end
      end
    end

    # Was this message produced by an LLM?
    def llm?
      model? || assistant?
    end

    # Was this message produced by a model? (This value is specifically used by Google Gemini)
    def model?
      role == "model"
    end

    def assistant?
      role == "assistant"
    end

    def system?
      role == "system"
    end

    def user?
      role == "user"
    end

    def tool?
      role == "tool"
    end

    def function?
      role == "function"
    end
  end
end
