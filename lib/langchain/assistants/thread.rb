# frozen_string_literal: true

module Langchain
  # Langchain::Thread keeps track of messages in a conversation.
  # TODO: Add functionality to persist to the thread to disk, DB, storage, etc.
  class Thread
    attr_accessor :messages

    # @param messages [Array<Langchain::Message>]
    def initialize(messages: [])
      raise ArgumentError, "messages array must only contain Langchain::Message instance(s)" unless messages.is_a?(Array) && messages.all? { |m| m.is_a?(Langchain::Message) }

      @messages = messages
    end

    # Convert the thread to an OpenAI API-compatible array of hashes
    #
    # @return [Array<Hash>] The thread as an OpenAI API-compatible array of hashes
    def openai_messages
      messages.map(&:to_openai_format)
    end

    # Convert the thread to a Google Gemini API-compatible array of hashes
    # 
    # @return [Array<Hash>] The thread as a Google Gemini API-compatible array of hashes
    def google_gemini_messages
      messages.map(&:to_google_gemini_format)
    end

    # Add a message to the thread
    #
    # @param message [Langchain::Message] The message to add
    # @return [Array<Langchain::Message>] The updated messages array
    def add_message(message)
      raise ArgumentError, "message must be a Langchain::Message instance" unless message.is_a?(Langchain::Message)

      # Prepend the message to the thread
      messages << message
    end
  end
end
