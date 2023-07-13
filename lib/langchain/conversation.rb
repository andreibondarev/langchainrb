# frozen_string_literal: true

module Langchain
  #
  # A high-level API for running a conversation with an LLM.
  # Currently supports: OpenAI and Google PaLM LLMs.
  #
  # Usage:
  #     llm = Langchain::LLM::OpenAI.new(api_key: "YOUR_API_KEY")
  #     chat = Langchain::Conversation.new(llm: llm)
  #     chat.set_context("You are a chatbot from the future")
  #     chat.message("Tell me about future technologies")
  #
  # To stream the chat response:
  #     chat = Langchain::Conversation.new(llm: llm) do |chunk|
  #       print(chunk)
  #     end
  #
  class Conversation
    attr_reader :options

    # Intialize Conversation with a LLM
    #
    # @param llm [Object] The LLM to use for the conversation
    # @param options [Hash] Options to pass to the LLM, like temperature, top_k, etc.
    # @return [Langchain::Conversation] The Langchain::Conversation instance
    def initialize(llm:, **options, &block)
      @llm = llm
      @context = nil
      @examples = []
      @memory = ConversationMemory.new(
        llm: llm,
        messages: options.delete(:messages) || [],
        strategy: options.delete(:memory_strategy)
      )
      @options = options
      @block = block
    end

    def set_functions(functions)
      @llm.functions = functions
      @llm.complete_response = true
    end

    # Set the context of the conversation. Usually used to set the model's persona.
    # @param message [String] The context of the conversation
    def set_context(message)
      @memory.set_context message
    end

    # Add examples to the conversation. Used to give the model a sense of the conversation.
    # @param examples [Array<Hash>] The examples to add to the conversation
    def add_examples(examples)
      @memory.add_examples examples
    end

    # Message the model with a prompt and return the response.
    # @param message [String] The prompt to message the model with
    # @return [String] The response from the model
    def message(message)
      @memory.append_user_message(message)
      response = llm_response(message)
      @memory.append_ai_message(response)
      response
    end

    # Messages from conversation memory
    # @return [Array<Hash>] The messages from the conversation memory
    def messages
      @memory.messages
    end

    # Context from conversation memory
    # @return [String] Context from conversation memory
    def context
      @memory.context
    end

    # Examples from conversation memory
    # @return [Array<Hash>] Examples from the conversation memory
    def examples
      @memory.examples
    end

    private

    def llm_response(prompt)
      @llm.chat(messages: @memory.messages, context: @memory.context, examples: @memory.examples, **@options, &@block)
    rescue Langchain::Utils::TokenLength::TokenLimitExceeded => exception
      @memory.reduce_messages(exception)
      retry
    end
  end
end
