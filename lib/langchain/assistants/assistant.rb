# frozen_string_literal: true

module Langchain
  # Assistants are Agent-like objects that leverage helpful instructions, LLMs, tools and knowledge to respond to user queries.
  # Assistants can be configured with an LLM of your choice (currently only OpenAI), any vector search database and easily extended with additional tools.
  class Assistant
    attr_reader :llm, :thread, :instructions
    attr_accessor :tools

    # Create a new assistant
    #
    # @param llm [Langchain::LLM::Base] LLM instance that the assistant will use
    # @param thread [Langchain::Thread] The thread that'll keep track of the conversation
    # @param tools [Array<Langchain::Tool::Base>] Tools that the assistant has access to
    # @param instructions [String] The system instructions to include in the thread
    def initialize(
      llm:,
      thread:,
      tools: [],
      instructions: nil
    )
      raise ArgumentError, "Invalid LLM; currently only Langchain::LLM::OpenAI and Langchain::LLM::GoogleGemini are supported" unless [Langchain::LLM::OpenAI, Langchain::LLM::GoogleGemini].include?(llm.class) 
      raise ArgumentError, "Thread must be an instance of Langchain::Thread" unless thread.is_a?(Langchain::Thread)
      raise ArgumentError, "Tools must be an array of Langchain::Tool::Base instance(s)" unless tools.is_a?(Array) && tools.all? { |tool| tool.is_a?(Langchain::Tool::Base) }

      @llm = llm
      @thread = thread
      @tools = tools
      @instructions = instructions

      # The first message in the thread should be the system instructions
      # TODO: What if the user added old messages and the system instructions are already in there? Should this overwrite the existing instructions?
      # add_message(role: "system", content: instructions) if instructions

      # TODO: Figure out what to do with Google Gemini
      # add_message(role: "user", content: instructions) if instructions
    end

    # Add a user message to the thread
    #
    # @param content [String] The content of the message
    # @param role [String] The role attribute of the message. Default: "user"
    # @param tool_calls [Array<Hash>] The tool calls to include in the message
    # @param tool_call_id [String] The ID of the tool call to include in the message
    # @return [Array<Langchain::Message>] The messages in the thread
    def add_message(content: nil, role: "user", tool_calls: [], tool_call_id: nil)
      message = build_message(role: role, content: content, tool_calls: tool_calls, tool_call_id: tool_call_id)
      thread.add_message(message)
    end

    # Run the assistant
    #
    # @param auto_tool_execution [Boolean] Whether or not to automatically run tools
    # @return [Array<Langchain::Message>] The messages in the thread
    def run(auto_tool_execution: false)
      running = true

      while running
        # TODO: I think we need to look at all messages and not just the last one.

        # TODO: Fix the following error when thread.messages is empty
        # /Users/andrei/Code/langchain/lib/langchain/assistants/assistant.rb:60:in `run': undefined method `role' for nil (NoMethodError)
        last_message = thread.messages.last

        if last_message.role == "system"
          # Do nothing
          running = false
        elsif ["assistant", "model"].include?(last_message.role)
          if last_message.tool_calls.any?
            if auto_tool_execution
              run_tools(last_message.tool_calls)
            else
              # Maybe log and tell the user that there's outstanding tool calls?
              running = false
            end
          else
            # Last message was from the assistant without any tools calls.
            # Do nothing
            running = false
          end
        elsif last_message.role == "user"
          # Run it!
          response = chat_with_llm

          if response.tool_calls.any?
            # Re-run the while(running) loop to process the tool calls
            running = true
            add_message(role: response.role, tool_calls: response.tool_calls)
          elsif response.chat_completion
            # Stop the while(running) loop and add the assistant's response to the thread
            running = false
            add_message(role: response.role, content: response.chat_completion)
          end
        elsif last_message.role == "tool"
          # Run it!
          response = chat_with_llm
          running = true

          if response.tool_calls.any?
            add_message(role: response.role, tool_calls: response.tool_calls)
          elsif response.chat_completion
            add_message(role: response.role, content: response.chat_completion)
          end
        end
      end

      thread.messages
    end

    # Add a user message to the thread and run the assistant
    #
    # @param content [String] The content of the message
    # @param auto_tool_execution [Boolean] Whether or not to automatically run tools
    # @return [Array<Langchain::Message>] The messages in the thread
    def add_message_and_run(content:, auto_tool_execution: false)
      add_message(content: content, role: "user")
      run(auto_tool_execution: auto_tool_execution)
    end

    # Submit tool output to the thread
    #
    # @param tool_call_id [String] The ID of the tool call to submit output for
    # @param output [String] The output of the tool
    # @return [Array<Langchain::Message>] The messages in the thread
    def submit_tool_output(tool_call_id:, output:)
      # TODO: Validate that `tool_call_id` is valid
      add_message(role: "function", content: output, tool_call_id: tool_call_id)
    end

    private

    # Call to the LLM#chat() method
    #
    # @return [Langchain::LLM::BaseResponse] The LLM response object
    def chat_with_llm
      # params = {messages: thread.openai_messages}
      params = {messages: thread.google_gemini_messages}

      if tools.any?
        # params[:tools] = tools.map(&:to_openai_tools).flatten
        params[:tools] = tools.map(&:to_google_gemini_tools).flatten
        # TODO: Not sure that tool_choice should always be "auto"; Maybe we can let the user toggle it.
        params[:tool_choice] = "auto"
      end

      llm.chat(**params)
    end

    # Run the tools automatically
    #
    # @param tool_calls [Array<Hash>] The tool calls to run
    def run_tools(tool_calls)
      # Iterate over each function invocation and submit tool output
      tool_calls.each do |tool_call|
        # Use the OpenAI "id" OR Google Gemini function name
        tool_call_id = tool_call.dig("id") || tool_call.dig("functionCall", "name")

        # Dig OpenAI format OR Google Gemini format
        function_name = tool_call.dig("function", "name") || tool_call.dig("functionCall", "name")
        # dash or underscore?
        tool_name, method_name = function_name.split("_")

        # tool_arguments = JSON.parse(tool_call.dig("function", "arguments"), symbolize_names: true)
        tool_arguments = tool_call.dig("functionCall", "args").transform_keys(&:to_sym)

        tool_instance = tools.find do |t|
          t.name == tool_name
        end or raise ArgumentError, "Tool not found in assistant.tools"

        output = tool_instance.send(method_name, **tool_arguments)

        submit_tool_output(tool_call_id: tool_call_id, output: output)
      end

      response = chat_with_llm

      if response.tool_calls.any?
        add_message(role: response.role, tool_calls: response.tool_calls)
      elsif response.chat_completion
        add_message(role: response.role, content: response.chat_completion)
      end
    end

    # Build a message
    #
    # @param role [String] The role of the message
    # @param content [String] The content of the message
    # @param tool_calls [Array<Hash>] The tool calls to include in the message
    # @param tool_call_id [String] The ID of the tool call to include in the message
    # @return [Langchain::Message] The Message object
    def build_message(role:, content: nil, tool_calls: [], tool_call_id: nil)
      Message.new(role: role, content: content, tool_calls: tool_calls, tool_call_id: tool_call_id)
    end

    # TODO: Fix the message truncation when context window is exceeded
  end
end
