# frozen_string_literal: true

module Langchain
  # Assistants are Agent-like objects that leverage helpful instructions, LLMs, tools and knowledge to respond to user queries.
  # Assistants can be configured with an LLM of your choice (currently only OpenAI), any vector search database and easily extended with additional tools.
  class Assistant
    attr_reader :llm, :thread, :instructions
    attr_accessor :tools

    SUPPORTED_LLMS = [
      Langchain::LLM::Anthropic,
      Langchain::LLM::OpenAI,
      Langchain::LLM::GoogleGemini,
      Langchain::LLM::GoogleVertexAI
    ]

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
      raise ArgumentError, "Thread must be an instance of Langchain::Thread" unless thread.is_a?(Langchain::Thread)
      raise ArgumentError, "Tools must be an array of Langchain::Tool::Base instance(s)" unless tools.is_a?(Array) && tools.all? { |tool| tool.is_a?(Langchain::Tool::Base) }

      @llm = llm
      @thread = thread
      @tools = tools
      @instructions = instructions

      # The first message in the thread should be the system instructions
      # TODO: What if the user added old messages and the system instructions are already in there? Should this overwrite the existing instructions?
      if llm.is_a?(Langchain::LLM::OpenAI)
        add_message(role: "system", content: instructions) if instructions
      end
      # For Google Gemini, and Anthropic system instructions are added to the `system:` param in the `chat` method
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
      if thread.messages.empty?
        Langchain.logger.warn("No messages in the thread")
        return
      end

      running = true

      while running
        # TODO: I think we need to look at all messages and not just the last one.
        last_message = thread.messages.last

        if last_message.system?
          # Do nothing
          running = false
        elsif last_message.llm?
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
        elsif last_message.user?
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
        elsif last_message.tool?
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
      tool_role = if llm.is_a?(Langchain::LLM::OpenAI)
        Langchain::Messages::OpenAIMessage::TOOL_ROLE
      elsif [Langchain::LLM::GoogleGemini, Langchain::LLM::GoogleVertexAI].include?(llm.class)
        Langchain::Messages::GoogleGeminiMessage::TOOL_ROLE
      elsif llm.is_a?(Langchain::LLM::Anthropic)
        Langchain::Messages::AnthropicMessage::TOOL_ROLE
      end

      # TODO: Validate that `tool_call_id` is valid by scanning messages and checking if this tool call ID was invoked
      add_message(role: tool_role, content: output, tool_call_id: tool_call_id)
    end

    # Delete all messages in the thread
    #
    # @return [Array] Empty messages array
    def clear_thread!
      # TODO: If this a bug? Should we keep the "system" message?
      thread.messages = []
    end

    # Set new instructions
    #
    # @param [String] New instructions that will be set as a system message
    # @return [Array<Langchain::Message>] The messages in the thread
    def instructions=(new_instructions)
      @instructions = new_instructions

      # Find message with role: "system" in thread.messages and delete it from the thread.messages array
      thread.messages.delete_if(&:system?)

      # Set new instructions by adding new system message
      message = build_message(role: "system", content: new_instructions)
      thread.messages.unshift(message)
    end

    private

    # Call to the LLM#chat() method
    #
    # @return [Langchain::LLM::BaseResponse] The LLM response object
    def chat_with_llm
      Langchain.logger.info("Sending a call to #{llm.class}", for: self.class)

      params = {messages: thread.array_of_message_hashes}

      if tools.any?
        if llm.is_a?(Langchain::LLM::OpenAI)
          params[:tools] = tools.map(&:to_openai_tools).flatten
          params[:tool_choice] = "auto"
        elsif llm.is_a?(Langchain::LLM::Anthropic)
          params[:tools] = tools.map(&:to_anthropic_tools).flatten
          params[:system] = instructions if instructions
          params[:tool_choice] = {type: "auto"}
        elsif [Langchain::LLM::GoogleGemini, Langchain::LLM::GoogleVertexAI].include?(llm.class)
          params[:tools] = tools.map(&:to_google_gemini_tools).flatten
          params[:system] = instructions if instructions
          params[:tool_choice] = "auto"
        end
        # TODO: Not sure that tool_choice should always be "auto"; Maybe we can let the user toggle it.
      end

      llm.chat(**params)
    end

    # Run the tools automatically
    #
    # @param tool_calls [Array<Hash>] The tool calls to run
    def run_tools(tool_calls)
      # Iterate over each function invocation and submit tool output
      tool_calls.each do |tool_call|
        tool_call_id, tool_name, method_name, tool_arguments = if llm.is_a?(Langchain::LLM::OpenAI)
          extract_openai_tool_call(tool_call: tool_call)
        elsif [Langchain::LLM::GoogleGemini, Langchain::LLM::GoogleVertexAI].include?(llm.class)
          extract_google_gemini_tool_call(tool_call: tool_call)
        elsif llm.is_a?(Langchain::LLM::Anthropic)
          extract_anthropic_tool_call(tool_call: tool_call)
        end

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

    # Extract the tool call information from the OpenAI tool call hash
    #
    # @param tool_call [Hash] The tool call hash
    # @return [Array] The tool call information
    def extract_openai_tool_call(tool_call:)
      tool_call_id = tool_call.dig("id")

      function_name = tool_call.dig("function", "name")
      tool_name, method_name = function_name.split("__")
      tool_arguments = JSON.parse(tool_call.dig("function", "arguments"), symbolize_names: true)

      [tool_call_id, tool_name, method_name, tool_arguments]
    end

    # Extract the tool call information from the Anthropic tool call hash
    #
    # @param tool_call [Hash] The tool call hash, format: {"type"=>"tool_use", "id"=>"toolu_01TjusbFApEbwKPRWTRwzadR", "name"=>"news_retriever__get_top_headlines", "input"=>{"country"=>"us", "page_size"=>10}}], "stop_reason"=>"tool_use"}
    # @return [Array] The tool call information
    def extract_anthropic_tool_call(tool_call:)
      tool_call_id = tool_call.dig("id")

      function_name = tool_call.dig("name")
      tool_name, method_name = function_name.split("__")
      tool_arguments = tool_call.dig("input").transform_keys(&:to_sym)

      [tool_call_id, tool_name, method_name, tool_arguments]
    end

    # Extract the tool call information from the Google Gemini tool call hash
    #
    # @param tool_call [Hash] The tool call hash, format: {"functionCall"=>{"name"=>"weather__execute", "args"=>{"input"=>"NYC"}}}
    # @return [Array] The tool call information
    def extract_google_gemini_tool_call(tool_call:)
      tool_call_id = tool_call.dig("functionCall", "name")

      function_name = tool_call.dig("functionCall", "name")
      tool_name, method_name = function_name.split("__")
      tool_arguments = tool_call.dig("functionCall", "args").transform_keys(&:to_sym)

      [tool_call_id, tool_name, method_name, tool_arguments]
    end

    # Build a message
    #
    # @param role [String] The role of the message
    # @param content [String] The content of the message
    # @param tool_calls [Array<Hash>] The tool calls to include in the message
    # @param tool_call_id [String] The ID of the tool call to include in the message
    # @return [Langchain::Message] The Message object
    def build_message(role:, content: nil, tool_calls: [], tool_call_id: nil)
      if llm.is_a?(Langchain::LLM::OpenAI)
        Langchain::Messages::OpenAIMessage.new(role: role, content: content, tool_calls: tool_calls, tool_call_id: tool_call_id)
      elsif [Langchain::LLM::GoogleGemini, Langchain::LLM::GoogleVertexAI].include?(llm.class)
        Langchain::Messages::GoogleGeminiMessage.new(role: role, content: content, tool_calls: tool_calls, tool_call_id: tool_call_id)
      elsif llm.is_a?(Langchain::LLM::Anthropic)
        Langchain::Messages::AnthropicMessage.new(role: role, content: content, tool_calls: tool_calls, tool_call_id: tool_call_id)
      end
    end

    # TODO: Fix the message truncation when context window is exceeded
  end
end
