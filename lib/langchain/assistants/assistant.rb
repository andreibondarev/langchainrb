# frozen_string_literal: true

module Langchain
  # Assistants are Agent-like objects that leverage helpful instructions, LLMs, tools and knowledge to respond to user queries.
  # Assistants can be configured with an LLM of your choice, any vector search database and easily extended with additional tools.
  #
  # Usage:
  #     llm = Langchain::LLM::GoogleGemini.new(api_key: ENV["GOOGLE_GEMINI_API_KEY"])
  #     assistant = Langchain::Assistant.new(
  #       llm: llm,
  #       instructions: "You're a News Reporter AI",
  #       tools: [Langchain::Tool::NewsRetriever.new(api_key: ENV["NEWS_API_KEY"])]
  #     )
  class Assistant
    extend Forwardable
    def_delegators :thread, :messages, :messages=

    attr_reader :llm, :thread, :instructions, :state
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
      thread: nil,
      tools: [],
      instructions: nil,
      &block
    )
      unless SUPPORTED_LLMS.include?(llm.class)
        raise ArgumentError, "Invalid LLM; currently only #{SUPPORTED_LLMS.join(", ")} are supported"
      end
      raise ArgumentError, "Tools must be an array of Langchain::Tool::Base instance(s)" unless tools.is_a?(Array) && tools.all? { |tool| tool.is_a?(Langchain::Tool::Base) }

      @llm = llm
      @thread = thread || Langchain::Thread.new
      @tools = tools
      @instructions = instructions
      @state = :ready
      @block = block

      raise ArgumentError, "Thread must be an instance of Langchain::Thread" unless @thread.is_a?(Langchain::Thread)

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
      messages = thread.add_message(message)
      @state = :ready

      messages
    end

    # Run the assistant
    #
    # @param auto_tool_execution [Boolean] Whether or not to automatically run tools
    # @return [Array<Langchain::Message>] The messages in the thread
    def run(auto_tool_execution: false)
      if thread.messages.empty?
        Langchain.logger.warn("No messages in the thread")
        @state = :completed
        return
      end

      @state = :in_progress
      @state = handle_state until run_finished?(auto_tool_execution)

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
      tool_role = determine_tool_role

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

    # Check if the run is finished
    #
    # @param auto_tool_execution [Boolean] Whether or not to automatically run tools
    # @return [Boolean] Whether the run is finished
    def run_finished?(auto_tool_execution)
      finished_states = [:completed, :failed]

      requires_manual_action = (@state == :requires_action) && !auto_tool_execution
      finished_states.include?(@state) || requires_manual_action
    end

    # Handle the current state and transition to the next state
    #
    # @param state [Symbol] The current state
    # @return [Symbol] The next state
    def handle_state
      case @state
      when :in_progress
        process_latest_message
      when :requires_action
        execute_tools
      end
    end

    # Process the latest message in the thread
    #
    # @return [Symbol] The next state
    def process_latest_message
      last_message = thread.messages.last

      case last_message.standard_role
      when :system
        handle_system_message
      when :llm
        handle_llm_message
      when :user, :tool
        handle_user_or_tool_message
      else
        handle_unexpected_message
      end
    end

    # Handle system message scenario
    #
    # @return [Symbol] The completed state
    def handle_system_message
      Langchain.logger.warn("At least one user message is required after a system message")
      :completed
    end

    # Handle LLM message scenario
    #
    # @param auto_tool_execution [Boolean] Flag to indicate if tools should be executed automatically
    # @return [Symbol] The next state
    def handle_llm_message
      thread.messages.last.tool_calls.any? ? :requires_action : :completed
    end

    # Handle unexpected message scenario
    #
    # @return [Symbol] The failed state
    def handle_unexpected_message
      Langchain.logger.error("Unexpected message role encountered: #{thread.messages.last.standard_role}")
      :failed
    end

    # Handle user or tool message scenario by processing the LLM response
    #
    # @return [Symbol] The next state
    def handle_user_or_tool_message
      response = chat_with_llm
      add_message(role: response.role, content: response.chat_completion, tool_calls: response.tool_calls)

      if response.tool_calls.any?
        :in_progress
      elsif response.chat_completion
        :completed
      else
        Langchain.logger.error("LLM response does not contain tool calls or chat completion")
        :failed
      end
    end

    # Execute the tools based on the tool calls in the last message
    #
    # @return [Symbol] The next state
    def execute_tools
      run_tools(thread.messages.last.tool_calls)
      :in_progress
    rescue => e
      Langchain.logger.error("Error running tools: #{e.message}")
      :failed
    end

    # Determine the tool role based on the LLM type
    #
    # @return [String] The tool role
    def determine_tool_role
      case llm
      when Langchain::LLM::OpenAI
        Langchain::Messages::OpenAIMessage::TOOL_ROLE
      when Langchain::LLM::GoogleGemini, Langchain::LLM::GoogleVertexAI
        Langchain::Messages::GoogleGeminiMessage::TOOL_ROLE
      when Langchain::LLM::Anthropic
        Langchain::Messages::AnthropicMessage::TOOL_ROLE
      end
    end

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

      llm.chat(**params, &@block)
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
