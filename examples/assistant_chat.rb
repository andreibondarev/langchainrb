require "dotenv/load"
require "langchain"
require "openai"
require "reline"  # gem install reline

DONE = %w[done end eof print exit].freeze

def prompt_for_message
  puts "(multiline input; type 'end' on its own line when done. or 'print' to print thread, or 'exit' to exit)"

  user_message = Reline.readmultiline("User message: ", true) do |multiline_input|
    last = multiline_input.split.last
    DONE.include?(last)
  end

  return :noop unless user_message
  return :print if user_message == "print"

  lines = user_message.split("\n")

  if lines.size > 1 && DONE.include?(lines.last)
    # remove the "done" from the message
    user_message = lines[0..-2].join("\n")
  end

  return :exit if DONE.include?(user_message.downcase)

  user_message
end

def print_thread(assistant)
  assistant.thread.messages.each do |message|
    puts "----"
    puts message.role + ": " + message.content
    case message.role
    when "assistant"
      message.tool_calls.each do |tool_call|
        puts " " + tool_call["function"]["name"] + ": " << tool_call["function"]["arguments"]
      end
    when "tool"
      puts " " + message.tool_call_id
    end
  end
  puts "---"
end

begin
  assistant = Langchain::Assistant.new(
    llm: Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"]),
    thread: Langchain::Thread.new,
    instructions: "You are a Meteorologist Assistant that is able to pull the weather for any city in imperial units.",
    tools: [
      Langchain::Tool::Weather.new(api_key: ENV["OPEN_WEATHER_API_KEY"])
    ]
  )

  puts "Welcome to your Meteorological assistant!"

  loop do
    user_message = prompt_for_message

    case user_message
    when :noop
      next
    when :print
      print_thread(assistant)
      next
    when :exit
      break
    end

    assistant.add_message_and_run content: user_message, auto_tool_execution: true
    puts assistant.thread.messages.last.content
  end
rescue Interrupt
  exit 0
end
