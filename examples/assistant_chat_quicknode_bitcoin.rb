require "dotenv/load"
require "langchain"
require "openai"
require "reline"

# gem install reline
# or add `gem "reline"` to your Gemfile

openai = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"], default_options: {
  # chat_completion_model_name: "gpt-4-turbo"
})
thread = Langchain::Thread.new
assistant = Langchain::Assistant.new(
  llm: openai,
  thread: thread,
  instructions: "You are a Bitcoin Blockchain Analyst specializes in analyzing and interpreting blockchain data",
  tools: [
    Langchain::Tool::QuicknodeBitcoin.new(url: ENV["QUICKNODE_BITCOIN_URL"])
  ]
)

DONE = %w[done end eof exit].freeze

puts "Welcome to your Bitcoin Blockchain Analyst assistant!"
puts "-" * 20
puts "(multiline input; type 'end' on its own line when done. or exit to exit)"

def prompt_for_message
  user_message = Reline.readmultiline("Question: ", true) do |multiline_input|
    last = multiline_input.split.last
    DONE.include?(last)
  end

  return :noop unless user_message

  lines = user_message.split("\n")
  if lines.size > 1 && DONE.include?(lines.last)
    # remove the "done" from the message
    user_message = lines[0..-2].join("\n")
  end

  return :exit if DONE.include?(user_message.downcase)

  user_message
end

begin
  loop do
    user_message = prompt_for_message

    case user_message
    when :noop
      next
    when :exit
      break
    end

    assistant.add_message_and_run content: user_message, auto_tool_execution: true
    puts "\n"
    puts assistant.thread.messages.last.content
    puts "-" * 5
    puts "\n"
  end
rescue Interrupt
  exit 0
end
