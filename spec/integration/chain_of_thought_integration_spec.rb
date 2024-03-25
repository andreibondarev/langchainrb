# frozen_string_literal: true

# execute with command:
#   INTEGRATION_TESTS_ENABLED=true bundle exec rspec spec/integration/chain_of_thought_integration_spec.rb
#
# requires the following environment variables:
#   SERPAPI_API_KEY
#   OPENAI_API_KEY

RSpec.describe "Chain of Thought integration with other components", type: :integration do
  it "Should run with search and calculator" do
    question = "How many full soccer fields would be needed to cover the distance between NYC and DC in a straight line?"

    search_tool = Langchain::Tool::SerpApi.new(api_key: ENV["SERPAPI_API_KEY"])
    calculator_tool = Langchain::Tool::Calculator.new

    openai = Langchain::LLM::OpenAI.new(api_key: ENV["OPENAI_API_KEY"])

    agent = Langchain::Agent::ChainOfThoughtAgent.new(
      llm: openai,
      tools: [search_tool, calculator_tool]
    )
    result = agent.run(question: question)

    expect(result).to include("distance between NYC and DC")
  end
end
