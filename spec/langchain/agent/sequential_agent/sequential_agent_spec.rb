# frozen_string_literal: true

RSpec.describe Langchain::Agent::SequentialAgent do
  let(:calculator) { Langchain::Tool::Calculator.new }
  let(:search) { Langchain::Tool::GoogleSearch.new(api_key: "123") }
  let(:wikipedia) { Langchain::Tool::Wikipedia.new }

  let(:openai) { Langchain::LLM::OpenAI.new(api_key: "123") }

  let(:question) { "2+2" }
  let(:calculator_response) { "4" }
  let(:search_response) { "4 - Wikipedia" }
  let(:final_answer) { "4 (four) is a number, numeral and digit. It is the natural number following 3 and preceding 5. It is a square number, the smallest semiprime and composite number, and is considered unlucky in many East Asian cultures." }

  subject { described_class.new(llm: openai, tools: [calculator, search, wikipedia]) }

  before do
    allow(subject.tools[0]).to receive(:execute).with(
      input: question
    ).and_return(calculator_response)

    allow(subject.tools[1]).to receive(:execute).with(
      input: calculator_response
    ).and_return(search_response)

    allow(subject.tools[2]).to receive(:execute).with(
      input: search_response
    ).and_return(final_answer)
  end

  describe "#tools" do
    it "sets new tools" do
      expect(subject.tools.count).to eq(3)
      subject.tools = [wikipedia]
      expect(subject.tools.count).to eq(1)
    end
  end

  describe "#run" do
    it "runs the agent tools in sequence" do
      expect(subject.run(question: question)).to eq(final_answer)
    end
  end
end
