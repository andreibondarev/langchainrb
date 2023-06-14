# frozen_string_literal: true

RSpec.describe Langchain::Tool::Base do
  describe "#validate_tools!" do
    let(:calculator_tool) { Langchain::Tool::Calculator.new }
    let(:sql_db_tool) { Langchain::Tool::Database.new(connection_string: "mock:///") }
    let(:search_tool) { Langchain::Tool::GoogleSearch.new(api_key: "123") }

    it "does not raise an error" do
      expect {
        described_class.validate_tools!(tools: [calculator_tool, sql_db_tool, search_tool])
      }.not_to raise_error
    end

    it "does raise an error" do
      expect {
        described_class.validate_tools!(tools: [calculator_tool, calculator_tool])
      }.to raise_error(ArgumentError)
    end
  end

  describe "#execute" do
    it "raises an error" do
      expect {
        described_class.execute(input: "input")
      }.to raise_error(NotImplementedError)
    end
  end
end
