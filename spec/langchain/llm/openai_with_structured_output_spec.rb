# frozen_string_literal: true

require "spec_helper"

RSpec.describe Langchain::LLM::OpenAI, "with structured output" do
  let(:client) { described_class.new(api_key: ENV.fetch("OPENAI_API_KEY")) }

  let(:user_details) do
    Class.new do
      include EasyTalk::Model
      def self.name
        "UserDetails"
      end

      define_schema do
        title "User Details"
        property :name, String
        property :age, Integer
      end
    end
  end

  it "returns structured output", :vcr do
    response = client.chat(
      messages: [
        role: "user",
        content: "Extract James is 25 years old"
      ],
      tools: [EasyTalk::Tools::FunctionBuilder.new(user_details)]
    )

    result = response.by_function_name(user_details.function_name)
    parsed_result = JSON.parse(result)

    expect(user_details.validate_json(parsed_result)).to eq(true)
    expect(parsed_result).to eq("name" => "James", "age" => 25)
  end
end
