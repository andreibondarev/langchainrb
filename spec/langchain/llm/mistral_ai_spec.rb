# frozen_string_literal: true

RSpec.describe Langchain::LLM::MistralAI do
  let(:subject) { described_class.new(api_key: "123") }

  xdescribe "#chat"

  xdescribe "#embed"
end
