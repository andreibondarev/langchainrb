# frozen_string_literal: true

RSpec.describe Langchain::Chunk do
  let(:text) { "Hello World" }
  let(:source) { "/path/to/file.pdf" }

  subject { described_class.new(text: text, source: source) }

  it "has a text" do
    expect(subject.text).to eq("Hello World")
  end

  it "has a source" do
    expect(subject.source).to eq("/path/to/file.pdf")
  end
end
