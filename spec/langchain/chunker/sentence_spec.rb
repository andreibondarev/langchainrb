# frozen_string_literal: true

RSpec.describe Langchain::Chunker::Sentence do
  let(:source) { "spec/fixtures/loaders/the_alchemist.txt" }
  let(:text) { File.read(source) }

  subject { described_class.new(text) }

  describe "#chunks" do
    it "returns an array of chunks" do
      expect(subject.chunks.count).to eq(25)
      expect(subject.chunks).to all(be_a(Langchain::Chunk))
    end

    it "sets source for each chunk" do
      expect(subject.chunks(source: source).map(&:source).uniq).to eq([source])
    end
  end
end
