# frozen_string_literal: true

RSpec.describe Langchain::Chunker::RecursiveText do
  let(:source) { "spec/fixtures/loaders/example.txt" }
  let(:text) { File.read(source) }

  subject {
    described_class.new(text,
      chunk_size: 1000,
      chunk_overlap: 200,
      separators: ["\n\n"])
  }

  describe "#chunks" do
    it "returns an array of chunks" do
      expect(Baran::RecursiveCharacterTextSplitter).to receive(:new).with(
        chunk_size: 1000,
        chunk_overlap: 200,
        separators: ["\n\n"]
      ).and_call_original

      allow_any_instance_of(Baran::RecursiveCharacterTextSplitter).to receive(:chunks)
        .with(text)
        .and_call_original

      subject.chunks
    end
  end
end
