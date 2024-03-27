# frozen_string_literal: true

RSpec.describe Langchain::Data do
  let(:source) { "spec/fixtures/loaders/example.txt" }
  let(:data) { File.read(source) }

  subject { described_class.new(data, source: source) }

  describe "#chunks" do
    it "returns an array of chunks" do
      chunks = subject.chunks
      split_data = data.split("\n\n")

      expect(chunks).to all(be_a(Langchain::Chunk))
      expect(chunks[0].text).to eq(split_data[0])
      expect(chunks[1].text).to eq(split_data[1])
      expect(chunks[2].text).to eq(split_data[2])

      expect(chunks[0].source).to eq(source)
      expect(chunks[1].source).to eq(source)
      expect(chunks[2].source).to eq(source)
    end

    context "with an optional chunker class" do
      subject { described_class.new(data, source: source, chunker: Langchain::Chunker::RecursiveText) }
      let(:chunker) { instance_double(Langchain::Chunker::RecursiveText) }

      before do
        expect(Langchain::Chunker::RecursiveText).to receive(:new).and_return(chunker)
      end

      it "uses an optional chunker class" do
        expect(chunker).to receive(:chunks)
        subject.chunks
      end
    end
  end
end
