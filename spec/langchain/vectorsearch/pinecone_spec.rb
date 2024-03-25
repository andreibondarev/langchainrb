# frozen_string_literal: true

require "pinecone"

RSpec.describe Langchain::Vectorsearch::Pinecone do
  let(:index_name) { "documents" }
  let(:namespace) { "namespaced" }
  let(:llm) { Langchain::LLM::OpenAI.new(api_key: "123") }

  subject {
    described_class.new(
      environment: "test",
      api_key: "secret",
      index_name: index_name,
      llm: llm
    )
  }

  describe "#create_default_schema" do
    it "returns true" do
      allow(subject.client).to receive(:create_index).with(
        metric: described_class::DEFAULT_METRIC,
        name: index_name,
        dimension: subject.llm.default_dimension
      ).and_return(true)
      expect(subject.create_default_schema).to eq(true)
    end
  end

  describe "#destroy_default_schema" do
    it "returns true" do
      allow(subject.client).to receive(:delete_index).with(index_name).and_return(true)
      expect(subject.destroy_default_schema).to eq(true)
    end
  end

  describe "#get_default_schema" do
    let(:index) { Pinecone::Index.new }

    it "returns true" do
      allow(subject.client).to receive(:index).with(index_name).and_return(index)
      expect(subject.get_default_schema).to eq(index)
    end
  end

  let(:text) { "Hello World" }
  let(:embedding) { [0.1, 0.2, 0.3] }
  let(:query) { "Greetings Earth" }
  let(:k) { 4 }
  let(:metadata) do
    {
      "foo" => "bar",
      "meaningful" => "data"
    }
  end
  let(:filter) do
    {
      foo: {"$eq": "bar"}
    }
  end
  let(:matches) do
    [
      {
        "id" => "example-vector-1",
        "score" => 0.08,
        "values" => embedding,
        "metadata" => metadata
      }
    ]
  end

  let(:results) do
    {
      "matches" => matches,
      "namespace" => ""
    }
  end

  describe "add_texts" do
    let(:vectors) do
      [
        {
          id: "123",
          metadata: {content: text},
          values: embedding
        }
      ]
    end

    before do
      allow(SecureRandom).to receive(:uuid).and_return("123")
      allow(subject.llm).to receive_message_chain(:embed, :embedding).with(text: text).with(no_args).and_return(embedding)
      allow(subject.client).to receive(:index).with(index_name).and_return(Pinecone::Index.new)
    end

    describe "#find" do
      let(:index) { Pinecone::Index.new }

      before(:each) do
        allow(subject.client).to receive(:index).and_return(index)
        allow(subject.client.index).to receive(:fetch).and_return(vectors)
        allow_any_instance_of(Pinecone::Index).to receive(:upsert).with(
          vectors: vectors, namespace: namespace
        ).and_return(true)
      end

      it "returns ArgumentError if no ids supplied" do
        expect { subject.find(ids: []) }.to raise_error(ArgumentError, /Ids must be provided/)
      end

      it "finds vectors with the correct ids and namespace" do
        expect(subject.find(ids: ["123"], namespace: namespace)).to eq(vectors)
      end
    end

    describe "without a namespace" do
      before(:each) do
        allow_any_instance_of(Pinecone::Index).to receive(:upsert).with(
          vectors: vectors, namespace: ""
        ).and_return(true)
      end

      it "adds texts" do
        expect(subject.add_texts(texts: [text])).to eq(true)
      end
    end

    describe "with a namespace" do
      before(:each) do
        allow_any_instance_of(Pinecone::Index).to receive(:upsert).with(
          vectors: vectors, namespace: namespace
        ).and_return(true)
      end

      it "adds texts" do
        expect(subject.add_texts(texts: [text], namespace: namespace)).to eq(true)
      end
    end

    describe "without a namespace" do
      before(:each) do
        allow_any_instance_of(Pinecone::Index).to receive(:upsert).with(
          vectors: vectors, namespace: ""
        ).and_return(true)
      end

      it "adds texts" do
        expect(subject.add_texts(texts: [text])).to eq(true)
      end
    end

    describe "with supplied metadata" do
      let!(:vectors) do
        [
          {
            id: "123",
            metadata: metadata,
            values: embedding
          }
        ]
      end

      before(:each) do
        allow_any_instance_of(Pinecone::Index).to receive(:upsert).with(
          vectors: vectors, namespace: ""
        ).and_return(true)
      end

      it "adds texts" do
        expect(subject.add_texts(texts: [text], metadata: metadata)).to eq(true)
      end
    end

    describe "with ids" do
      let!(:vectors) do
        [
          {
            id: "456",
            metadata: {content: text},
            values: embedding
          }
        ]
      end

      before(:each) do
        allow_any_instance_of(Pinecone::Index).to receive(:upsert).with(
          vectors: vectors, namespace: ""
        ).and_return(true)
      end

      it "adds texts" do
        expect(subject.add_texts(texts: [text], ids: [456])).to eq(true)
      end
    end
  end

  describe "#add_data" do
    it "allows adding multiple paths" do
      paths = [
        Langchain.root.join("../spec/fixtures/loaders/cairo-unicode.pdf"),
        Langchain.root.join("../spec/fixtures/loaders/clearscan-with-image-removed.pdf"),
        Langchain.root.join("../spec/fixtures/loaders/example.txt")
      ]

      expect(subject).to receive(:add_texts).with(texts: array_with_strings_matcher(size: 14), namespace: "")

      subject.add_data(paths: paths)
    end

    it "requires paths" do
      expect { subject.add_data(paths: []) }.to raise_error(ArgumentError, /Paths must be provided/)
    end

    it "allows namespaces" do
      paths = [
        Langchain.root.join("../spec/fixtures/loaders/cairo-unicode.pdf"),
        Langchain.root.join("../spec/fixtures/loaders/clearscan-with-image-removed.pdf"),
        Langchain.root.join("../spec/fixtures/loaders/example.txt")
      ]

      expect(subject).to receive(:add_texts).with(texts: array_with_strings_matcher(size: 14), namespace: "earthlings")

      subject.add_data(paths: paths, namespace: "earthlings")
    end

    context "with an optional chunker class" do
      let(:paths) { Langchain.root.join("../spec/fixtures/loaders/example.txt") }

      it "passes an optional chunker class to Langchain::Loader", :aggregate_failures do
        expect(Langchain::Loader).to receive(:new).with(paths, {}, chunker: Langchain::Chunker::RecursiveText).and_call_original
        expect(subject).to receive(:add_texts).and_return(true)
        subject.add_data(paths: paths, chunker: Langchain::Chunker::RecursiveText)
      end
    end
  end

  describe "#update_texts" do
    let(:vectors) do
      [
        {
          id: "123",
          metadata: {content: text},
          values: embedding
        }
      ]
    end

    before do
      vector = double(Pinecone::Vector)
      allow(subject.llm).to receive_message_chain(:embed, :embedding).with(text: text).with(no_args).and_return(embedding)
      allow(subject.client).to receive(:index).with(index_name).and_return(vector)
      allow(vector).to receive(:update).with(
        values: embedding, id: "123", namespace: "", set_metadata: nil
      ).and_return(true)
    end

    it "updates texts" do
      expect(subject.update_texts(texts: [text], ids: [123], metadata: metadata)).to eq([true])
    end
  end

  describe "#similarity_search_by_vector" do
    before do
      allow(subject.llm).to receive_message_chain(:embed, :embedding).with(text: text).and_return(embedding)
      allow(subject.client).to receive(:index).with(index_name).and_return(Pinecone::Index.new)
    end

    describe "without a namespace" do
      before(:each) do
        allow_any_instance_of(Pinecone::Index).to receive(:query).with(
          vector: embedding,
          namespace: "",
          top_k: k,
          include_values: true,
          include_metadata: true
        ).and_return(results)
      end

      it "searches for similar texts" do
        expect(subject.similarity_search_by_vector(embedding: embedding)).to be_a(Array)
      end
    end

    describe "with a namespace" do
      before(:each) do
        allow_any_instance_of(Pinecone::Index).to receive(:query).with(
          vector: embedding,
          namespace: namespace,
          top_k: k,
          include_values: true,
          include_metadata: true
        ).and_return(results)
      end

      it "searches for similar texts" do
        expect(subject.similarity_search_by_vector(embedding: embedding, namespace: namespace)).to be_a(Array)
      end
    end

    describe "with a filter" do
      before(:each) do
        allow_any_instance_of(Pinecone::Index).to receive(:query).with(
          vector: embedding,
          namespace: "",
          filter: filter,
          top_k: k,
          include_values: true,
          include_metadata: true
        ).and_return(results)
      end

      it "searches for similar texts" do
        expect(subject.similarity_search_by_vector(embedding: embedding, filter: filter)).to be_a(Array)
      end
    end
  end

  describe "#similarity_search" do
    before do
      allow(subject.llm).to receive_message_chain(:embed, :embedding).with(text: query).with(no_args).and_return(embedding)
    end

    describe "without a namespace" do
      before do
        allow(subject).to receive(:similarity_search_by_vector).with(
          embedding: embedding, k: k, namespace: "", filter: nil
        ).and_return(matches)
      end

      it "searches for similar texts" do
        response = subject.similarity_search(query: query, k: k)
        expect(response).to be_a(Array)
        expect(response).to eq(matches)
      end
    end

    describe "with a namespace" do
      before do
        allow(subject).to receive(:similarity_search_by_vector).with(
          embedding: embedding, k: k, namespace: namespace, filter: nil
        ).and_return(matches)
      end

      it "searches for similar texts" do
        response = subject.similarity_search(query: query, k: k, namespace: namespace)
        expect(response).to eq(matches)
      end
    end

    describe "with a filter" do
      before do
        allow(subject).to receive(:similarity_search_by_vector).with(
          embedding: embedding, k: k, namespace: "", filter: filter
        ).and_return(matches)
      end

      it "searches for similar texts" do
        response = subject.similarity_search(query: query, k: k, filter: filter)
        expect(response).to eq(matches)
      end
    end
  end

  describe "#ask" do
    let(:question) { "How many times is \"lorem\" mentioned in this text?" }
    let(:messages) { [{role: "user", content: "Context:\n#{metadata}\n---\nQuestion: #{question}\n---\nAnswer:"}] }
    let(:response) { double(completion: answer) }
    let(:answer) { "5 times" }
    let(:k) { 4 }

    describe "without a namespace" do
      before do
        allow(subject).to receive(:similarity_search).with(
          query: question, namespace: "", filter: nil, k: k
        ).and_return(matches)
        allow(subject.llm).to receive(:chat).with(messages: messages).and_return(response)
        expect(response).to receive(:context=).with(metadata.to_s)
      end

      it "asks a question" do
        expect(subject.ask(question: question).completion).to eq(answer)
      end
    end

    describe "with a namespace" do
      before do
        allow(subject).to receive(:similarity_search).with(
          query: question, namespace: namespace, filter: nil, k: k
        ).and_return(matches)
        allow(subject.llm).to receive(:chat).with(messages: messages).and_return(response)
        expect(response).to receive(:context=).with(metadata.to_s)
      end

      it "asks a question" do
        expect(subject.ask(question: question, namespace: namespace).completion).to eq(answer)
      end
    end

    describe "with a filter" do
      before do
        allow(subject).to receive(:similarity_search).with(
          query: question, namespace: "", filter: filter, k: k
        ).and_return(matches)
        allow(subject.llm).to receive(:chat).with(messages: messages).and_return(response)
        expect(response).to receive(:context=).with(metadata.to_s)
      end

      it "asks a question" do
        expect(subject.ask(question: question, filter: filter).completion).to eq(answer)
      end
    end

    describe "with block" do
      let(:block) { proc { |chunk| puts "Received chunk: #{chunk}" } }

      before do
        allow(subject.llm).to receive(:chat) do |parameters|
          if parameters[:prompt] == prompt && parameters[:stream].is_a?(Proc)
            parameters[:stream].call("Received chunk from llm.chat")
          end
        end
      end

      it "asks a question and yields the chunk to the block" do
        expect do
          captured_output = capture(:stdout) do
            subject.ask(question: question, &block)
          end
          expect(captured_output).to match(/Received chunk from llm.chat/)
        end
      end
    end
  end
end
