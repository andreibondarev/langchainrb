# frozen_string_literal: true

require "pg"

if ENV["POSTGRES_URL"]
  client = ::PG.connect(ENV["POSTGRES_URL"])

  subject = Langchain::Vectorsearch::Pgvector.new(
    url: ENV["POSTGRES_URL"],
    index_name: "products",
    llm: Langchain::LLM::OpenAI.new(api_key: "123")
  )
  subject.create_default_schema

  RSpec.describe Langchain::Vectorsearch::Pgvector do
    let(:client) { client }

    subject {
      subject
    }

    after { client.exec("TRUNCATE TABLE products;") }

    describe "#add_texts" do
      before do
        allow_any_instance_of(
          OpenAI::Client
        ).to receive(:embeddings)
          .with(
            parameters: {
              dimensions: 1536,
              model: "text-embedding-3-small",
              input: "Hello World"
            }
          )
          .and_return({
            "object" => "list",
            "data" => [
              {"embedding" => 1536.times.map { rand }}
            ]
          })
      end

      it "adds texts" do
        result = subject.add_texts(texts: ["Hello World", "Hello World"])
        expect(result.size).to eq(2)
      end
    end

    describe "#update_texts" do
      let(:text_embedding_mapping) do
        {
          "Hello World" => 1536.times.map { rand },
          "Hello World".reverse => 1536.times.map { rand }
        }
      end

      before do
        text_embedding_mapping.each do |input, embedding|
          allow_any_instance_of(
            OpenAI::Client
          ).to receive(:embeddings)
            .with(
              parameters: {
                dimensions: 1536,
                model: "text-embedding-3-small",
                input: input
              }
            )
            .and_return({
              "object" => "list",
              "data" => [
                {"embedding" => embedding}
              ]
            })
        end
      end

      it "updates texts" do
        values = subject.add_texts(texts: ["Hello World", "Hello World"])
        ids = values.flatten
        result = subject.update_texts(texts: ["Hello World", "Hello World".reverse], ids: ids)

        expect(result.size).to eq(2)
      end

      it "adds texts with a namespace" do
        count_query = "SELECT count(*) FROM products WHERE namespace = 'test_namespace';"
        count = client.exec_params(count_query)
        expect(count[0]["count"].to_i).to eq(0)

        allow(subject).to receive(:namespace).and_return("test_namespace")
        ids = subject.add_texts(texts: ["Hello World", "Hello World"])
        expect(ids.length).to eq(2)

        count = client.exec_params(count_query)
        expect(count[0]["count"].to_i).to eq(2)
      end
    end

    describe "#remove_texts" do
      before do
        allow_any_instance_of(
          OpenAI::Client
        ).to receive(:embeddings)
          .with(
            parameters: {
              dimensions: 1536,
              model: "text-embedding-3-small",
              input: "Hello World"
            }
          )
          .and_return({
            "object" => "list",
            "data" => [
              {"embedding" => 1536.times.map { rand }}
            ]
          })
      end

      it "removes texts" do
        values = subject.add_texts(texts: ["Hello World", "Hello World"])
        ids = values.flatten
        expect(ids.length).to eq(2)

        result = subject.remove_texts(ids: ids)
        expect(result).to eq(2)
      end
    end

    describe "#similarity_search" do
      before do
        allow_any_instance_of(
          OpenAI::Client
        ).to receive(:embeddings)
          .with(
            parameters: {
              dimensions: 1536,
              model: "text-embedding-3-small",
              input: "earth"
            }
          )
          .and_return({
            "object" => "list",
            "data" => [
              {"embedding" => 1536.times.map { 0 }}
            ]
          })
      end

      before {
        subject.documents_model.new(content: "something about earth", vectors: 1536.times.map { 0 }).save
        4.times do |i|
          subject.documents_model.new(content: "Hello World", vectors: 1536.times.map { rand }).save
        end
      }

      it "searches for similar texts" do
        result = subject.similarity_search(query: "earth")

        expect(result.first.content).to eq("something about earth")
      end

      it "searches for similar texts using a namespace" do
        namespace = "foo_namespace"

        subject.documents_model.new(content: "a namespaced chunk of text", vectors: 1536.times.map { 0 }, namespace: namespace).save

        allow(subject).to receive(:namespace).and_return(namespace)
        result = subject.similarity_search(query: "earth")
        expect(result.first.content).to eq("a namespaced chunk of text")
      end
    end

    describe "#similarity_search_by_vector" do
      before {
        subject.documents_model.new(content: "Some valuable data", vectors: 1536.times.map { 0 }).save
        4.times do |i|
          subject.documents_model.new(content: "Hello World", vectors: 1536.times.map { rand }).save
        end
      }

      it "searches for similar vectors" do
        result = subject.similarity_search_by_vector(embedding: 1536.times.map { 0 })

        expect(result.count).to eq(4)
        expect(result.first.content).to eq("Some valuable data")
      end

      it "should use the cosine distance operator by default" do
        expect(subject.operator).to eq("cosine")
      end
    end

    describe "#ask" do
      let(:question) { "How many times is 'lorem' mentioned in this text?" }
      let(:text) { "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor." }
      let(:messages) { [{role: "user", content: "Context:\n#{text}\n---\nQuestion: #{question}\n---\nAnswer:"}] }
      let(:response) { double(completion: answer) }
      let(:answer) { "5 times" }
      let(:k) { 4 }

      before do
        allow_any_instance_of(
          OpenAI::Client
        ).to receive(:embeddings)
          .with(
            parameters: {
              dimensions: 1536,
              model: "text-embedding-3-small",
              input: question
            }
          )
          .and_return({
            "object" => "list",
            "data" => [
              {"embedding" => 1536.times.map { 0 }}
            ]
          })
      end

      before do
        subject.documents_model.new(content: text, vectors: 1536.times.map { 0 }).save
      end

      context "without block" do
        before do
          allow(subject.llm).to receive(:chat).with(messages: messages).and_return(response)
          expect(response).to receive(:context=).with(text)
        end

        it "asks a question and returns the answer" do
          expect(subject.ask(question: question, k: k).completion).to eq(answer)
        end
      end

      context "with block" do
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
end
