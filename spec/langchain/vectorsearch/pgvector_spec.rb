# frozen_string_literal: true

require "pg"

if ENV["POSTGRES_URL"]
  RSpec.describe Langchain::Vectorsearch::Pgvector do
    let(:client) { ::PG.connect(ENV["POSTGRES_URL"]) }

    let(:url) { ENV["POSTGRES_URL"] }
    subject {
      described_class.new(
        url: url,
        api_key: "123",
        index_name: "products",
        llm: Langchain::LLM::OpenAI.new(api_key: "123")
      )
    }

    after { client.exec("TRUNCATE TABLE products;") }

    describe "#create_default_schema" do
      it "creates the default schema" do
        command = subject.create_default_schema
        expect(command).to be_a(::PG::Result)
        expect(command.result_status).to eq(::PG::PGRES_COMMAND_OK)
      end
    end

    describe "#add_texts" do
      before do
        allow_any_instance_of(
          OpenAI::Client
        ).to receive(:embeddings)
          .with(
            parameters: {
              model: "text-embedding-ada-002",
              input: "Hello World"
            }
          )
          .and_return({
            "data" => [
              {"embedding" => 1536.times.map { rand }}
            ]
          })
      end

      it "adds texts" do
        command = subject.add_texts(texts: ["Hello World", "Hello World"])
        expect(command).to be_a(::PG::Result)
        expect(command.result_status).to eq(::PG::PGRES_COMMAND_OK)
        expect(command.cmd_tuples).to eq(2)
      end

      it "adds texts with a namespace" do
        count_query = "SELECT count(*) FROM products WHERE namespace = 'test_namespace';"
        count = client.exec_params(count_query)
        expect(count[0]["count"].to_i).to eq(0)

        allow(subject).to receive(:namespace).and_return("test_namespace")
        command = subject.add_texts(texts: ["Hello World", "Hello World"])
        expect(command).to be_a(::PG::Result)
        expect(command.result_status).to eq(::PG::PGRES_COMMAND_OK)
        expect(command.cmd_tuples).to eq(2)

        count = client.exec_params(count_query)
        expect(count[0]["count"].to_i).to eq(2)
      end
    end

    describe "#similarity_search" do
      let(:fixture) { JSON.parse(File.read("spec/fixtures/vectorsearch/weaviate_search.json")) }

      before do
        allow_any_instance_of(
          OpenAI::Client
        ).to receive(:embeddings)
          .with(
            parameters: {
              model: "text-embedding-ada-002",
              input: "earth"
            }
          )
          .and_return({
            "data" => [
              {"embedding" => 1536.times.map { 0 }}
            ]
          })
      end

      before {
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["something about earth", 1536.times.map { 0 }])
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["Hello World", 1536.times.map { rand }])
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["Hello World", 1536.times.map { rand }])
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["Hello World", 1536.times.map { rand }])
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["Hello World", 1536.times.map { rand }])
      }

      it "searches for similar texts" do
        result = subject.similarity_search(query: "earth")
        expect(result[0]["content"]).to eq("something about earth")
      end

      it "searches for similar texts using a namespace" do
        namespace = "foo_namespace"
        client.exec_params("INSERT INTO products (content, vectors, namespace) VALUES ($1, $2, $3);", ["a namespaced chunk of text", 1536.times.map { 0 }, namespace])
        allow(subject).to receive(:namespace).and_return(namespace)
        result = subject.similarity_search(query: "earth")
        expect(result[0]["content"]).to eq("a namespaced chunk of text")
      end
    end

    describe "#similarity_search_by_vector" do
      before {
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["Some valuable data", 1536.times.map { 0 }])
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["Hello World", 1536.times.map { rand }])
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["Hello World", 1536.times.map { rand }])
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["Hello World", 1536.times.map { rand }])
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", ["Hello World", 1536.times.map { rand }])
      }

      it "searches for similar vectors" do
        result = subject.similarity_search_by_vector(embedding: 1536.times.map { 0 })

        expect(result.length).to eq(4)
        expect(result[0]["content"]).to eq("Some valuable data")
      end

      it "should use the cosine distance operator by default" do
        expect_any_instance_of(PG::Connection).to receive(:prepare) do |_, name, query|
          expect(name).to eq("similarity_search_by_vector")
          expect(query).to include("ORDER BY vectors <=> $1")
          "prepared_statement"
        end
        expect_any_instance_of(PG::Connection).to receive(:exec_prepared) do |_, name, data|
          expect(name).to eq("similarity_search_by_vector")
          []
        end
        subject.similarity_search_by_vector(embedding: 1536.times.map { 0 })
      end
    end

    describe "#ask" do
      let(:question) { "How many times is 'lorem' mentioned in this text?" }
      let(:text) { "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, dignissim sit amet, adipiscing nec, ultricies sed, dolor." }
      let(:prompt) { "Context:\n#{text}\n---\nQuestion: #{question}\n---\nAnswer:" }
      let(:answer) { "5 times" }

      before do
        allow_any_instance_of(
          OpenAI::Client
        ).to receive(:embeddings)
          .with(
            parameters: {
              model: "text-embedding-ada-002",
              input: question
            }
          )
          .and_return({
            "data" => [
              {"embedding" => 1536.times.map { 0 }}
            ]
          })
      end

      before do
        client.exec_params("INSERT INTO products (content, vectors) VALUES ($1, $2);", [text, 1536.times.map { 0 }])
      end

      before do
        allow(subject.llm).to receive(:chat).with(prompt: prompt).and_return(answer)
      end

      it "asks a question" do
        expect(subject.ask(question: question)).to eq(answer)
      end
    end
  end
end
