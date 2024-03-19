# frozen_string_literal: true

require "elasticsearch"

RSpec.describe Langchain::Vectorsearch::Elasticsearch do
  let!(:llm) { Langchain::LLM::HuggingFace.new(api_key: "123456") }
  subject {
    Langchain::Vectorsearch::Elasticsearch.new(
      url: "http://localhost:9200",
      index_name: "langchain",
      llm: llm
    )
  }

  before do
    allow(subject.llm).to receive_message_chain(:embed, :embedding).and_return([0.1, 0.2, 0.3])
  end

  describe "#add_texts" do
    it "indexes data into elasticsearch" do
      es_body = [
        {index: {_index: "langchain"}},
        {input: "simple text", input_vector: [0.1, 0.2, 0.3]}
      ]

      allow_any_instance_of(::Elasticsearch::Client).to receive(:bulk).with(body: es_body)
      expect_any_instance_of(::Elasticsearch::Client).to receive(:bulk).with(body: es_body).once

      subject.add_texts(texts: ["simple text"])
    end
  end

  describe "#update_texts" do
    before do
      allow(subject.llm).to receive_message_chain(:embed, :embedding)
        .with(text: "updated text")
        .with(no_args)
        .and_return([0.1, 0.2, 0.3, 0.4])
    end

    it "updates respective document" do
      es_body = [
        {index: {_index: "langchain", _id: 1}},
        {input: "updated text", input_vector: [0.1, 0.2, 0.3, 0.4]}
      ]

      allow_any_instance_of(::Elasticsearch::Client).to receive(:bulk).with(body: es_body)
      expect_any_instance_of(::Elasticsearch::Client).to receive(:bulk).with(body: es_body).once

      subject.update_texts(texts: ["updated text"], ids: [1])
    end
  end

  describe "#remove_texts" do
    it "removes respective document" do
      es_body = [
        {delete: {_index: "langchain", _id: 1}}
      ]

      allow_any_instance_of(::Elasticsearch::Client).to receive(:bulk).with(body: es_body)
      expect_any_instance_of(::Elasticsearch::Client).to receive(:bulk).with(body: es_body).once

      subject.remove_texts(ids: [1])
    end
  end

  describe "#default_vector_settings" do
    it "returns default vector settings" do
      expect(subject.default_vector_settings).to eq({type: "dense_vector", dims: 384})
    end
  end

  describe "#create_default_schema" do
    it "creates default schema" do
      allow_any_instance_of(::Elasticsearch::Client)
        .to receive_message_chain("indices.create").with(index: "langchain", body: subject.default_schema)
      expect_any_instance_of(::Elasticsearch::Client)
        .to receive_message_chain("indices.create").with(index: "langchain", body: subject.default_schema)

      subject.create_default_schema
    end
  end

  describe "#delete_default_schema" do
    it "deletes default schema" do
      allow_any_instance_of(::Elasticsearch::Client)
        .to receive_message_chain("indices.delete").with(index: "langchain")
      expect_any_instance_of(::Elasticsearch::Client)
        .to receive_message_chain("indices.delete").with(index: "langchain")

      subject.delete_default_schema
    end
  end

  describe "#default_schema" do
    it "returns default schema" do
      schema = {
        mappings: {
          properties: {
            input: {
              type: "text"
            },
            input_vector: {type: "dense_vector", dims: 384}
          }
        }
      }

      expect(subject.default_schema).to eq(schema)
    end

    it "override default vector settings" do
      subject.options[:vector_settings] = {type: "dense_vector", dims: 500}

      schema = {
        mappings: {
          properties: {
            input: {
              type: "text"
            },
            input_vector: {type: "dense_vector", dims: 500}
          }
        }
      }

      expect(subject.default_schema).to eq(schema)
    end
  end

  describe "#default_query" do
    it "returns cosineSimilarity as default query" do
      query = {
        script_score: {
          query: {match_all: {}},
          script: {
            source: "cosineSimilarity(params.query_vector, 'input_vector') + 1.0",
            params: {
              query_vector: [0.1, 0.2, 0.3]
            }
          }
        }
      }

      expect(subject.default_query([0.1, 0.2, 0.3])).to eq(query)
    end
  end

  describe "#similarity_search" do
    it "should return similar documents" do
      response = [
        {_id: 1, input: "simple text", input_vector: [0.1, 0.5, 0.6]},
        {_id: 2, input: "update text", input_vector: [0.5, 0.3, 0.1]}
      ]
      es_response = double("Elasticsearch::API::Response")

      allow(es_response).to receive(:body).and_return(response)
      allow_any_instance_of(::Elasticsearch::Client)
        .to receive(:search).with(body: {query: subject.default_query([0.1, 0.2, 0.3]), size: 5}).and_return(es_response)

      expect_any_instance_of(::Elasticsearch::Client)
        .to receive(:search).with(body: {query: subject.default_query([0.1, 0.2, 0.3]), size: 5})
      expect(es_response).to receive(:body)

      expect(subject.similarity_search(text: "simple", k: 5)).to eq(response)
    end

    it "able to search with custom query" do
      es_response = double("Elasticsearch::API::Response")
      response = [
        {_id: 1, input: "simple text", input_vector: [0.1, 0.5, 0.6]},
        {_id: 2, input: "update text", input_vector: [0.5, 0.3, 0.1]}
      ]
      custom_query = {
        script_score: {
          query: {match_all: {}},
          script: {
            source: "cosineSimilarity(params.query_vector, 'input_vector') + 2.0",
            params: {
              query_vector: [0.1, 0.2, 0.3]
            }
          }
        }
      }

      allow(es_response).to receive(:body).and_return(response)
      allow_any_instance_of(::Elasticsearch::Client)
        .to receive(:search).with(body: {query: custom_query, size: 10}).and_return(es_response)

      expect_any_instance_of(::Elasticsearch::Client)
        .to receive(:search).with(body: {query: custom_query, size: 10})
      expect(es_response).to receive(:body)
      expect(subject.similarity_search(query: custom_query)).to eq(response)
    end

    it "either text or query parameter is mandatory" do
      expect { subject.similarity_search }.to raise_error("Either text or query should pass as an argument")
    end
  end

  describe "#similarity_search_by_vector" do
    it "should return similar documents" do
      response = [
        {_id: 1, input: "simple text", input_vector: [0.1, 0.5, 0.6]},
        {_id: 2, input: "update text", input_vector: [0.5, 0.3, 0.1]}
      ]
      es_response = double("Elasticsearch::API::Response")

      allow(es_response).to receive(:body).and_return(response)
      allow_any_instance_of(::Elasticsearch::Client)
        .to receive(:search).with(body: {query: subject.default_query([0.5, 0.6, 0.7]), size: 5}).and_return(es_response)

      expect_any_instance_of(::Elasticsearch::Client)
        .to receive(:search).with(body: {query: subject.default_query([0.5, 0.6, 0.7]), size: 5})
      expect(es_response).to receive(:body)

      expect(subject.similarity_search_by_vector(embedding: [0.5, 0.6, 0.7], k: 5)).to eq(response)
    end

    it "able to search with custom query" do
      es_response = double("Elasticsearch::API::Response")
      response = [
        {_id: 1, input: "simple text", input_vector: [0.1, 0.5, 0.6]},
        {_id: 2, input: "update text", input_vector: [0.5, 0.3, 0.1]}
      ]
      custom_query = {
        script_score: {
          query: {match_all: {}},
          script: {
            source: "cosineSimilarity(params.query_vector, 'input_vector') + 2.0",
            params: {
              query_vector: [0.5, 0.6, 0.7]
            }
          }
        }
      }

      allow(es_response).to receive(:body).and_return(response)
      allow_any_instance_of(::Elasticsearch::Client)
        .to receive(:search).with(body: {query: custom_query, size: 10}).and_return(es_response)

      expect_any_instance_of(::Elasticsearch::Client)
        .to receive(:search).with(body: {query: custom_query, size: 10})
      expect(es_response).to receive(:body)
      expect(subject.similarity_search_by_vector(query: custom_query)).to eq(response)
    end

    it "either text or query parameter is mandatory" do
      expect { subject.similarity_search_by_vector }.to raise_error("Either embedding or query should pass as an argument")
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
      response = [
        {_id: 1, input: text, input_vector: [0.1, 0.5, 0.6]}
      ]
      allow(subject).to receive(:similarity_search).with(query: question, k: 4).and_return(response)
    end

    context "without block" do
      before do
        allow(subject.llm).to receive(:chat).with(messages: messages).and_return(response)
        expect(response).to receive(:context=).with(text)
      end

      it "asks a question and returns the answer" do
        expect(subject.ask(question: question, k: 4).completion).to eq(answer)
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
