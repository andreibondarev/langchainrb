# frozen_string_literal: true

require "epsilla"

if ENV["EPSILLA_URL"]
  RSpec.describe Langchain::Vectorsearch::Epsilla do
    let(:index_name) { "documents" }

    subject {
      described_class.new(
        url: ENV["EPSILLA_URL"],
        index_name: index_name,
        db_name: "langchainrb_tests",
        db_path: "/tmp/langchainrb_tests",
        llm: Langchain::LLM::OpenAI.new(api_key: "123")
      )
    }

    describe "#create_default_schema" do
      before do
        # clean up
        subject.destroy_default_schema
      rescue
        # do nothing
      end

      it "returns 200 OK" do
        expect(subject.create_default_schema["statusCode"]).to eq(200)
      end
    end

    describe "#destroy_default_schema" do
      before do
        subject.create_default_schema
      rescue
        # do nothing
      end

      it "returns 200 OK" do
        expect(subject.destroy_default_schema["statusCode"]).to eq(200)
      end
    end

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

        begin
          # make sure table is created
          subject.create_default_schema
        rescue
          # do nothing
        end
      end

      it "adds texts" do
        result = subject.add_texts(texts: ["Hello World", "Hello World"])
        expect(result["statusCode"]).to eq(200)
      end

      it "adds texts with IDs" do
        result = subject.add_texts(texts: ["Hello World", "Hello World"], ids: [100, 101])
        expect(result["statusCode"]).to eq(200)
      end

      it "raises when text and ids have different lengths" do
        expect { subject.add_texts(texts: ["Hello World", "Hello World"], ids: [100]) }.to raise_error("The number of ids must match the number of texts")
      end

      after do
        subject.destroy_default_schema
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

        allow_any_instance_of(
          OpenAI::Client
        ).to receive(:embeddings)
          .with(
            parameters: {
              dimensions: 1536,
              model: "text-embedding-3-small",
              input: "something about earth"
            }
          )
          .and_return({
            "object" => "list",
            "data" => [
              {"embedding" => 1536.times.map { 0 }}
            ]
          })

        4.times do |i|
          allow_any_instance_of(
            OpenAI::Client
          ).to receive(:embeddings)
            .with(
              parameters: {
                dimensions: 1536,
                model: "text-embedding-3-small",
                input: "Hello World #{i}"
              }
            )
            .and_return({
              "object" => "list",
              "data" => [
                {"embedding" => 1536.times.map { rand }}
              ]
            })
        end

        begin
          # make sure table is created
          subject.create_default_schema
        rescue
          # do nothing
        end

        subject.add_texts(texts: ["something about earth"])
        subject.add_texts(texts: [0, 1, 2, 3].map { |i| "Hello World #{i}" })
      end

      it "searches for similar texts" do
        result = subject.similarity_search(query: "earth")

        expect(result.first).to eq("something about earth")
      end

      it "searches for similar vectors" do
        result = subject.similarity_search_by_vector(embedding: 1536.times.map { 0 })

        expect(result.count).to eq(4)
        expect(result.first).to eq("something about earth")
      end

      after do
        subject.destroy_default_schema
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
        allow_any_instance_of(
          OpenAI::Client
        ).to receive(:embeddings)
          .with(
            parameters: {
              dimensions: 1536,
              model: "text-embedding-3-small",
              input: text
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
        begin
          # make sure table is created
          subject.create_default_schema
        rescue
          # do nothing
        end

        subject.add_texts(texts: [text])
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

      after do
        subject.destroy_default_schema
      end
    end
  end
end
