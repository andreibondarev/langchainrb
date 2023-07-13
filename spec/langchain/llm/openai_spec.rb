# frozen_string_literal: true

RSpec.describe Langchain::LLM::OpenAI do
  let(:subject) { described_class.new(api_key: "123") }

  describe "#initialize" do
    context "when only required options are passed" do
      it "initializes the client without any errors" do
        expect { subject }.not_to raise_error
      end
    end

    context "when llm_options are passed" do
      let(:subject) { described_class.new(api_key: "123", llm_options: {uri_base: "http://localhost:1234"}) }

      it "initializes the client without any errors" do
        expect { subject }.not_to raise_error
      end

      it "passes correct options to the client" do
        # openai-ruby sets global configuration options here: https://github.com/alexrudall/ruby-openai/blob/main/lib/openai/client.rb
        expect(OpenAI.configuration.uri_base).to eq("http://localhost:1234")
      end
    end
  end

  describe "#embed" do
    let(:response) do
      {
        "object" => "list",
        "data" => [
          {
            "object" => "embedding",
            "index" => 0,
            "embedding" => [
              -0.007097351,
              0.0035200312,
              -0.0069700438
            ]
          }
        ]
      }
    end

    before do
      allow(subject.client).to receive(:embeddings).with(parameters).and_return(response)
    end

    context "with default parameters" do
      let(:parameters) do
        {parameters: {input: "Hello World", model: "text-embedding-ada-002"}}
      end
      it "returns an embedding" do
        expect(subject.embed(text: "Hello World")).to eq([-0.007097351, 0.0035200312, -0.0069700438])
      end
    end

    context "with text and  parameters" do
      let(:parameters) do
        {parameters: {input: "Hello World", model: "text-embedding-ada-001", user: "id"}}
      end
      it "returns an embedding" do
        expect(subject.embed(text: "Hello World", model: "text-embedding-ada-001", user: "id")).to eq([-0.007097351, 0.0035200312, -0.0069700438])
      end
    end
  end

  describe "#complete" do
    let(:response) do
      {
        "id" => "cmpl-7BZg4cP5xzga4IyLI6u97WMepAJj2",
        "object" => "text_completion",
        "created" => 1682993108,
        "model" => "text-davinci-003",
        "choices" => [
          {
            "text" => "\n\nThe meaning of life is subjective and can vary from person to person.",
            "index" => 0,
            "logprobs" => nil,
            "finish_reason" => "length"
          }
        ],
        "usage" => {
          "prompt_tokens" => 7,
          "completion_tokens" => 16,
          "total_tokens" => 23
        }
      }
    end

    before do
      allow(subject.client).to receive(:completions).with(parameters).and_return(response)
    end

    context "with default parameters" do
      let(:parameters) do
        {parameters: {model: "text-davinci-003", prompt: "Hello World", temperature: 0.0, max_tokens: 4095}}
      end

      it "returns a completion" do
        expect(subject.complete(prompt: "Hello World")).to eq("\n\nThe meaning of life is subjective and can vary from person to person.")
      end
    end

    context "with custom default_options" do
      let(:subject) {
        described_class.new(
          api_key: "123",
          default_options: {completion_model_name: "gpt-3.5-turbo-16k"}
        )
      }

      let(:parameters) do
        {parameters:
          {model: "text-davinci-003",
           prompt: "Hello World",
           temperature: 0.0,
           max_tokens: 4095}}
      end

      it "passes correct options to the completions method" do
        expect(subject.client).to receive(:completions).with(
          {parameters: {max_tokens: 16382,
                        model: "gpt-3.5-turbo-16k",
                        prompt: "Hello World",
                        temperature: 0.0}}
        ).and_return(response)
        subject.complete(prompt: "Hello World")
      end
    end

    context "with prompt and parameters" do
      let(:parameters) do
        {parameters: {model: "text-curie-001", prompt: "Hello World", temperature: 1.0, max_tokens: 2047}}
      end

      it "returns a completion" do
        expect(subject.complete(prompt: "Hello World", model: "text-curie-001", temperature: 1.0)).to eq("\n\nThe meaning of life is subjective and can vary from person to person.")
      end
    end
  end

  describe "#default_dimension" do
    it "returns the default dimension" do
      expect(subject.default_dimension).to eq(1536)
    end
  end

  describe "#chat" do
    let(:prompt) { "What is the meaning of life?" }
    let(:model) { "gpt-3.5-turbo" }
    let(:temperature) { 0.0 }
    let(:history) { [content: prompt, role: "user"] }
    let(:parameters) { {parameters: {messages: history, model: model, temperature: temperature, max_tokens: be_between(4015, 4096)}} }
    let(:answer) { "As an AI language model, I don't have feelings, but I'm functioning well. How can I assist you today?" }
    let(:response) do
      {
        "id" => "chatcmpl-7Hcl1sXOtsaUBKJGGhNujEIwhauaD",
        "object" => "chat.completion",
        "created" => 1684434915,
        "model" => model,
        "usage" => {
          "prompt_tokens" => 14,
          "completion_tokens" => 25,
          "total_tokens" => 39
        },
        "choices" => [
          {
            "message" => {
              "role" => "assistant",
              "content" => answer
            },
            "finish_reason" => "stop",
            "index" => 0
          }
        ]
      }
    end

    before do
      allow(subject.client).to receive(:chat).with(parameters).and_return(response)
    end

    context "with prompt" do
      it "sends prompt within messages" do
        expect(subject.chat(prompt: prompt)).to eq(answer)
      end
    end

    context "with messages" do
      it "sends messages" do
        expect(subject.chat(messages: [role: "user", content: prompt])).to eq(answer)
      end
    end

    context "with context" do
      let(:context) { "You are a chatbot" }
      let(:history) do
        [
          {role: "system", content: context},
          {role: "user", content: prompt}
        ]
      end

      it "sends context and prompt as messages" do
        expect(subject.chat(prompt: prompt, context: context)).to eq(answer)
      end

      it "sends context and messages as joint messages" do
        expect(subject.chat(messages: [role: "user", content: prompt], context: context)).to eq(answer)
      end
    end

    context "with context and examples" do
      let(:context) { "You are a chatbot" }
      let(:examples) do
        [
          {role: "user", content: "Hello"},
          {role: "assistant", content: "Hi. How can I assist you today?"}
        ]
      end
      let(:history) do
        [
          {role: "system", content: context},
          {role: "user", content: "Hello"},
          {role: "assistant", content: "Hi. How can I assist you today?"},
          {role: "user", content: prompt}
        ]
      end

      it "sends context, prompt and examples as joint messages" do
        expect(subject.chat(prompt: prompt, context: context, examples: examples)).to eq(answer)
      end

      it "sends context, messages and examples as joint messages" do
        expect(subject.chat(messages: [role: "user", content: prompt], context: context, examples: examples)).to eq(answer)
      end

      context "with prompt, messages, context and examples" do
        let(:messages) do
          [
            {role: "user", content: "Can you answer questions?"},
            {role: "ai", content: "Yes, I can answer questions."}
          ]
        end
        let(:history) do
          [
            {role: "system", content: context},
            {role: "user", content: "Hello"},
            {role: "assistant", content: "Hi. How can I assist you today?"},
            {role: "user", content: "Can you answer questions?"},
            {role: "assistant", content: "Yes, I can answer questions."},
            {role: "user", content: prompt}
          ]
        end

        it "sends context, prompt, messages and examples as joint messages" do
          expect(subject.chat(prompt: prompt, messages: messages, context: context, examples: examples)).to eq(answer)
        end
      end

      context "when context is already present in messages" do
        let(:messages) do
          [
            {role: "system", content: context},
            {role: "user", content: "Hello"},
            {role: "assistant", content: "Hi. How can I assist you today?"},
            {role: "user", content: prompt}
          ]
        end
        let(:history) do
          [
            {role: "system", content: "You are a human being"},
            {role: "user", content: "Hello"},
            {role: "assistant", content: "Hi. How can I assist you today?"},
            {role: "user", content: prompt}
          ]
        end

        it "it overrides system message with context" do
          expect(subject.chat(messages: messages, context: "You are a human being")).to eq(answer)
        end
      end

      context "when last message is from user and prompt is present" do
        let(:messages) do
          [
            {role: "system", content: context},
            {role: "user", content: "Hello"},
            {role: "assistant", content: "Hi. How can I assist you today?"},
            {role: "user", content: "I want to ask a question"}
          ]
        end
        let(:history) do
          [
            {role: "system", content: context},
            {role: "user", content: "Hello"},
            {role: "assistant", content: "Hi. How can I assist you today?"},
            {role: "user", content: "I want to ask a question\n#{prompt}"}
          ]
        end

        it "it combines last message and prompt" do
          expect(subject.chat(prompt: prompt, messages: messages)).to eq(answer)
        end
      end
    end

    context "with options" do
      let(:temperature) { 0.75 }
      let(:model) { "gpt-3.5-turbo-0301" }

      it "sends prompt as message and additional params and returns a response message" do
        expect(subject.chat(prompt: prompt, model: model, temperature: temperature)).to eq("As an AI language model, I don't have feelings, but I'm functioning well. How can I assist you today?")
      end

      it "complete response" do
        subject.complete_response = true
        expect(subject.chat(prompt: prompt, model: model, temperature: temperature)).to be_a Hash
      end

      context "functions" do
        let(:parameters) { {parameters: {messages: history, model: model, temperature: temperature, functions: [{foo: :bar}]}} }

        it "functions will be passed on options as accessor" do
          subject.complete_response = true
          subject.functions = [{foo: :bar}]
          expect(subject.chat(prompt: prompt, model: model, temperature: temperature)).to be_a Hash
        end
      end
    end

    context "with failed API call" do
      let(:response) do
        {"error" => {"code" => 400, "message" => "User location is not supported for the API use.", "type" => "invalid_request_error"}}
      end

      it "raises an error" do
        expect {
          subject.chat(prompt: prompt)
        }.to raise_error(Langchain::LLM::ApiError, "Chat completion failed: User location is not supported for the API use.")
      end
    end
  end

  describe "#summarize" do
    let(:text) { "Text to summarize" }

    before do
      allow(subject).to receive(:complete).and_return("Summary")
    end

    it "returns a summary" do
      expect(subject.summarize(text: text)).to eq("Summary")
    end
  end
end
