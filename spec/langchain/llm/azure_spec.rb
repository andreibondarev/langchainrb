# frozen_string_literal: true

RSpec.describe Langchain::LLM::Azure do
  let(:subject) do
    described_class.new(
      api_key: "123",
      embedding_deployment_url: "http://localhost:1234/deployments/embedding",
      chat_deployment_url: "http://localhost:1234/deployments/chat"
    )
  end

  describe "#initialize" do
    context "when only required options are passed" do
      it "initializes the client without any errors" do
        expect { subject }.not_to raise_error
      end
    end

    context "when llm_options are passed" do
      let(:subject) do
        described_class.new(
          api_key: "123",
          llm_options: {api_type: :azure},
          embedding_deployment_url: "http://localhost:1234/deployments/embedding",
          chat_deployment_url: "http://localhost:1234/deployments/chat"
        )
      end

      it "initializes the client without any errors" do
        expect { subject }.not_to raise_error
      end

      it "passes correct options to the client" do
        # openai-ruby sets global configuration options here: https://github.com/alexrudall/ruby-openai/blob/main/lib/openai/client.rb
        result = subject
        expect(result.embed_client.api_type).to eq(:azure)
        expect(result.chat_client.api_type).to eq(:azure)
      end
    end
  end

  describe "#embed" do
    let(:result) { [-0.007097351, 0.0035200312, -0.0069700438] }
    let(:parameters) do
      {parameters: {input: "Hello World", model: "text-embedding-ada-002"}}
    end
    let(:response) do
      {
        "object" => "list",
        "model" => parameters[:parameters][:model],
        "data" => [
          {
            "object" => "embedding",
            "index" => 0,
            "embedding" => result
          }
        ],
        "usage" => {
          "prompt_tokens" => 2,
          "total_tokens" => 2
        }
      }
    end

    before do
      allow(subject.embed_client).to receive(:embeddings).with(parameters).and_return(response)
    end

    it "returns valid llm response object" do
      response = subject.embed(text: "Hello World")

      expect(response).to be_a(Langchain::LLM::OpenAIResponse)
      expect(response.model).to eq("text-embedding-ada-002")
      expect(response.embedding).to eq([-0.007097351, 0.0035200312, -0.0069700438])
      expect(response.prompt_tokens).to eq(2)
      expect(response.completion_tokens).to eq(nil)
      expect(response.total_tokens).to eq(2)
    end

    context "with default parameters" do
      it "returns an embedding" do
        response = subject.embed(text: "Hello World")

        expect(response).to be_a(Langchain::LLM::OpenAIResponse)
        expect(response.embedding).to eq(result)
      end
    end

    context "with text and parameters" do
      let(:parameters) do
        {parameters: {input: "Hello World", model: "text-embedding-ada-001", user: "id"}}
      end

      it "returns an embedding" do
        response = subject.embed(text: "Hello World", model: "text-embedding-ada-001", user: "id")

        expect(response).to be_a(Langchain::LLM::OpenAIResponse)
        expect(response.embedding).to eq(result)
      end
    end
  end

  describe "#complete" do
    let(:response) do
      {
        "id" => "cmpl-7BZg4cP5xzga4IyLI6u97WMepAJj2",
        "object" => "chat.completion",
        "created" => 1682993108,
        "model" => "gpt-3.5-turbo",
        "choices" => [
          {
            "message" => {
              "role" => "assistant",
              "content" => "The meaning of life is subjective and can vary from person to person."
            },
            "finish_reason" => "stop",
            "index" => 0
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
      allow(subject.chat_client).to receive(:chat).with(parameters).and_return(response)
      allow(subject.chat_client).to receive(:completions).with(parameters).and_return(response)
    end

    context "with default parameters" do
      let(:parameters) do
        {
          parameters: {
            n: 1,
            model: "gpt-3.5-turbo",
            messages: [{content: "Hello World", role: "user"}],
            temperature: 0.0,
            max_tokens: 4086
          }
        }
      end

      it "returns valid llm response object" do
        response = subject.complete(prompt: "Hello World")

        expect(response).to be_a(Langchain::LLM::OpenAIResponse)
        expect(response.model).to eq("gpt-3.5-turbo")
        expect(response.completion).to eq("The meaning of life is subjective and can vary from person to person.")
        expect(response.prompt_tokens).to eq(7)
        expect(response.completion_tokens).to eq(16)
        expect(response.total_tokens).to eq(23)
      end

      it "returns a completion" do
        response = subject.complete(prompt: "Hello World")

        expect(response).to be_a(Langchain::LLM::OpenAIResponse)
        expect(response.model).to eq("gpt-3.5-turbo")
        expect(response.completions).to eq([{"message" => {"role" => "assistant", "content" => "The meaning of life is subjective and can vary from person to person."}, "finish_reason" => "stop", "index" => 0}])
        expect(response.completion).to eq("The meaning of life is subjective and can vary from person to person.")
      end
    end

    context "with custom default_options" do
      context "with legacy model" do
        let(:logger) { double("logger") }
        let(:subject) {
          described_class.new(
            api_key: "123",
            default_options: {completion_model_name: "text-davinci-003"}
          )
        }
        let(:parameters) do
          {
            parameters:
            {
              n: 1,
              model: "text-davinci-003",
              prompt: "Hello World",
              temperature: 0.0,
              max_tokens: 4095
            }
          }
        end
      end

      context "with new model" do
        let(:subject) {
          described_class.new(
            api_key: "123",
            default_options: {completion_model_name: "gpt-3.5-turbo-16k"}
          )
        }

        let(:parameters) do
          {
            parameters: {
              n: 1,
              model: "gpt-3.5-turbo",
              messages: [{content: "Hello World", role: "user"}],
              temperature: 0.0,
              max_tokens: 4086
            }
          }
        end

        it "passes correct options to the chat method" do
          expect(subject.chat_client).to receive(:chat).with({
            parameters: {
              n: 1,
              max_tokens: 16374,
              model: "gpt-3.5-turbo-16k",
              messages: [{content: "Hello World", role: "user"}],
              temperature: 0.0
            }
          }).and_return(response)
          subject.complete(prompt: "Hello World")
        end
      end
    end

    context "with prompt and parameters" do
      let(:parameters) do
        {parameters: {n: 1, model: "gpt-3.5-turbo", messages: [{content: "Hello World", role: "user"}], temperature: 1.0, max_tokens: 16}}
      end

      it "returns a completion" do
        response = subject.complete(
          prompt: "Hello World",
          model: "gpt-3.5-turbo",
          temperature: 1.0,
          max_tokens: 16 # `max_tokens` can be passed in and overwritten.
        )

        expect(response.completion_tokens).to eq(16)
        expect(response.completion).to eq("The meaning of life is subjective and can vary from person to person.")
      end
    end

    context "with failed API call" do
      let(:parameters) do
        {parameters: {n: 1, model: "gpt-3.5-turbo", messages: [{content: "Hello World", role: "user"}], temperature: 0.0, max_tokens: 4086}}
      end
      let(:response) do
        {"error" => {"code" => 400, "message" => "User location is not supported for the API use.", "type" => "invalid_request_error"}}
      end

      it "raises an error" do
        expect {
          subject.complete(prompt: "Hello World")
        }.to raise_error(Langchain::LLM::ApiError, "OpenAI API error: User location is not supported for the API use.")
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
    let(:n) { 1 }
    let(:history) { [content: prompt, role: "user"] }
    let(:parameters) { {parameters: {n: n, messages: history, model: model, temperature: temperature, max_tokens: be_between(4014, 4096)}} }
    let(:answer) { "As an AI language model, I don't have feelings, but I'm functioning well. How can I assist you today?" }
    let(:answer_2) { "Alternative answer" }
    let(:choices) do
      [
        {
          "message" => {
            "role" => "assistant",
            "content" => answer
          },
          "finish_reason" => "stop",
          "index" => 0
        }
      ]
    end
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
        "choices" => choices
      }
    end

    before do
      allow(subject.chat_client).to receive(:chat).with(parameters).and_return(response)
    end

    it "returns valid llm response object" do
      response = subject.chat(prompt: "What is the meaning of life?")

      expect(response).to be_a(Langchain::LLM::OpenAIResponse)
      expect(response.model).to eq("gpt-3.5-turbo")
      expect(response.chat_completion).to eq("As an AI language model, I don't have feelings, but I'm functioning well. How can I assist you today?")
      expect(response.prompt_tokens).to eq(14)
      expect(response.completion_tokens).to eq(25)
      expect(response.total_tokens).to eq(39)
    end

    context "with prompt" do
      it "sends prompt within messages" do
        response = subject.chat(prompt: prompt)

        expect(response).to be_a(Langchain::LLM::OpenAIResponse)
        expect(response.model).to eq(model)
        expect(response.completions).to eq(choices)
        expect(response.chat_completion).to eq(answer)
      end
    end

    context "with messages" do
      it "sends messages" do
        response = subject.chat(messages: [{role: "user", content: prompt}])

        expect(response.chat_completion).to eq(answer)
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
        response = subject.chat(prompt: prompt, context: context)

        expect(response.chat_completion).to eq(answer)
      end

      it "sends context and messages as joint messages" do
        response = subject.chat(messages: [{role: "user", content: prompt}], context: context)

        expect(response.chat_completion).to eq(answer)
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
        response = subject.chat(prompt: prompt, context: context, examples: examples)

        expect(response.chat_completion).to eq(answer)
      end

      it "sends context, messages and examples as joint messages" do
        response = subject.chat(messages: [{role: "user", content: prompt}], context: context, examples: examples)

        expect(response.chat_completion).to eq(answer)
      end

      context "with prompt, messages, context and examples" do
        let(:messages) do
          [
            {role: "user", content: "Can you answer questions?"},
            {role: "assistant", content: "Yes, I can answer questions."}
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
          response = subject.chat(prompt: prompt, messages: messages, context: context, examples: examples)

          expect(response.chat_completion).to eq(answer)
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
          response = subject.chat(messages: messages, context: "You are a human being")

          expect(response.chat_completion).to eq(answer)
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
          response = subject.chat(prompt: prompt, messages: messages)

          expect(response.chat_completion).to eq(answer)
        end
      end
    end

    context "with options" do
      let(:temperature) { 0.75 }
      let(:model) { "gpt-3.5-turbo-0301" }

      it "sends prompt as message and additional params and returns a response message" do
        response = subject.chat(prompt: prompt, model: model, temperature: temperature)

        expect(response.chat_completion).to eq(answer)
      end

      context "with multiple choices" do
        let(:n) { 2 }
        let(:choices) do
          [
            {
              "message" => {"role" => "assistant", "content" => answer},
              "finish_reason" => "stop",
              "index" => 0
            },
            {
              "message" => {"role" => "assistant", "content" => answer_2},
              "finish_reason" => "stop",
              "index" => 1
            }
          ]
        end

        it "returns multiple response messages" do
          response = subject.chat(prompt: prompt, model: model, temperature: temperature, n: 2)

          expect(response.completions).to eq(choices)
        end
      end

      context "functions" do
        let(:parameters) { {parameters: {n: 1, messages: history, model: model, temperature: temperature, functions: [{foo: :bar}]}} }

        it "functions will be passed on options as accessor" do
          subject.functions = [{foo: :bar}]
          response = subject.chat(prompt: prompt, model: model, temperature: temperature)

          expect(response.chat_completion).to eq(answer)
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
        }.to raise_error(Langchain::LLM::ApiError, "OpenAI API error: User location is not supported for the API use.")
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
