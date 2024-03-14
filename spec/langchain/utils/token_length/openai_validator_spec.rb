# frozen_string_literal: true

RSpec.describe Langchain::Utils::TokenLength::OpenAIValidator do
  describe "#validate_max_tokens!" do
    subject { described_class.validate_max_tokens!(content, model) }

    context "with text argument" do
      context "when the text is too long" do
        let(:content) { "lorem ipsum" * 9000 }
        let(:model) { "text-davinci-003" }

        it "raises an error" do
          expect {
            subject
          }.to raise_error(
            Langchain::Utils::TokenLength::TokenLimitExceeded,
            "This model's maximum context length is 4097 tokens, but the given text is 45000 tokens long."
          )
        end
      end

      context "when the text is not too long" do
        let(:content) { "lorem ipsum" * 100 }
        let(:model) { "gpt-4" }

        context "when the token_counter is tiktoken" do
          it "does not raise an error" do
            expect { subject }.not_to raise_error
          end

          it "returns the correct max_tokens" do
            expect(subject).to eq(7892)
          end
        end

        context "when the token_counter is openai" do
          subject { described_class.validate_max_tokens!(content, model, token_counter: :openai) }

          it "does not raise an error" do
            expect { subject }.not_to raise_error
          end

          it "returns the correct max_tokens" do
            expect(subject).to eq(7987)
          end
        end
      end

      context "when the model has a separate completion token limit" do
        let(:model) { "gpt-4-1106-preview" }

        context "where the leftover tokens are great than the completion token limit" do
          # 202 tokens
          let(:content) { "lorem ipsum " * 100 }

          it "does not raise an error" do
            expect { subject }.not_to raise_error
          end

          it "returns the correct max_tokens" do
            expect(subject).to eq(4096)
          end
        end

        context "where the leftover tokens are below the completion token limit" do
          # 126002 tokens, just under the input token limit of gpt-4-1106-preview
          let(:content) { "lorem ipsum " * 63_000 }

          it "does not raise an error" do
            expect { subject }.not_to raise_error
          end

          it "returns the correct max_tokens" do
            expect(subject).to eq(1998)
          end
        end
      end

      context "when the token is equal to the limit" do
        let(:content) { "lorem ipsum" * 9000 }
        let(:model) { "text-embedding-ada-002" }

        before do
          allow(described_class).to receive(:token_length).and_return(
            Langchain::Utils::TokenLength::OpenAIValidator::TOKEN_LIMITS[model]
          )
        end

        it "does not raise an error" do
          expect { subject }.not_to raise_error
        end

        it "returns the correct max_tokens" do
          expect(subject).to eq(0)
        end
      end

      context "when :max_tokens is passed in" do
        context "when :max_tokens is lower than the leftover tokens" do
          subject { described_class.validate_max_tokens!(content, model, max_tokens: 10) }
          let(:content) { "lorem ipsum" * 100 }
          let(:model) { "gpt-4" }

          it "returns the correct max_tokens" do
            expect(subject).to eq(10)
          end
        end

        context "when :max_tokens is greater than the leftover tokens" do
          subject { described_class.validate_max_tokens!(content, model, max_tokens: 8000) }
          let(:content) { "lorem ipsum" * 100 }
          let(:model) { "gpt-4" }

          it "returns the correct max_tokens" do
            expect(subject).to eq(7892)
          end
        end
      end
    end

    # Is there an actual usecase where we're passing in an array of texts to the OpenAIValidator?
    xcontext "with array argument" do
      let(:content) { ["lorem ipsum" * 100, "lorem ipsum" * 100] }
      let(:model) { "gpt-4" }

      context "when the text is not too long" do
        it "returns the correct max_tokens" do
          expect(subject).to eq(7588)
        end
      end
    end
  end

  describe "#token_length_from_messages" do
    let(:example_messages) {
      [{
        role: "system",
        content: "You are a helpful, pattern-following assistant that translates corporate jargon into plain English."
      }, {
        role: "system",
        name: "example_user",
        content: "New synergies will help drive top-line growth."
      }, {
        role: "system",
        name: "example_assistant",
        content: "Things working well together will increase revenue."
      }, {
        role: "system",
        name: "example_user",
        content: "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage."
      }, {
        role: "system",
        name: "example_assistant",
        content: "Let's talk later when we're less busy about how to do better."
      }, {
        role: "user",
        content: "This late pivot means we don't have time to boil the ocean for the client deliverable."
      }]
    }

    context "when the token counter is tiktoken" do
      it "returns the correct token length for gpt-3.5-turbo-0301" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-3.5-turbo-0301")
        ).to eq(127)
      end

      it "returns the correct token length for gpt-3.5-turbo-0613" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-3.5-turbo-0613")
        ).to eq(129)
      end

      it "returns the correct token length for gpt-3.5-turbo" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-3.5-turbo")
        ).to eq(129)
      end

      it "returns the correct token length for gpt-4-0613" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-4-0613")
        ).to eq(129)
      end

      it "returns the correct token length for gpt-4" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-4")
        ).to eq(129)
      end
    end

    context "when the token counter is openai" do
      let(:options) { {token_counter: :openai} }

      it "returns the correct token length for gpt-3.5-turbo-0301" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-3.5-turbo-0301", options)
        ).to eq(141)
      end

      it "returns the correct token length for gpt-3.5-turbo-0613" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-3.5-turbo-0613", options)
        ).to eq(143)
      end

      it "returns the correct token length for gpt-3.5-turbo" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-3.5-turbo", options)
        ).to eq(143)
      end

      it "returns the correct token length for gpt-4-0613" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-4-0613", options)
        ).to eq(143)
      end

      it "returns the correct token length for gpt-4" do
        expect(
          described_class.token_length_from_messages(example_messages, "gpt-4", options)
        ).to eq(143)
      end
    end
  end
end
