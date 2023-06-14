# frozen_string_literal: true

RSpec.describe Langchain::Conversation do
  let(:llm) { double("Langchain::LLM::OpenaAI") }

  subject { described_class.new(llm: llm) }

  describe "#set_context" do
    let(:context) { "You are a chatbot" }

    it "sets the context" do
      subject.set_context(context)

      expect(subject.context).to eq(context)
    end
  end

  describe "#add_examples" do
    let(:examples1) { [{role: "user", content: "Hello"}, {role: "ai", content: "Hi"}] }
    let(:examples2) { [{role: "user", content: "How are you doing?"}, {role: "ai", content: "I'm doing well. How about you?"}] }

    it "adds examples" do
      subject.add_examples(examples1)

      expect(subject.instance_variable_get(:@examples)).to eq(examples1)

      subject.add_examples(examples2)
      expect(subject.instance_variable_get(:@examples)).to eq(examples1 | examples2)
    end
  end

  describe "#message" do
    let(:context) { "You are a chatbot" }
    let(:examples) { [{role: "user", content: "Hello"}, {role: "ai", content: "Hi"}] }
    let(:prompt) { "How are you doing?" }
    let(:response) { "I'm doing well. How about you?" }

    context "with stream: true option and block passed in" do
      let(:block) { proc { |chunk| print(chunk) } }
      let(:conversation) { described_class.new(llm: llm, &block) }

      it "messages the model and yields the response" do
        expect(llm).to receive(:chat).with(
          context: nil,
          examples: [],
          messages: [{role: "user", content: prompt}],
          &block
        ).and_return(response)

        expect(conversation.message(prompt)).to eq(response)
      end
    end

    context "with simple prompt" do
      it "messages the model and returns the response" do
        expect(llm).to receive(:chat).with(
          context: nil,
          examples: [],
          messages: [{role: "user", content: prompt}]
        ).and_return(response)

        expect(subject.message(prompt)).to eq(response)
      end
    end

    context "with context" do
      before do
        subject.set_context(context)
      end

      it "messages the model and returns the response" do
        expect(llm).to receive(:chat).with(
          context: context,
          examples: [],
          messages: [{role: "user", content: prompt}]
        ).and_return(response)

        expect(subject.message(prompt)).to eq(response)
      end
    end

    context "with context and examples" do
      before do
        subject.set_context(context)
        subject.add_examples(examples)
      end

      it "messages the model and returns the response" do
        expect(llm).to receive(:chat).with(
          context: context,
          examples: [
            {role: "user", content: "Hello"},
            {role: "ai", content: "Hi"}
          ],
          messages: [
            {role: "user", content: prompt}
          ]
        ).and_return(response)

        expect(subject.message(prompt)).to eq(response)
      end
    end

    context "with options" do
      subject { described_class.new(llm: llm, temperature: 0.7) }

      it "messages the model with passed options" do
        expect(llm).to receive(:chat).with(
          context: nil,
          examples: [],
          messages: [{role: "user", content: prompt}],
          temperature: 0.7
        ).and_return(response)

        expect(subject.message(prompt)).to eq(response)
      end
    end
  end
end
