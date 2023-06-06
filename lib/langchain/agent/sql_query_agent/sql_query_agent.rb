# frozen_string_literal: true

module Langchain::Agent
  class SQLQueryAgent < Base
    attr_reader :llm, :db, :schema

    #
    # Initializes the Agent
    #
    # @param llm [Object] The LLM client to use
    # @param db [Object] Database connection info
    #
    def initialize(llm:, db:)
      @llm = llm
      @db = db
      @schema = @db.schema
    end

    #
    # Ask a question and get an answer
    #
    # @param question [String] Question to ask the LLM/Database
    # @return [String] Answer to the question
    #
    def run(question:)
      prompt = create_prompt_for_sql(question: question)

      # Get the SQL string to execute
      Langchain.logger.info("[#{self.class.name}]".red + ":  Passing the inital prompt to the #{llm.class} LLM")
      max_tokens = Langchain::Utils::TokenLengthValidator.calculate_max_tokens(prompt, llm.model_name)
      sql_string = llm.complete(prompt: prompt, max_tokens: max_tokens)

      # Execute the SQL string and collect the results
      Langchain.logger.info("[#{self.class.name}]".red + ":  Passing the SQL to the Database: #{sql_string}")
      results = db.execute(input: sql_string)

      # Pass the results and get the LLM to synthesize the answer to the question
      Langchain.logger.info("[#{self.class.name}]".red + ":  Passing the synthesize prompt to the #{llm.class} LLM with results: #{results}")
      prompt2 = create_prompt_for_answer(question: question, sql_query: sql_string, results: results)
      max_tokens = Langchain::Utils::TokenLengthValidator.calculate_max_tokens(prompt2, llm.model_name)
      llm.complete(prompt: prompt2, max_tokens: max_tokens)
    end

    private

    # Create the initial prompt to pass to the LLM
    # @param question[String] Question to ask
    # @return [String] Prompt
    def create_prompt_for_sql(question:)
      prompt_template_sql.format(
        dialect: "standard SQL",
        schema: schema,
        question: question
      )
    end

    # Load the PromptTemplate from the JSON file
    # @return [PromptTemplate] PromptTemplate instance
    def prompt_template_sql
      Langchain::Prompt.load_from_path(
        file_path: Langchain.root.join("langchain/agent/sql_query_agent/sql_query_agent_sql_prompt.json")
      )
    end

    # Create the second prompt to pass to the LLM
    # @param question [String] Question to ask
    # @return [String] Prompt
    def create_prompt_for_answer(question:, sql_query:, results:)
      prompt_template_answer.format(
        question: question,
        sql_query: sql_query,
        results: results
      )
    end

    # Load the PromptTemplate from the JSON file
    # @return [PromptTemplate] PromptTemplate instance
    def prompt_template_answer
      Langchain::Prompt.load_from_path(
        file_path: Langchain.root.join("langchain/agent/sql_query_agent/sql_query_agent_answer_prompt.json")
      )
    end
  end
end
