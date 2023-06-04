# frozen_string_literal: true

module Langchain::Agent
  class SQLQueryAgent < Base
    attr_reader :llm

    # Initializes the Agent
    #
    # @param llm [Symbol] The LLM to use
    # @param llm_api_key [String] The API key for the LLM
    # @param db_connection_string [String] Database connection info
    def initialize(llm:, db_connection_string:)
      @llm = llm
      @db = Langchain::Tool::Database.new(db_connection_string)
      @schema = @db.schema
    end

    def ask(question:)
      prompt = create_prompt_for_sql(question: question)

      # Get the SQL string to execute
      Langchain.logger.info("[#{self.class.name}]".red + ":  Passing the inital prompt to the #{@llm} LLM")
      sql_string = llm.complete(prompt: prompt)

      # Execute the SQL string and collect the results
      Langchain.logger.info("[#{self.class.name}]".red + ":  Passing the SQL to the Database: #{sql_string}")
      results = @db.execute(input: sql_string)

      # Pass the results and get the LLM to synthesize the answer to the question
      Langchain.logger.info("[#{self.class.name}]".red + ":  Passing the synthesize prompt to the #{@llm} LLM with results: #{results}")
      prompt2 = create_prompt_for_answer(question: question, sql_query: sql_string, results: results)
      llm.complete(prompt: prompt2)
    end

    private

    # Create the initial prompt to pass to the LLM
    # @param question[String] Question to ask
    # @return [String] Prompt
    def create_prompt_for_sql(question:)
      prompt_template_sql.format(
        dialect: "standard SQL",
        schema: @schema,
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
