from crewai import Agent, Task, Crew, Process
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from tools.yf_tech_analysis_tool import yf_tech_analysis
from tools.yf_fundamental_analysis_tool import yf_fundamental_analysis
from tools.sentiment_analysis_tool import sentiment_analysis
from tools.competitor_analysis_tool import competitor_analysis
from tools.rist_assessment_tool import risk_assessment

from dotenv import load_dotenv
load_dotenv()

def create_crew(stock_symbol):
    # Initialize Ollama LLM 
    # llm = OllamaLLM(model="llama2")
    llm = "ollama/llama3.1"

    # Define agents
    researcher = Agent(
        role="Stock Market Researcher",
        goal="Gather and analyze comprehensive data about the stock",
        backstory="You're an experienced stock market researcher with a keen eye for detail and a talent for uncovering hidden trends.",
        tools=[yf_tech_analysis, yf_fundamental_analysis, competitor_analysis],
        # llm="ollama/llama2",
        llm=llm
    )

    analyst = Agent(
        role="Financial Analyst",
        goal="Analyze the gathered data and provide investment insights",
        backstory="You're a seasoned financial analyst known for your accurate predictions and ability to synthesize complex information.",
        tools=[yf_tech_analysis, yf_fundamental_analysis, risk_assessment],
        # llm="ollama/llama2",
        llm=llm
    )

    sentiment_analyst = Agent(
        role="Sentiment Analyst",
        goal="Analyze market sentiment and its potential impact on the stock",
        backstory="You're an expert in behaviorla finance and sentiment analysis, capable of gauging emotions and their effects on stock performance.",
        tools=[sentiment_analysis],
        # llm="ollama/llama2",
        llm=llm
    )

    strategist = Agent(
        role="Investment Strategist",
        goal="Develop a comprehensive investment strategy based on all available data",
        backstory="You're a renowned investment strategist known for creating tailored investment plans that balance risk and reward.",
        tools=[],
        # llm="ollama/llama2",
        llm=llm
    )


    # Create Crew
    research_task = Task(
        description=f"Research {stock_symbol} using advanced technical and fundamental anlysis tools. Provide a comprehensive summary of key metrics, including chart patterns, financial ratios, and competitor analysis.",
        agent=researcher,
        expected_output=(
        "A detailed research summary of stock metrics, including:\n"
        "- Key financial ratios (P/E, ROE, etc.), technical indicators (RSI, MACD), and chart patterns.\n"
        "- Competitor analysis with key metrics and comparisons.\n"
        "The output should combine fundamental, technical, and competitive insights."
        )
    )

    sentiment_task = Task(
        description=f"Analyze the market sentiment for {stock_symbol} using news and social media data. Evaluate how current sentiment might affect the stock's performance.",
        agent=sentiment_analyst,
        expected_output=(
        "A sentiment analysis report summarizing:\n"
        "- Average sentiment from news and social media.\n"
        "- Overall sentiment score and its potential impact on the stock.\n"
        "The output should highlight key sentiment trends and their effects."
        )   
    )

    analysis_task = Task(
        description=f"Synthesize the research data and sentiment analysis for {stock_symbol}. Conduct a thorough risk assessment and provide a detailed analysis of the stock's potential.",
        agent=analyst,
        expected_output=(
        "A comprehensive analysis including:\n"
        "- Insights from research, sentiment analysis, and risk assessment (beta, Sharpe ratio, etc.).\n"
        "- Key risks and opportunities for the stock.\n"
        "The output should evaluate the stock’s potential and provide actionable insights."
        )
    )

    strategy_task = Task(
        description=f"Based on all the gathered information about {stock_symbol}, develop a comprehensive investment strategy. Consider various and provide actionable recommendatations for different investor profiles.",
        agent=strategist,
        expected_output=(
        "A comprehensive investment strategy including:\n"
        "- Recommendations for different investor profiles (conservative, aggressive, balanced).\n"
        "- Suggested buy/sell/hold strategies with entry/exit points.\n"
        "The output should provide actionable guidance based on research and risk analysis."
        )
    )

    crew = Crew(
        agents=[researcher, sentiment_analyst, analyst, strategist],
        tasks=[research_task, sentiment_task, analysis_task, strategy_task],
        process=Process.sequential
    )

    return crew


def run_analysis(stock_symbol):
    crew = create_crew(stock_symbol)
    result = crew.kickoff()
    return result