import requests
import praw
import yfinance as yf
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Reddit API initialization
reddit = praw.Reddit(
    client_id='vZAg9lHB2nvxY0wR_LbIwg',
    client_secret='owDrVmQdzYYQfyP4SHTjKFtaLKMe1w',
    user_agent='myApp:v1.0 (by /u/Sidhu1748)'
)

class ESGScorer:
    def __init__(self):
        self.keywords = {
            'environmental': {
                'carbon emissions': 5,
                'renewable energy': 4,
                'waste management': 3,
                'climate change': 5,
                'biodiversity': 4,
                'water conservation': 3,
            },
            'social': {
                'employee satisfaction': 4,
                'workplace safety': 5,
                'human rights': 5,
                'diversity inclusion': 4,
                'community engagement': 3,
                'labor practices': 4,
            },
            'governance': {
                'board diversity': 4,
                'business ethics': 5,
                'corruption': 5,
                'transparency': 4,
                'executive compensation': 3,
                'shareholder rights': 4,
            }
        }

    @lru_cache(maxsize=100)
    def fetch_wikirate_data(self, company_name: str) -> Dict:
        """Fetch data from WikiRate API with caching"""
        headers = {'Authorization': 'sN1yYSIMfAEwQ97LQqyZ5gtt'}
        response = requests.get(f'https://wikirate.org/{company_name}+Answer.json', headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"WikiRate data for {company_name}: {data}")
            return data
        else:
            print(f"Error fetching WikiRate data for {company_name}: {response.status_code}")
            return {}

    def fetch_reddit_sentiment(self, company_name: str) -> float:
        """Fetch Reddit posts and calculate sentiment"""
        subreddit = reddit.subreddit('investing')
        posts = subreddit.search(company_name, limit=100, time_filter='month')
        sentiment_scores = [analyzer.polarity_scores(post.title + ' ' + post.selftext)['compound'] for post in posts]
        return np.mean(sentiment_scores) if sentiment_scores else 0

    def fetch_news_sentiment(self, company_name: str) -> float:
        """Fetch news articles and calculate sentiment"""
        base_url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{company_name} AND (ESG OR sustainability OR environmental OR social OR governance)",
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': '389668f989f14dd3a72f1eae8f3a3495'
        }
        response = requests.get(base_url, params=params)
        articles = response.json().get('articles', [])
        sentiment_scores = [analyzer.polarity_scores(article.get('title', ''))['compound'] for article in articles if article.get('title')]
        return np.mean(sentiment_scores) if sentiment_scores else 0

    def calculate_keyword_score(self, text: str, category: str) -> float:
        """Calculate score based on keyword presence and importance"""
        score = 0
        for keyword, importance in self.keywords[category].items():
            if keyword in text.lower():
                score += importance
        return min(score, 100)  # Cap the score at 100

    def calculate_esg_score(self, company_name: str, ticker: str) -> Dict:
        """Calculate ESG score for a company and scale it to a 0-100 range"""
        wikirate_data = self.fetch_wikirate_data(company_name)
        reddit_sentiment = self.fetch_reddit_sentiment(company_name)
        news_sentiment = self.fetch_news_sentiment(company_name)

        def scale_sentiment(sentiment: float) -> float:
            return round((sentiment + 1) * 50)  # Converts [-1, 1] to [0, 100]

        scaled_reddit_sentiment = scale_sentiment(reddit_sentiment)
        scaled_news_sentiment = scale_sentiment(news_sentiment)

        # Calculate scores based on keywords in WikiRate data
        environmental_score = self.calculate_keyword_score(str(wikirate_data), 'environmental')
        social_score = self.calculate_keyword_score(str(wikirate_data), 'social')
        governance_score = self.calculate_keyword_score(str(wikirate_data), 'governance')

        # Incorporate sentiment scores
        environmental_score = (environmental_score + scaled_news_sentiment) / 2
        social_score = (social_score + scaled_reddit_sentiment) / 2
        governance_score = (governance_score + scaled_reddit_sentiment) / 2

        # Calculate total ESG score
        total_score = (environmental_score + social_score + governance_score) / 3

        return {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'total_score': round(total_score, 2),
            'environmental_score': round(environmental_score, 2),
            'social_score': round(social_score, 2),
            'governance_score': round(governance_score, 2),
        }

    def visualize_esg_scores(self, score_data: Dict):
        """Create an interactive radar chart for ESG scores"""
        categories = ['Environmental', 'Social', 'Governance']
        scores = [score_data['environmental_score'], score_data['social_score'], score_data['governance_score']]

        fig = go.Figure(data=go.Scatterpolar(
            r=scores + scores[:1],
            theta=categories + categories[:1],
            fill='toself',
            name=score_data['ticker']
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title=f"ESG Scores for {score_data['ticker']}"
        )
        fig.show()

    def visualize_esg_comparison(self, all_scores: List[Dict]):
        """Create a comparative bar chart for multiple companies"""
        companies = [score['ticker'] for score in all_scores]
        env_scores = [score['environmental_score'] for score in all_scores]
        soc_scores = [score['social_score'] for score in all_scores]
        gov_scores = [score['governance_score'] for score in all_scores]

        fig = go.Figure(data=[
            go.Bar(name='Environmental', x=companies, y=env_scores),
            go.Bar(name='Social', x=companies, y=soc_scores),
            go.Bar(name='Governance', x=companies, y=gov_scores)
        ])

        fig.update_layout(
            barmode='group',
            title='ESG Score Comparison',
            yaxis_title='Score',
            yaxis=dict(range=[0, 100])
        )
        fig.show()

def fetch_financial_data(ticker: str) -> Dict:
    """Fetch financial data for a given ticker"""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        'ticker': ticker,
        'pe_ratio': info.get('forwardPE', None),
        'pb_ratio': info.get('priceToBook', None),
        'beta': info.get('beta', None),
        'rsi': calculate_rsi(stock.history(period="1mo")['Close'])
    }

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs.iloc[-1]))

def main():
    esg_scorer = ESGScorer()
    companies = [('Apple', 'AAPL'), ('Microsoft', 'MSFT'), ('Google', 'GOOGL'), ('Amazon', 'AMZN'), ('Tesla', 'TSLA')]

    with ThreadPoolExecutor(max_workers=5) as executor:
        esg_futures = {executor.submit(esg_scorer.calculate_esg_score, company, ticker): (company, ticker) for company, ticker in companies}
        financial_futures = {executor.submit(fetch_financial_data, ticker): (company, ticker) for company, ticker in companies}

    all_esg_scores = []
    all_financial_data = []

    for future in esg_futures:
        company, ticker = esg_futures[future]
        try:
            score_data = future.result()
            all_esg_scores.append(score_data)
            esg_scorer.visualize_esg_scores(score_data)
        except Exception as e:
            print(f"Error calculating ESG score for {company}: {e}")

    for future in financial_futures:
        company, ticker = financial_futures[future]
        try:
            financial_data = future.result()
            all_financial_data.append(financial_data)
        except Exception as e:
            print(f"Error fetching financial data for {company}: {e}")

    esg_scorer.visualize_esg_comparison(all_esg_scores)
    visualize_financial_data(all_financial_data)

def visualize_financial_data(financial_data: List[Dict]):
    """Create a comprehensive financial dashboard"""
    fig = make_subplots(rows=2, cols=2, subplot_titles=("P/E Ratio", "P/B Ratio", "Beta", "RSI"))

    companies = [data['ticker'] for data in financial_data]
    pe_ratios = [data['pe_ratio'] for data in financial_data]
    pb_ratios = [data['pb_ratio'] for data in financial_data]
    betas = [data['beta'] for data in financial_data]
    rsis = [data['rsi'] for data in financial_data]

    fig.add_trace(go.Bar(x=companies, y=pe_ratios, name="P/E Ratio"), row=1, col=1)
    fig.add_trace(go.Bar(x=companies, y=pb_ratios, name="P/B Ratio"), row=1, col=2)
    fig.add_trace(go.Bar(x=companies, y=betas, name="Beta"), row=2, col=1)
    fig.add_trace(go.Bar(x=companies, y=rsis, name="RSI"), row=2, col=2)

    fig.update_layout(height=800, width=1000, title_text="Financial Indicators Dashboard")
    fig.show()

if __name__ == "__main__":
    main()
