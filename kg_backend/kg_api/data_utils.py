import os
import requests
import pandas as pd
from typing import List, Dict, Union, Optional
from datetime import datetime, timedelta
from goose3 import Goose
from dotenv import load_dotenv
from pathlib import Path
import tiktoken

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_DIR = os.path.join(BASE_DIR, '.env')
load_dotenv(ENV_DIR)

# API keys
FMP_API_KEY = os.getenv('FMP_API_KEY')
EOD_API_KEY = os.getenv('EOD_API_KEY')

if not FMP_API_KEY or not EOD_API_KEY:
    raise ValueError("API keys are not set. Please set FMP_API_KEY and EOD_API_KEY in your environment variables.")

def get_json_data(url: str) -> Optional[Union[Dict, List]]:
    """Fetches JSON data from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error estimating tokens: {e}")
        return 0

def fetch_articles(tickers: List[str], window: int = 1, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Retrieves news articles for given tickers within a specified time window or limit.
    
    Args:
        tickers (List[str]): List of stock symbols to fetch news for.
        window (int): Number of days from the current date to fetch articles. Defaults to 1.
        limit (Optional[int]): Limit on the number of articles to fetch per ticker. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame containing information about the fetched articles.
    """
    master_list = []
    g = Goose()
    date_format = "%Y-%m-%d"
    max_tokens = 0

    start_date = (datetime.now() - timedelta(days=window)).strftime(date_format)
    end_date = datetime.now().strftime(date_format)

    for ticker in tickers:
        print(f'Fetching articles for {ticker}')

        # Construct URLs for both FMP and EOD APIs
        if limit is None:
            fmp_url = f'https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&from={start_date}&to={end_date}&apikey={FMP_API_KEY}'
            eod_url = f'https://eodhistoricaldata.com/api/news?api_token={EOD_API_KEY}&s={ticker}.US&offset=0&from={start_date}&to={end_date}'
        else:
            fmp_url = f'https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit={limit}&apikey={FMP_API_KEY}'
            eod_url = f'https://eodhistoricaldata.com/api/news?api_token={EOD_API_KEY}&s={ticker}.US&offset=0&limit={limit}'

        # Fetch and process articles from both sources
        for source, url in [('fmp', fmp_url), ('eod', eod_url)]:
            articles = get_json_data(url)
            if articles:
                for article in articles:
                    try:
                        article_url = article.get('url') or article.get('link')
                        extracted_article = g.extract(url=article_url)
                        text = extracted_article.cleaned_text
                        full_text_available = 'yes'
                    except:
                        text = article.get('text') or article.get('content')
                        full_text_available = 'no'

                    # Estimate token count
                    estimated_tokens = estimate_tokens(text)
                    max_tokens = max(max_tokens, estimated_tokens)

                    new_row = {
                        'date': article.get('publishedDate') or article.get('date'),
                        'title': article['title'],
                        'link': article_url,
                        'news_text': text,
                        'ticker': ticker,
                        'full_text_available': full_text_available,
                        'source': source,
                        'estimated_tokens': estimated_tokens
                    }
                    master_list.append(new_row)

    print(f"Max tokens in an article: {max_tokens}")
    return pd.DataFrame(master_list)

def get_company_info(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches basic company information for the given tickers.
    
    Args:
        tickers (List[str]): List of stock symbols.
    
    Returns:
        pd.DataFrame: DataFrame containing company information.
    """
    company_info = []
    for ticker in tickers:
        url = f'https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}'
        data = get_json_data(url)
        if data and isinstance(data, list) and len(data) > 0:
            info = data[0]
            company_info.append({
                'symbol': info['symbol'],
                'name': info['companyName'],
                'sector': info['sector'],
                'industry': info['industry'],
                'description': info['description']
            })
    return pd.DataFrame(company_info)

if __name__ == "__main__":
    # Example usage
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    # Fetch company info
    companies = get_company_info(tickers)
    print("Company Information:")
    print(companies)
    
    # Fetch articles
    articles = fetch_articles(tickers, window=1, limit=100)
    print("\nFetched Articles:")
    print(articles.head())
    print(f"\nTotal articles fetched: {len(articles)}")
    articles.to_csv('articles.csv', index=False)