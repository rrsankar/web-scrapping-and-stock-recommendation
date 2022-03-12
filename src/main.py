""" Importing all the necessary libraries/modules for the program."""

import re
import warnings
from collections import Counter
from datetime import datetime, timedelta
from typing import Tuple

import datefinder
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import tensorflow as tf
import yfinance as yf
from bs4 import BeautifulSoup
from tqdm import tqdm

warnings.filterwarnings("ignore")


MODEL_PATH = "/models/LSTM_model"

# Initialize constants, setting the number of articles extracted from the website to 4.
ARTICLES_TO_EXTRACT_PER_DAY = 4


class Parser:
    """
    Class with methods to generate links to news releases, links to articles, parse/scrape articles and extract required information.
    """

    def generate_links_for_date_wise_news_releases(self) -> list:
        """
        Method to generate a list of links that fetches news release of past two weeks.
        :return: list of links to get date wise news releases.
        """

        # Get start and end date to fetch articles.
        current_time = datetime.now()
        time_two_weeks_back = current_time - timedelta(days=13)
        print(f"\nParsing news from "
              + time_two_weeks_back.strftime("%m/%d/%Y %H:00")
              + " to " + current_time.strftime("%m/%d/%Y %H:00") + "\n")

        # Generate a list of links that shows news release of past two weeks.
        filtered_link_list = []
        for single_date in pd.date_range(time_two_weeks_back, current_time):
            filtered_link_list.append(f"https://www.prnewswire.com/news-releases/news-releases-list/"
                                      f"?page=1&pagesize={ARTICLES_TO_EXTRACT_PER_DAY}&month={single_date.month:02}"
                                      f"&day={single_date.day:02}&year={single_date.year:04}&hour={single_date.hour:02}")

        # Display links.
        for itr in filtered_link_list:
            print(itr)

        return filtered_link_list

    def generate_links_to_articles(self, filtered_news_links_list) -> list:
        """
        Method to generate a list of links to articles.

        :param filtered_news_links_list: list of links to news releases.
        :return articles_link_list: list of article links.
        """

        print("\n\nFetching links to articles from past two weeks..\n")

        # Looping through each date wise filtered news release links and fetching links to that days articles.
        articles_link_list = []
        for i in tqdm(filtered_news_links_list):
            response = requests.get(i)
            home_page = BeautifulSoup(response.text, 'html.parser')
            # Links to articles will be in the "a" tag with class "newsreleaseconsolidatelink display-outline".
            news_release_list = list(
                home_page.find_all("a", attrs={'class': 'newsreleaseconsolidatelink display-outline'}))
            # Adding all article links to one list.
            articles_link_list.extend(
                [f"https://www.prnewswire.com{i.attrs.get('href')}" for i in news_release_list if
                 i.attrs.get("href", "")])

        print("\n\nFetching complete.\n\n")

        # Display all article links.
        for itr in articles_link_list:
            print(itr)

        print(f"\n\nNumber of articles: {len(articles_link_list)}")

        return articles_link_list

    def parse_data_from_web(self, articles_link_list) -> pd.DataFrame:
        """
        Method to loop through article links, parse article body, date and add to dataframe.
        :param articles_link_list: list of links to articles.
        :return: Parsed data in dataframe.
        """

        print("\n\nExtracting information from scrapped content..\n")

        # Define dataframe.
        data = pd.DataFrame(columns=["url", "article_date", "article_content"])

        # Looping through article links and parsing, retrieving, storing information.
        for i in tqdm(articles_link_list):
            blog = requests.get(i)
            blog_soup = BeautifulSoup(blog.text, 'html.parser')
            # Article body is present in "section tag" with class "release-body container"
            blog_body = blog_soup.find("section", attrs={'class': 'release-body container'})
            if not blog_body:
                blog_body = blog_soup.find("section", attrs={'class': 'release-body container '})
            blog_body = blog_body.text if blog_body else ""

            # Fetch article date which is present in "meta" tag.
            blog_date = blog_soup.find("meta", attrs={'name': 'date'}).attrs.get("content", "")
            matches = list(datefinder.find_dates(blog_date))
            blog_date = str(matches[0]) if matches else ""

            # Append retrieved information to dataframe.
            data = data.append({
                "url": i,
                "article_date": blog_date,
                "article_content": blog_body},
                ignore_index=True)

        print("\n\nExtraction complete. All information added to dataframe.\n")

        return data


class Tracker:
    """
    Class to store extracted data in xlsx format and fetch stock tickers from it.
    """

    def __init__(self, data):
        self.data = data

    def store_data_as_excel(self) -> None:
        """
        Method to store data in dataframe as excel.
        :return None:
        """

        # Saving dataframe as excel with pandas to_excel function.
        self.data.to_excel(f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_scrapped_data.xlsx", index=False)

    def preprocess_data(self) -> None:
        """
        Method to do preprocessing of data before extracting tickers.
        :return:
        """

        # Dropping duplicates if present.
        self.data.drop_duplicates(subset="article_content", inplace=True, ignore_index=True)

    def fetch_tickers(self) -> list:
        """
        Method to fetch tickers from all article content.
        :return: Set of tickers.
        """

        tickers = []
        for i in range(len(self.data)):

            # Find all ticker tokens in each parsed article content.
            tmp_ticker_tokens = re.findall(r':\s[A-Z]{1,5}[)]', self.data.iloc[i]["article_content"])

            # Using lambda function, loop through all extracted ticker tokens and convert to required format.
            tickers.extend(list(map(lambda x: x[-(len(x) - 2):-1], tmp_ticker_tokens)))

        return tickers

    def fetch_most_frequent_ticker(self, tickers) -> str:
        """
        Method to fetch most frequent ticker from all article content.
        :return: Set of tickers.
        """

        frequent_stock = ""
        if tickers:
            cnt = Counter(tickers)
            frequent_stock = cnt.most_common(1)[0][0] if cnt else ""
        else:
            print("No tickers found.")

        # Check if format is correct.
        if not isinstance(frequent_stock, str):
            raise Exception(f"Format error for 'frequent_stock' variable. Supposed to be 'str', instead got {type(frequent_stock)}")

        return frequent_stock


class Retriever:
    """
    Class to retrieve stock information for tickers found in scrapped content.
    """

    def __init__(self, tickers):
        self.tickers = tickers

    def retrieve(self) -> Tuple[dict, list]:
        """
        Method to retrieve ticker prices using yahoo finance api.
        :return: Stock information as dict.
        """

        stocks = {}
        for tick in self.tickers:
            # If ticker info is unavailable to fetch from yahoo API, remove ticker from ticker list.
            if yf.Ticker(tick).history(period="YTD").empty:
                self.tickers.remove(tick)
            else:
                stocks[tick] = yf.Ticker(tick).history(period="YTD")

        return stocks, self.tickers


class Visualizer:
    """
    Class to display visualization of stock information.
    """

    def __init__(self, stocks):
        self.stocks = stocks

    def generate_candle_stick_visualization(self, ticker, increasing_line, decreasing_line):
        fig = go.Figure(data=[go.Candlestick(
            x=self.stocks[ticker].index,
            open=self.stocks[ticker]['Open'],
            high=self.stocks[ticker]['High'],
            low=self.stocks[ticker]['Low'],
            close=self.stocks[ticker]['Close'],
            increasing_line_color=increasing_line,
            decreasing_line_color=decreasing_line)
            ])
        fig.update_layout(autosize=False,
                          width=1000,
                          height=800,)
        fig.show()

    def plot_tickers(self, ticker, index=0):
        fig = plt.figure(figsize=(30, 25))
        ax1 = plt.subplot(2, 2, 1)
        plt.title("Close Price", fontsize=25)
        plt.xlabel("Date", fontsize=25)
        plt.ylabel("Prie in USD", fontsize=25)
        plt.xticks(rotation=45)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax2 = plt.subplot(2, 2, 2)
        plt.title("Volume", fontsize=25)
        plt.xticks(rotation=45)
        plt.xticks(fontsize=20)
        plt.xlabel("Date", fontsize=25)
        plt.ylabel("Shares per day", fontsize=25)
        plt.yticks(fontsize=20)
        ax2.yaxis.offsetText.set_fontsize(20)
        ax1.plot(self.stocks[ticker]["Close"][index:])
        ax2.plot(self.stocks[ticker]["Volume"][index:])
        plt.suptitle("Charts for " + ticker, fontsize=40)


class StockRecommender:
    """
    Class to predict if a stock is worth purchasing.
    """

    def __init__(self, stocks):
        self.stocks = stocks

        # Loading stock recommendation model.
        self.model = tf.keras.models.load_model(f"{MODEL_PATH}")

    def model_predict(self, X):
        temp = self.model.predict(X)
        if 0.55 > temp[0][0] > 0.45:
            print("Wait before buying stock.")
        else:
            if np.argmax(self.model.predict(X)) == 0:
                print("Don't buy stock.")
            else:
                print("Buy stock.")

    def preprocess(self, ticker):
        series = self.stocks[ticker]["Close"]
        series = series[-31:]
        series = series.pct_change()
        series.dropna(inplace=True)
        X = np.array(series)
        return self.model_predict(X.reshape(1, X.shape[0], 1))


def main():

    # Web scraping.
    parser_obj = Parser()
    date_wise_news_releases_links = parser_obj.generate_links_for_date_wise_news_releases()
    articles_links = parser_obj.generate_links_to_articles(date_wise_news_releases_links)
    data = parser_obj.parse_data_from_web(articles_links)

    # Tracking: storing data and generating tickers.
    tracker_obj = Tracker(data)
    tracker_obj.store_data_as_excel()
    tracker_obj.preprocess_data()
    tickers = tracker_obj.fetch_tickers()

    # Retrieving stock info.
    retriever_obj = Retriever(tickers)
    stocks, tickers = retriever_obj.retrieve()
    try:
        common_ticker = tracker_obj.fetch_most_frequent_ticker(tickers)
    except Exception as err:
        common_ticker = ""
        print("\nERROR:", str(err))

    # Visualization
    visualizer_obj = Visualizer(stocks)
    i_cl = ["gold", "green", "cyan"]
    d_cl = ["gray", "red", "black"]
    widgets.interact(visualizer_obj.generate_candle_stick_visualization, ticker=stocks.keys(), increasing_line=i_cl, decreasing_line=d_cl)
    widgets.interact(visualizer_obj.plot_tickers, ticker=stocks.keys())

    # Stock recommendation
    stock_recommender_obj = StockRecommender(stocks)
    widgets.interact(stock_recommender_obj.preprocess, ticker=stocks.keys())

    # Show recommendation and visualization for most frequent stock symbol.
    if common_ticker:
        stock_recommender_obj.preprocess(common_ticker)
        visualizer_obj.plot_tickers(common_ticker, index=-30)
    else:
        print("\nERROR: common ticker not available.")


main()
