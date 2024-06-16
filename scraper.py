import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from transformers import pipeline
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

def get_stock_data(tickerStr, date_time_dict):
    # data = yf.download(ticker, '2024-01-01', '2024-05-01')
    # data['Adj Close'].plot()
    # plt.title(ticker)
    # plt.show()

    results = []
    for date_i in date_time_dict.keys():
        next_day = datetime.datetime.strptime(date_i, '%Y-%m-%d')
        next_day += datetime.timedelta(days=1)
        next_day = str(next_day)[:10]

        df = yf.download(tickerStr, start=date_i, end=next_day, rounding=True)
        if not df.empty:
            df.reset_index(inplace=True)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            results.append(df)
    df_res = pd.concat(results)
    return df_res


def monthStrToNum(month):
    if month == 'Jan':
        return "01"
    if month == 'Feb':
        return "02"
    if month == 'Mar':
        return "03"
    if month == 'Apr':
        return "04"
    if month == 'May':
        return "05"
    if month == 'Jun':
        return "06"
    if month == 'Jul':
        return "07"
    if month == 'Aug':
        return "08"
    if month == 'Sep':
        return "09"
    if month == 'Oct':
        return "10"
    if month == 'Nov':
        return "11"
    if month == 'Dec':
        return "12"


def scrape_google_news(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        article_titles = soup.find_all('a', class_='JtKRv')
        # for title in article_titles:
        #    print(title.text)

        article_dates = soup.find_all('time', class_='hvbAAd')  # find dates

        date_title_dict = {}
        for idx in range(len(article_dates)):
            title = article_titles[idx]
            article_date = article_dates[idx]

            datestr = article_date.attrs['datetime']

            ''''
            # print(article_date.attrs)
            
            # print(article_date.attrs['datetime'])

            # tokens = article_date.text.split(" ")
            # datestr = ""

            # print(tokens)
            
            if "ago" in tokens:
                now = datetime.datetime.now()
                if "hours" in tokens:
                    time_change = datetime.timedelta(hours=-1*int(tokens[0]))
                    datestr = str(now + time_change)
                else:
                    # days ago
                    time_change = datetime.timedelta(days=-1 * int(tokens[0]))
                    datestr = str(now + time_change)
            else:
                # Month Day, maybe Year (if prev year)
                if len(tokens) > 2:
                    datestr = tokens[2]
                    tokens[1] = tokens[1][:-1]
                else:
                    today = date.today()
                    datestr = str(today.year)
                datestr += "-"
                datestr += monthStrToNum(tokens[0])
                datestr += "-"
                datestr += tokens[1]
            '''

            if len(datestr) > 10:
                datestr = datestr[:10]

            if datestr not in date_title_dict.keys():
                date_title_dict[datestr] = []
            date_title_dict[datestr].append(title.text)

            # print("Title:", title.text)
            # print("Date:", datestr)

        print(date_title_dict)
        return date_title_dict
    else:
        print('failed', response.status_code)


def combine_data(date_time_dict, stock_data):
    col_open_list = stock_data['Open'].tolist()
    col_close_list = stock_data['Close'].tolist()
    price_changes = [closePrice - openPrice for closePrice, openPrice in zip(col_close_list, col_open_list)]
    dates = stock_data['Date'].tolist()
    for date_i, price_change_i in zip(dates, price_changes):
        if date_i in date_time_dict.keys() and dates.count(date_i) == 1:
            date_time_dict[date_i] = (price_change_i, date_time_dict[date_i])

    date_time_dict = {k:date_time_dict[k] for k in date_time_dict.keys() if type(date_time_dict[k]) is not list}

    return date_time_dict


def score_data(data):
    sentiment_pipeline = pipeline("sentiment-analysis")
    price_change_list = []
    avg_score_list = []
    for key in data.keys():
        price_change, title_list = data[key]
        scores = sentiment_pipeline(title_list)

        for i in range(len(scores)):
            if scores[i]['label'] == 'NEGATIVE':
                scores[i] = -1 * scores[i]['score']
            else:
                scores[i] = scores[i]['score']

        # storing the average score
        # data[key] = (price_change, sum(scores) / len(scores))
        price_change_list.append(price_change)
        avg_score_list.append(sum(scores)/len(scores))
    return price_change_list, avg_score_list


def check_data_structure(data):
    for key in data.keys():
        if not isinstance(data[key], tuple):
            print(f"Data entry for key '{key}' is not a tuple: {data[key]}")
        elif len(data[key]) != 2:
            print(f"Data entry for key '{key}' does not have 2 elements: {data[key]}")
        elif not isinstance(data[key][1], list):
            print(f"Second element of data entry for key '{key}' is not a list: {data[key][1]}")
        else:
            print(f"Data entry for key '{key}' is correctly structured: {data[key]}")


def train(price_change_list, avg_score_list):
    x = np.array(price_change_list)
    y = np.array(avg_score_list)

    model = Sequential()
    model.add(Dense(2, input_dim=1, activation='sigmoid'))  # Hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=100, verbose=0)

    return model


if __name__ == "__main__":
    url_begin = 'https://news.google.com/search?q='
    url_end = '&hl=en-US&gl=US&ceid=US%3Aen'
    companies = ['LVMH', 'Tesla', 'Walmart']
    tickers = ['LVMHF', 'TSLA', 'WMT']
    for i in range(len(companies)):
        date_time_dict = scrape_google_news(url_begin + companies[i] + url_end)
        stock_data = get_stock_data(tickers[i], date_time_dict)
        # print(stock_data)

        # combine data into dict[date] = (price change, [article titles])
        data = combine_data(date_time_dict, stock_data)
        # print(data)

        # check_data_structure(data)

        # use hugging face sentiment to get score for each day-to-day
        price_change_list, avg_score_list = score_data(data)

        test_data = price_change_list[0]
        actual_output = avg_score_list[0]
        price_change_list = price_change_list[1:]
        avg_score_list = avg_score_list[1:]
        # print(data)

        # learn a neural network
        model = train(price_change_list, avg_score_list)

        # Basic test of model
        output = model.predict(list(test_data))
        print("Predicted output:")
        print(output)
        print("Actual output:")
        print(actual_output)

        break
