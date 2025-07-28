---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(pl)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# {index}`Polars <single: Polars>`

```{index} single: Python; Polars
```

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install --upgrade polars
!pip install --upgrade wbgapi
!pip install --upgrade yfinance
```

## Overview

[Polars](https://pola.rs/) is a blazingly fast DataFrame library for Python, implemented in Rust.

Its popularity has been surging recently due to its excellent performance characteristics and intuitive API that's similar to pandas but with many improvements.

Just as [NumPy](http://www.numpy.org/) provides the basic array data type plus core array operations, polars

1. defines fundamental structures for working with data and
1. endows them with methods that facilitate operations such as
    * reading in data
    * adjusting indices
    * working with dates and time series
    * sorting, grouping, re-ordering and general data munging [^mung]
    * dealing with missing values, etc., etc.

More sophisticated statistical functionality is left to other packages, such
as [statsmodels](http://www.statsmodels.org/) and [scikit-learn](http://scikit-learn.org/), which can work with polars DataFrames.

This lecture will provide a basic introduction to polars.

Throughout the lecture, we will assume that the following imports have taken
place

```{code-cell} ipython3
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import requests
```

Two important data types defined by polars are `Series` and `DataFrame`.

You can think of a `Series` as a "column" of data, such as a collection of observations on a single variable.

A `DataFrame` is a two-dimensional object for storing related columns of data.

## Series

```{index} single: Polars; Series
```

Let's start with Series.

We begin by creating a series of four random observations

```{code-cell} ipython3
s = pl.Series(name='daily returns', values=np.random.randn(4))
s
```

Here you can imagine the indices `0, 1, 2, 3` as indexing four listed
companies, and the values being daily returns on their shares.

Polars `Series` support many operations similar to NumPy arrays

```{code-cell} ipython3
s * 100
```

```{code-cell} ipython3
s.abs()
```

But `Series` provide more than NumPy arrays.

They have additional (statistically oriented) methods

```{code-cell} ipython3
s.describe()
```

We can create a Series with custom names/indices using a DataFrame approach

```{code-cell} ipython3
df_series = pl.DataFrame({
    'company': ['AMZN', 'AAPL', 'MSFT', 'GOOG'],
    'daily_returns': s.to_list()
})
df_series
```

To access values by company name, we can filter the DataFrame

```{code-cell} ipython3
df_series.filter(pl.col('company') == 'AMZN')['daily_returns'].item()
```

We can update values using conditional logic

```{code-cell} ipython3
df_series = df_series.with_columns(
    pl.when(pl.col('company') == 'AMZN')
    .then(0.0)
    .otherwise(pl.col('daily_returns'))
    .alias('daily_returns')
)
df_series
```

We can check if a company exists in our data

```{code-cell} ipython3
'AAPL' in df_series['company'].to_list()
```

## DataFrames

```{index} single: Polars; DataFrames
```

While a `Series` is a single column of data, a `DataFrame` is several columns, one for each variable.

In essence, a `DataFrame` in polars is analogous to a (highly optimized) Excel spreadsheet.

Thus, it is a powerful tool for representing and analyzing data that are naturally organized into rows and columns, often with descriptive column names for individual variables.

Let's look at an example that reads data from the CSV file `pandas/data/test_pwt.csv`, which is taken from the [Penn World Tables](https://www.rug.nl/ggdc/productivity/pwt/pwt-releases/pwt-7.0).

The dataset contains the following indicators 

| Variable Name | Description |
| :-: | :-: |
| POP | Population (in thousands) |
| XRAT | Exchange Rate to US Dollar |                     
| tcgdp | Total PPP Converted GDP (in million international dollar) |
| cc | Consumption Share of PPP Converted GDP Per Capita (%) |
| cg | Government Consumption Share of PPP Converted GDP Per Capita (%) |

We'll read this in from a URL using the polars function `read_csv`.

```{code-cell} ipython3
df = pl.read_csv('https://raw.githubusercontent.com/QuantEcon/lecture-python-programming/master/source/_static/lecture_specific/pandas/data/test_pwt.csv')
type(df)
```

Here's the content of `test_pwt.csv`

```{code-cell} ipython3
df
```

### Select Data by Position

In practice, one thing that we do all the time is to find, select and work with a subset of the data of our interests. 

We can select particular rows using slice notation

```{code-cell} ipython3
df.slice(2, 3)  # Start at row 2, take 3 rows
```

To select columns, we can pass a list containing the names of the desired columns

```{code-cell} ipython3
df.select(['country', 'tcgdp'])
```

To select both rows and columns using integers, we can combine slicing and selection

```{code-cell} ipython3
df.slice(2, 3).select(df.columns[0:4])
```

### Select Data by Conditions

Instead of indexing rows and columns using integers and names, we can also obtain a sub-dataframe of our interests that satisfies certain (potentially complicated) conditions.

This section demonstrates various ways to do that.

The most straightforward way is with the `filter()` method.

```{code-cell} ipython3
df.filter(pl.col('POP') >= 20000)
```

We can combine multiple conditions using `&` (and) and `|` (or) operators

```{code-cell} ipython3
df.filter(
    (pl.col('country').is_in(['Argentina', 'India', 'South Africa'])) & 
    (pl.col('POP') > 40000)
)
```

We can also allow arithmetic operations between different columns.

```{code-cell} ipython3
df.filter((pl.col('cc') + pl.col('cg') >= 80) & (pl.col('POP') <= 20000))
```

For example, we can use the conditioning to select the country with the largest household consumption - gdp share `cc`.

```{code-cell} ipython3
df.filter(pl.col('cc') == pl.col('cc').max())
```

When we only want to look at certain columns of a selected sub-dataframe, we can chain the filter and select operations.

```{code-cell} ipython3
df.filter(
    (pl.col('cc') + pl.col('cg') >= 80) & (pl.col('POP') <= 20000)
).select(['country', 'year', 'POP'])
```

**Application: Subsetting Dataframe**

Real-world datasets can be [enormous](https://developers.google.com/machine-learning/data-prep/construct/collect/data-size-quality).

It is sometimes desirable to work with a subset of data to enhance computational efficiency and reduce redundancy.

Let's imagine that we're only interested in the population (`POP`) and total GDP (`tcgdp`).

One way to strip the data frame `df` down to only these variables is to use the select method

```{code-cell} ipython3
df_subset = df.select(['country', 'POP', 'tcgdp'])
df_subset
```

We can then save the smaller dataset for further analysis.

```{code-block} python3
:class: no-execute

df_subset.write_csv('pwt_subset.csv')
```

### Apply-like Operations with Expressions

Polars uses a powerful expression system instead of traditional apply methods. 

Expressions are more efficient and often more readable than apply functions.

Here's how to get the maximum value for each numeric column

```{code-cell} ipython3
df.select([
    pl.col('year', 'POP', 'XRAT', 'tcgdp', 'cc', 'cg').max()
])
```

We can create complex expressions using `when().then().otherwise()` logic

```{code-cell} ipython3
df.with_columns(
    pl.when(
        (pl.col('country').is_in(['Argentina', 'India', 'South Africa'])) & 
        (pl.col('POP') > 40000)
    )
    .then(pl.lit('Large Pop Country'))
    .when(pl.col('POP') < 20000)
    .then(pl.lit('Small Pop Country'))
    .otherwise(pl.lit('Medium Pop Country'))
    .alias('pop_category')
)
```

### Make Changes in DataFrames

The ability to make changes in dataframes is important to generate a clean dataset for future analysis.

**1.** We can use conditional expressions to modify values

```{code-cell} ipython3
df.with_columns(
    pl.when(pl.col('POP') < 20000)
    .then(None)
    .otherwise(pl.col('POP'))
    .alias('POP')
)
```

**2.** We can modify specific values based on conditions

```{code-cell} ipython3
df.with_columns(
    pl.when(pl.col('cg') == pl.col('cg').max())
    .then(None)
    .otherwise(pl.col('cg'))
    .alias('cg')
)
```

**3.** We can create new columns or modify existing ones with complex logic

```{code-cell} ipython3
df.with_columns([
    pl.when(pl.col('POP') <= 10000)
    .then(None)
    .otherwise(pl.col('POP'))
    .alias('POP'),
    
    (pl.col('XRAT') / 10).alias('XRAT')
])
```

**4.** We can apply functions to all columns using `map_elements()` for more complex transformations

```{code-cell} ipython3
# Round all numeric columns to 2 decimal places
numeric_cols = df.select(pl.col(pl.Float64, pl.Int64)).columns
df.with_columns([
    pl.col(col).round(2) for col in numeric_cols
])
```

**Application: Missing Value Imputation**

Replacing missing values is an important step in data munging. 

Let's randomly insert some NaN values

```{code-cell} ipython3
# Create a copy and introduce some NaN values
df_with_nan = df.clone()
for idx, col_idx in zip([0, 3, 5, 6], [3, 4, 6, 2]):
    col_name = df_with_nan.columns[col_idx]
    df_with_nan = df_with_nan.with_row_index().with_columns(
        pl.when(pl.col('index') == idx)
        .then(None)
        .otherwise(pl.col(col_name))
        .alias(col_name)
    ).drop('index')

df_with_nan
```

We can replace all missing values with 0

```{code-cell} ipython3
df_with_nan.fill_null(0)
```

Polars also provides convenient methods to replace missing values.

For example, single imputation using variable means can be easily done

```{code-cell} ipython3
# Fill missing values with column means for numeric columns
numeric_cols = ['POP', 'XRAT', 'tcgdp', 'cc', 'cg']
df_filled = df_with_nan.with_columns([
    pl.col(col).fill_null(pl.col(col).mean()) for col in numeric_cols
])
df_filled
```

Missing value imputation is a big area in data science involving various machine learning techniques.

There are also more [advanced tools](https://scikit-learn.org/stable/modules/impute.html) in python to impute missing values.

### Standardization and Visualization

Let's imagine that we're only interested in the population (`POP`) and total GDP (`tcgdp`).

One way to strip the data frame `df` down to only these variables is to use the select method

```{code-cell} ipython3
df = df.select(['country', 'POP', 'tcgdp'])
df
```

Let's give the columns slightly better names

```{code-cell} ipython3
df = df.rename({'POP': 'population', 'tcgdp': 'total GDP'})
df
```

The `population` variable is in thousands, let's revert to single units

```{code-cell} ipython3
df = df.with_columns((pl.col('population') * 1e3).alias('population'))
df
```

Next, we're going to add a column showing real GDP per capita, multiplying by 1,000,000 as we go because total GDP is in millions

```{code-cell} ipython3
df = df.with_columns(
    (pl.col('total GDP') * 1e6 / pl.col('population')).alias('GDP percap')
)
df
```

For visualization, we need to convert to pandas or use polars' plotting capabilities.
Let's convert to pandas for matplotlib compatibility

```{code-cell} ipython3
df_pandas = df.to_pandas().set_index('country')
ax = df_pandas['GDP percap'].plot(kind='bar')
ax.set_xlabel('country', fontsize=12)
ax.set_ylabel('GDP per capita', fontsize=12)
plt.show()
```

At the moment the data frame is ordered alphabetically on the countries---let's change it to GDP per capita

```{code-cell} ipython3
df = df.sort('GDP percap', descending=True)
df
```

Plotting as before now yields

```{code-cell} ipython3
df_pandas = df.to_pandas().set_index('country')
ax = df_pandas['GDP percap'].plot(kind='bar')
ax.set_xlabel('country', fontsize=12)
ax.set_ylabel('GDP per capita', fontsize=12)
plt.show()
```

## On-Line Data Sources

```{index} single: Data Sources
```

Python makes it straightforward to query online databases programmatically.

An important database for economists is [FRED](https://research.stlouisfed.org/fred2/) --- a vast collection of time series data maintained by the St. Louis Fed.

For example, suppose that we are interested in the [unemployment rate](https://research.stlouisfed.org/fred2/series/UNRATE).

### Accessing Data with {index}`requests <single: requests>`

```{index} single: Python; requests
```

One option is to use [requests](https://requests.readthedocs.io/en/master/), a standard Python library for requesting data over the Internet.

To begin, try the following code on your computer

```{code-cell} ipython3
r = requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=UNRATE&scale=left&cosd=1948-01-01&coed=2024-06-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2024-07-29&revision_date=2024-07-29&nd=1948-01-01')
```

If there's no error message, then the call has succeeded.

```{code-cell} ipython3
url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=UNRATE&scale=left&cosd=1948-01-01&coed=2024-06-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2024-07-29&revision_date=2024-07-29&nd=1948-01-01'
source = requests.get(url).content.decode().split("\n")
source[0]
```

```{code-cell} ipython3
source[1]
```

```{code-cell} ipython3
source[2]
```

We could now write some additional code to parse this text and store it as an array.

But this is unnecessary --- polars' `read_csv` function can handle the task for us.

We use `try_parse_dates=True` so that polars recognizes our dates column, allowing for simple date filtering

```{code-cell} ipython3
data = pl.read_csv(url, try_parse_dates=True)
```

The data has been read into a polars DataFrame called `data` that we can now manipulate in the usual way

```{code-cell} ipython3
type(data)
```

```{code-cell} ipython3
data.head()  # A useful method to get a quick look at a data frame
```

```{code-cell} ipython3
data.describe()  # Your output might differ slightly
```

We can also plot the unemployment rate from 2006 to 2012 as follows

```{code-cell} ipython3
# Filter data for the date range and convert to pandas for plotting
plot_data = data.filter(
    (pl.col('DATE') >= pl.date(2006, 1, 1)) & 
    (pl.col('DATE') <= pl.date(2012, 12, 31))
).to_pandas().set_index('DATE')

ax = plot_data.plot(title='US Unemployment Rate', legend=False)
ax.set_xlabel('year', fontsize=12)
ax.set_ylabel('%', fontsize=12)
plt.show()
```

Note that polars offers many other file type alternatives.

Polars has [a wide variety](https://pola-rs.github.io/polars/py-polars/html/reference/io.html) of functions that we can use to read excel, json, parquet or plug straight into a database server.

### Using {index}`wbgapi <single: wbgapi>` and {index}`yfinance <single: yfinance>` to Access Data

The [wbgapi](https://pypi.org/project/wbgapi/) python library can be used to fetch data from the many databases published by the World Bank.

We will also use [yfinance](https://pypi.org/project/yfinance/) to fetch data from Yahoo finance
in the exercises.

For now let's work through one example of downloading and plotting data --- this
time from the World Bank.

The World Bank [collects and organizes data](http://data.worldbank.org/indicator) on a huge range of indicators.

For example, [here's](http://data.worldbank.org/indicator/GC.DOD.TOTL.GD.ZS/countries) some data on government debt as a ratio to GDP.

The next code example fetches the data for you and plots time series for the US and Australia

```{code-cell} ipython3
import wbgapi as wb
wb.series.info('GC.DOD.TOTL.GD.ZS')
```

```{code-cell} ipython3
govt_debt_pd = wb.data.DataFrame('GC.DOD.TOTL.GD.ZS', economy=['USA','AUS'], time=range(2005,2016))
# Convert to polars and then transpose for plotting
govt_debt = pl.from_pandas(govt_debt_pd.T)
govt_debt
```

```{code-cell} ipython3
# Convert back to pandas for plotting
govt_debt.to_pandas().plot(xlabel='year', ylabel='Government debt (% of GDP)');
```

## Exercises

```{exercise-start}
:label: pl_ex1
```

With these imports:

```{code-cell} ipython3
import datetime as dt
import yfinance as yf
import polars as pl
```

Write a program to calculate the percentage price change over 2021 for the following shares:

```{code-cell} ipython3
ticker_list = {'INTC': 'Intel',
               'MSFT': 'Microsoft',
               'IBM': 'IBM',
               'BHP': 'BHP',
               'TM': 'Toyota',
               'AAPL': 'Apple',
               'AMZN': 'Amazon',
               'C': 'Citigroup',
               'QCOM': 'Qualcomm',
               'KO': 'Coca-Cola',
               'GOOG': 'Google'}
```

Here's the first part of the program adapted for polars

```{code-cell} ipython3
def read_data(ticker_list,
          start=dt.datetime(2021, 1, 1),
          end=dt.datetime(2021, 12, 31)):
    """
    This function reads in closing price data from Yahoo
    for each tick in the ticker_list and returns a polars DataFrame.
    """
    ticker_data = []

    for tick in ticker_list:
        stock = yf.Ticker(tick)
        prices = stock.history(start=start, end=end)

        # Reset index to make Date a column
        prices = prices.reset_index()
        prices['Date'] = prices['Date'].dt.date
        
        # Convert to polars and select relevant columns
        pl_prices = pl.from_pandas(prices[['Date', 'Close']])
        pl_prices = pl_prices.with_columns(pl.lit(tick).alias('Ticker'))
        
        ticker_data.append(pl_prices)

    # Concatenate all dataframes
    combined = pl.concat(ticker_data)
    
    # Pivot to have tickers as columns
    ticker_df = combined.pivot(index='Date', columns='Ticker', values='Close')
    
    return ticker_df

ticker = read_data(ticker_list)
```

Complete the program to plot the result as a bar graph.

```{exercise-end}
```

```{solution-start} pl_ex1
:class: dropdown
```

There are a few ways to approach this problem using Polars to calculate
the percentage change.

First, you can extract the first and last rows and perform the calculation:

```{code-cell} ipython3
# Get first and last prices
first_prices = ticker.slice(0, 1).drop('Date').transpose(include_header=True)
last_prices = ticker.slice(-1, 1).drop('Date').transpose(include_header=True)

# Calculate percentage change
price_change = (
    pl.concat([first_prices, last_prices], how='horizontal')
    .with_columns([
        ((pl.col('column_1') - pl.col('column_0')) / pl.col('column_0') * 100).alias('pct_change')
    ])
    .with_row_index()
    .with_columns(pl.col('index').map_elements(lambda x: ticker.columns[x+1]).alias('ticker'))
    .select(['ticker', 'pct_change'])
)

price_change
```

Then to plot the chart:

```{code-cell} ipython3
# Sort by percentage change and rename tickers
price_change_sorted = price_change.sort('pct_change').with_columns(
    pl.col('ticker').map_elements(lambda x: ticker_list[x]).alias('company_name')
)

# Convert to pandas for plotting
price_change_pd = price_change_sorted.to_pandas().set_index('company_name')

fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlabel('stock', fontsize=12)
ax.set_ylabel('percentage change in price', fontsize=12)
price_change_pd['pct_change'].plot(kind='bar', ax=ax)
plt.show()
```

```{solution-end}
```

```{exercise-start}
:label: pl_ex2
```

Using the method `read_data` introduced in {ref}`pl_ex1`, write a program to obtain year-on-year percentage change for the following indices:

```{code-cell} ipython3
indices_list = {'^GSPC': 'S&P 500',
               '^IXIC': 'NASDAQ',
               '^DJI': 'Dow Jones',
               '^N225': 'Nikkei'}
```

Complete the program to show summary statistics and plot the result as a time series graph.

```{exercise-end}
```

```{solution-start} pl_ex2
:class: dropdown
```

Following the work you did in {ref}`pl_ex1`, you can query the data using `read_data` by updating the start and end dates accordingly.

```{code-cell} ipython3
indices_data = read_data(
        indices_list,
        start=dt.datetime(1971, 1, 1),  #Common Start Date
        end=dt.datetime(2021, 12, 31)
)
```

Then, calculate the yearly returns using polars operations:

```{code-cell} ipython3
# Add year column and calculate yearly returns
yearly_returns_list = []

for index, name in indices_list.items():
    if index in indices_data.columns:
        yearly_data = (
            indices_data
            .select(['Date', index])
            .with_columns(pl.col('Date').dt.year().alias('Year'))
            .group_by('Year')
            .agg([
                pl.col(index).first().alias('first_price'),
                pl.col(index).last().alias('last_price')
            ])
            .with_columns(
                ((pl.col('last_price') - pl.col('first_price')) / pl.col('first_price')).alias(name)
            )
            .select(['Year', name])
        )
        yearly_returns_list.append(yearly_data)

# Join all yearly returns
yearly_returns = yearly_returns_list[0]
for df in yearly_returns_list[1:]:
    yearly_returns = yearly_returns.join(df, on='Year', how='outer')

yearly_returns = yearly_returns.sort('Year')
yearly_returns
```

Next, you can obtain summary statistics:

```{code-cell} ipython3
yearly_returns.select(pl.col(pl.Float64)).describe()
```

Then, to plot the chart:

```{code-cell} ipython3
# Convert to pandas for plotting
yearly_returns_pd = yearly_returns.to_pandas().set_index('Year')

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for iter_, ax in enumerate(axes.flatten()):
    if iter_ < len(yearly_returns_pd.columns):
        index_name = yearly_returns_pd.columns[iter_]
        ax.plot(yearly_returns_pd.index, yearly_returns_pd[index_name])
        ax.set_ylabel("percent change", fontsize=12)
        ax.set_title(index_name)

plt.tight_layout()
```

```{solution-end}
```

[^mung]: Wikipedia defines munging as cleaning data from one raw form into a structured, purged one.
