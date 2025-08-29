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

[Polars](https://pola.rs/) is a lightning-fast data manipulation library for Python written in Rust.

Polars has gained significant popularity in recent years due to its superior performance
compared to traditional data analysis tools, making it an excellent choice for modern
data science and machine learning workflows.

Polars is designed with performance and memory efficiency in mind, leveraging:

* Arrow's columnar memory format for fast data access
* Lazy evaluation to optimize query execution
* Parallel processing for enhanced performance
* Expressive API similar to pandas but with better performance characteristics

Just as [NumPy](https://numpy.org/) provides the basic array data type plus core array operations, polars

1. defines fundamental structures for working with data and
1. endows them with methods that facilitate operations such as
    * reading in data
    * adjusting indices
    * working with dates and time series
    * sorting, grouping, re-ordering and general data munging [^mung]
    * dealing with missing values, etc., etc.

More sophisticated statistical functionality is left to other packages, such
as [statsmodels](https://www.statsmodels.org/) and [scikit-learn](https://scikit-learn.org/), which can work with polars DataFrames through their interoperability with pandas.

This lecture will provide a basic introduction to polars.

Throughout the lecture, we will assume that the following imports have taken
place

```{code-cell} ipython3
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import requests
```

Two important data types defined by polars are  `Series` and `DataFrame`.

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

Polars `Series` are built on top of Apache Arrow arrays and support many similar
operations

```{code-cell} ipython3
s * 100
```

```{code-cell} ipython3
s.abs()
```

But `Series` provide more than basic arrays.

Not only do they have some additional (statistically oriented) methods

```{code-cell} ipython3
s.describe()
```

But they can also be used with custom indices

```{code-cell} ipython3
# Create a new series with custom index using a DataFrame
df_temp = pl.DataFrame({
    'company': ['AMZN', 'AAPL', 'MSFT', 'GOOG'],
    'daily returns': s.to_list()
})
df_temp
```

To access specific values by company name, we can filter the DataFrame

```{code-cell} ipython3
# Get AMZN's return
df_temp.filter(pl.col('company') == 'AMZN').select('daily returns').item()
```

```{code-cell} ipython3
# Update AMZN's return to 0
df_temp = df_temp.with_columns(
    pl.when(pl.col('company') == 'AMZN')
    .then(0)
    .otherwise(pl.col('daily returns'))
    .alias('daily returns')
)
df_temp
```

```{code-cell} ipython3
# Check if AAPL is in the companies
'AAPL' in df_temp.get_column('company').to_list()
```

## DataFrames

```{index} single: Polars; DataFrames
```

While a `Series` is a single column of data, a `DataFrame` is several columns, one for each variable.

In essence, a `DataFrame` in polars is analogous to a (highly optimized) Excel spreadsheet.

Thus, it is a powerful tool for representing and analyzing data that are naturally organized into rows and columns, often with descriptive indexes for individual rows and individual columns.

Let's look at an example that reads data from the CSV file `pandas/data/test_pwt.csv`, which is taken from the [Penn World Tables](https://www.rug.nl/ggdc/productivity/pwt/pwt-releases/pwt-7.0).

The dataset contains the following indicators 

| Variable Name | Description |
| :-: | :-: |
| POP | Population (in thousands) |
| XRAT | Exchange Rate to US Dollar |                     
| tcgdp | Total PPP Converted GDP (in million international dollar) |
| cc | Consumption Share of PPP Converted GDP Per Capita (%) |
| cg | Government Consumption Share of PPP Converted GDP Per Capita (%) |


We'll read this in from a URL using the `polars` function `read_csv`.

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

We can select particular rows using array slicing notation

```{code-cell} ipython3
df[2:5]
```

To select columns, we can pass a list containing the names of the desired columns

```{code-cell} ipython3
df.select(['country', 'tcgdp'])
```

To select both rows and columns using integers, we can combine slicing with column selection

```{code-cell} ipython3
df[2:5].select(df.columns[0:4])
```

To select rows and columns using a mixture of integers and labels, we can use more complex selection

```{code-cell} ipython3
df[2:5].select(['country', 'tcgdp'])
```

### Select Data by Conditions

Instead of indexing rows and columns using integers and names, we can also obtain a sub-dataframe of our interests that satisfies certain (potentially complicated) conditions.

This section demonstrates various ways to do that.

The most straightforward way is with the `filter` method.

```{code-cell} ipython3
df.filter(pl.col('POP') >= 20000)
```

To understand what is going on here, notice that `pl.col('POP') >= 20000` creates a boolean expression.

```{code-cell} ipython3
df.select(pl.col('POP') >= 20000)
```

In this case, `df.filter()` takes a boolean expression and only returns rows with the `True` values.

Take one more example,

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

When we only want to look at certain columns of a selected sub-dataframe, we can combine filter with select.

```{code-cell} ipython3
df.filter((pl.col('cc') + pl.col('cg') >= 80) & (pl.col('POP') <= 20000)).select(['country', 'year', 'POP'])
```

**Application: Subsetting Dataframe**

Real-world datasets can be [enormous](https://developers.google.com/machine-learning/crash-course/overfitting).

It is sometimes desirable to work with a subset of data to enhance computational efficiency and reduce redundancy.

Let's imagine that we're only interested in the population (`POP`) and total GDP (`tcgdp`).

One way to strip the data frame `df` down to only these variables is to overwrite the dataframe using the selection method described above

```{code-cell} ipython3
df_subset = df.select(['country', 'POP', 'tcgdp'])
df_subset
```

We can then save the smaller dataset for further analysis.

```{code-block} python3
:class: no-execute

df_subset.write_csv('pwt_subset.csv')
```

### Apply and Map Operations

Polars provides powerful methods for applying functions to data. 

Instead of pandas' `apply` method, polars uses expressions within `select`, `with_columns`, or `filter` methods.

Here is an example using built-in functions

```{code-cell} ipython3
df.select([
    pl.col(['year', 'POP', 'XRAT', 'tcgdp', 'cc', 'cg']).max().name.suffix('_max')
])
```

This line of code applies the `max` function to all selected columns.

For more complex operations, we can use `map_elements` (similar to pandas' apply):

```{code-cell} ipython3
# A trivial example using map_elements
df.with_row_index().select([
    pl.col('index'),
    pl.col('country'),
    pl.col('POP').map_elements(lambda x: x * 2, return_dtype=pl.Float64).alias('POP_doubled')
])
```

We can use complex filtering conditions with boolean logic:

```{code-cell} ipython3
complex_condition = (
    pl.when(pl.col('country').is_in(['Argentina', 'India', 'South Africa']))
    .then(pl.col('POP') > 40000)
    .otherwise(pl.col('POP') < 20000)
)

df.filter(complex_condition).select(['country', 'year', 'POP', 'XRAT', 'tcgdp'])
```

### Make Changes in DataFrames

The ability to make changes in dataframes is important to generate a clean dataset for future analysis.

**1.** We can use conditional logic to "keep" certain values and replace others

```{code-cell} ipython3
df.with_columns(
    pl.when(pl.col('POP') >= 20000)
    .then(pl.col('POP'))
    .otherwise(None)
    .alias('POP_filtered')
).select(['country', 'POP', 'POP_filtered'])
```

**2.** We can modify specific values based on conditions

```{code-cell} ipython3
df_modified = df.with_columns(
    pl.when(pl.col('cg') == pl.col('cg').max())
    .then(None)
    .otherwise(pl.col('cg'))
    .alias('cg')
)
df_modified
```

**3.** We can use expressions to modify columns as a whole

```{code-cell} ipython3
df.with_columns([
    pl.when(pl.col('POP') <= 10000).then(None).otherwise(pl.col('POP')).alias('POP'),
    (pl.col('XRAT') / 10).alias('XRAT')
])
```

**4.** We can use `map_elements` to modify all individual entries in specific columns.

```{code-cell} ipython3
# Round all decimal numbers to 2 decimal places in numeric columns
df.with_columns([
    pl.col(pl.Float64).round(2)
])
```

**Application: Missing Value Imputation**

Replacing missing values is an important step in data munging. 

Let's randomly insert some null values

```{code-cell} ipython3
# Create a copy with some null values
df_with_nulls = df.clone()

# Set some specific positions to null
indices_to_null = [(0, 'XRAT'), (3, 'cc'), (5, 'tcgdp'), (6, 'POP')]

for row_idx, col_name in indices_to_null:
    df_with_nulls = df_with_nulls.with_columns(
        pl.when(pl.int_range(pl.len()) == row_idx)
        .then(None)
        .otherwise(pl.col(col_name))
        .alias(col_name)
    )

df_with_nulls
```

We can replace all missing values with 0

```{code-cell} ipython3
df_with_nulls.fill_null(0)
```

Polars also provides us with convenient methods to replace missing values.

For example, we can use forward fill, backward fill, or interpolation

```{code-cell} ipython3
# Fill with column means for numeric columns
df_filled = df_with_nulls.with_columns([
    pl.col(pl.Float64, pl.Int64).fill_null(pl.col(pl.Float64, pl.Int64).mean())
])
df_filled
```

Missing value imputation is a big area in data science involving various machine learning techniques.

There are also more [advanced tools](https://scikit-learn.org/stable/modules/impute.html) in python to impute missing values.

### Standardization and Visualization

Let's imagine that we're only interested in the population (`POP`) and total GDP (`tcgdp`).

One way to strip the data frame `df` down to only these variables is to overwrite the dataframe using the selection method described above

```{code-cell} ipython3
df = df.select(['country', 'POP', 'tcgdp'])
df
```

Here the index `0, 1,..., 7` is redundant because we can use the country names as an index.

While polars doesn't have a traditional index like pandas, we can work with country names directly

```{code-cell} ipython3
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

One of the nice things about polars `DataFrame` and `Series` objects is that they can be easily converted to pandas for visualization through Matplotlib.

For example, we can easily generate a bar plot of GDP per capita

```{code-cell} ipython3
# Convert to pandas for plotting
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
# Convert to pandas for plotting
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

An important database for economists is [FRED](https://fred.stlouisfed.org/) --- a vast collection of time series data maintained by the St. Louis Fed.

For example, suppose that we are interested in the [unemployment rate](https://fred.stlouisfed.org/series/UNRATE).

(To download the data as a csv, click on the top right `Download` and select the `CSV (data)` option).

Alternatively, we can access the CSV file from within a Python program.

This can be done with a variety of methods.

We start with a relatively low-level method and then return to polars.

### Accessing Data with {index}`requests <single: requests>`

```{index} single: Python; requests
```

One option is to use [requests](https://requests.readthedocs.io/en/latest/), a standard Python library for requesting data over the Internet.

To begin, try the following code on your computer

```{code-cell} ipython3
r = requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=UNRATE&scale=left&cosd=1948-01-01&coed=2024-06-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Monthly&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2024-07-29&revision_date=2024-07-29&nd=1948-01-01')
```

If there's no error message, then the call has succeeded.

If you do get an error, then there are two likely causes

1. You are not connected to the Internet --- hopefully, this isn't the case.
1. Your machine is accessing the Internet through a proxy server, and Python isn't aware of this.

In the second case, you can either

* switch to another machine
* solve your proxy problem by reading [the documentation](https://requests.readthedocs.io/en/latest/)

Assuming that all is working, you can now proceed to use the `source` object returned by the call `requests.get('https://research.stlouisfed.org/fred2/series/UNRATE/downloaddata/UNRATE.csv')`

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
# Filter data for the specified date range and convert to pandas for plotting
filtered_data = data.filter(
    (pl.col('observation_date') >= pl.date(2006, 1, 1)) & 
    (pl.col('observation_date') <= pl.date(2012, 12, 31))
).to_pandas().set_index('observation_date')

ax = filtered_data.plot(title='US Unemployment Rate', legend=False)
ax.set_xlabel('year', fontsize=12)
ax.set_ylabel('%', fontsize=12)
plt.show()
```

Note that polars offers many other file type alternatives.

Polars has [a wide variety](https://docs.pola.rs/user-guide/io/) of methods that we can use to read excel, json, parquet or plug straight into a database server.

### Using {index}`wbgapi <single: wbgapi>` and {index}`yfinance <single: yfinance>` to Access Data

The [wbgapi](https://pypi.org/project/wbgapi/) python library can be used to fetch data from the many databases published by the World Bank.

```{note}
You can find some useful information about the [wbgapi](https://pypi.org/project/wbgapi/) package in this [world bank blog post](https://blogs.worldbank.org/en/opendata/introducing-wbgapi-new-python-package-accessing-world-bank-data), in addition to this [tutorial](https://github.com/tgherzog/wbgapi/blob/master/examples/wbgapi-quickstart.ipynb)
```

We will also use [yfinance](https://pypi.org/project/yfinance/) to fetch data from Yahoo finance
in the exercises.

For now let's work through one example of downloading and plotting data --- this
time from the World Bank.

The World Bank [collects and organizes data](https://data.worldbank.org/indicator) on a huge range of indicators.

For example, [here's](https://data.worldbank.org/indicator/GC.DOD.TOTL.GD.ZS) some data on government debt as a ratio to GDP.

The next code example fetches the data for you and plots time series for the US and Australia

```{code-cell} ipython3
import wbgapi as wb
wb.series.info('GC.DOD.TOTL.GD.ZS')
```

```{code-cell} ipython3
govt_debt_pandas = wb.data.DataFrame('GC.DOD.TOTL.GD.ZS', economy=['USA','AUS'], time=range(2005,2016))
govt_debt_pandas = govt_debt_pandas.T    # move years from columns to rows for plotting

# Convert to polars
govt_debt = pl.from_pandas(govt_debt_pandas.reset_index())
```

```{code-cell} ipython3
# For plotting, convert back to pandas format
govt_debt.to_pandas().set_index('index').plot(xlabel='year', ylabel='Government debt (% of GDP)');
```

## Exercises

```{exercise-start}
:label: pl_ex1
```

With these imports:

```{code-cell} ipython3
import datetime as dt
import yfinance as yf
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

Here's the first part of the program

```{code-cell} ipython3
def read_data(ticker_list,
          start=dt.datetime(2021, 1, 1),
          end=dt.datetime(2021, 12, 31)):
    """
    This function reads in closing price data from Yahoo
    for each tick in the ticker_list.
    """
    
    all_data = []
    
    for tick in ticker_list:
        stock = yf.Ticker(tick)
        prices = stock.history(start=start, end=end)
        
        # Convert to polars DataFrame
        df = pl.from_pandas(prices.reset_index())
        df = df.with_columns([
            pl.col('Date').cast(pl.Date),
            pl.lit(tick).alias('ticker')
        ]).select(['Date', 'ticker', 'Close'])
        
        all_data.append(df)
    
    # Combine all data
    ticker_df = pl.concat(all_data)
    
    # Pivot to have tickers as columns
    ticker_df = ticker_df.pivot(values='Close', index='Date', columns='ticker')
    
    return ticker_df

ticker = read_data(ticker_list)
```

Complete the program to plot the result as a bar graph like this one:

```{image} /_static/lecture_specific/pandas/pandas_share_prices.png
:scale: 80
:align: center
```

```{exercise-end}
```

```{solution-start} pl_ex1
:class: dropdown
```

There are a few ways to approach this problem using Polars to calculate
the percentage change.

First, you can extract the data and perform the calculation such as:

```{code-cell} ipython3
# Get first and last prices for each ticker
first_prices = ticker[0]  # First row
last_prices = ticker[-1]  # Last row

# Convert to pandas for easier calculation
first_pd = ticker.head(1).to_pandas().iloc[0]
last_pd = ticker.tail(1).to_pandas().iloc[0]

price_change = (last_pd - first_pd) / first_pd * 100
price_change = price_change.dropna()  # Remove Date column
price_change
```

Alternatively you can use polars expressions to calculate percentage change:

```{code-cell} ipython3
# Calculate percentage change using polars
change_df = ticker.select([
    ((pl.col(col).last() - pl.col(col).first()) / pl.col(col).first() * 100).alias(f'{col}_pct_change')
    for col in ticker.columns if col != 'Date'
])

# Convert to series for plotting
price_change = change_df.to_pandas().iloc[0]
price_change.index = [col.replace('_pct_change', '') for col in price_change.index]
price_change
```

Then to plot the chart

```{code-cell} ipython3
price_change.sort_values(inplace=True)
price_change.rename(index=ticker_list, inplace=True)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlabel('stock', fontsize=12)
ax.set_ylabel('percentage change in price', fontsize=12)
price_change.plot(kind='bar', ax=ax)
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

Complete the program to show summary statistics and plot the result as a time series graph like this one:

```{image} /_static/lecture_specific/pandas/pandas_indices_pctchange.png
:scale: 80
:align: center
```

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

Then, calculate the yearly returns using polars:

```{code-cell} ipython3
# Add year column and calculate yearly returns
yearly_returns_list = []

for index_col in indices_data.columns:
    if index_col != 'Date':
        yearly_data = (indices_data
                      .with_columns(pl.col('Date').dt.year().alias('year'))
                      .group_by('year')
                      .agg([
                          pl.col(index_col).first().alias('first_price'),
                          pl.col(index_col).last().alias('last_price')
                      ])
                      .with_columns(
                          ((pl.col('last_price') - pl.col('first_price')) / pl.col('first_price')).alias(indices_list[index_col])
                      )
                      .select(['year', indices_list[index_col]]))
        
        yearly_returns_list.append(yearly_data)

# Join all yearly returns
yearly_returns = yearly_returns_list[0]
for df in yearly_returns_list[1:]:
    yearly_returns = yearly_returns.join(df, on='year', how='outer')

yearly_returns
```

Next, you can obtain summary statistics by using the method `describe`.

```{code-cell} ipython3
yearly_returns.select(pl.exclude('year')).describe()
```

Then, to plot the chart

```{code-cell} ipython3
# Convert to pandas for plotting
yearly_returns_pd = yearly_returns.to_pandas().set_index('year')

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for iter_, ax in enumerate(axes.flatten()):            # Flatten 2-D array to 1-D array
    if iter_ < len(yearly_returns_pd.columns):
        index_name = yearly_returns_pd.columns[iter_]         # Get index name per iteration
        ax.plot(yearly_returns_pd[index_name])                # Plot pct change of yearly returns per index
        ax.set_ylabel("percent change", fontsize = 12)
        ax.set_title(index_name)

plt.tight_layout()
```

```{solution-end}
```

[^mung]: Wikipedia defines munging as cleaning data from one raw form into a structured, purged one.