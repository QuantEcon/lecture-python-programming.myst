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

!pip install --upgrade polars wbgapi yfinance
```

## Overview

[Polars](https://pola.rs/) is a fast data manipulation library for Python written in Rust.

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
    * dealing with missing values, etc.

More sophisticated statistical functionality is left to other packages, such
as [statsmodels](https://www.statsmodels.org/) and [scikit-learn](https://scikit-learn.org/), which can work with polars DataFrames through their interoperability with pandas.

This lecture will provide a basic introduction to polars.

```{tip} 
*Why use Polars over pandas?* One reason is *performance*. As a general rule, it is recommended to have 5 to 10 times as much RAM as the size of the dataset to carry out operations in pandas, compared to 2 to 4 times  needed for Polars. In addition, Polars is between 10 and 100 times as fast as pandas for common operations. A great article comparing the Polars and pandas can be found [in this JetBrains blog post](https://blog.jetbrains.com/pycharm/2024/07/polars-vs-pandas/).
```

Throughout the lecture, we will assume that the following imports have taken place

```{code-cell} ipython3
import polars as pl
import pandas as pd
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

```{note}
You may notice the above series has no indices, unlike in [pd.Series](pandas:series). This is because Polars' is column centric and accessing data is predominantly managed through filtering and boolean masks. Here is [an interesting blog post discussing this in more detail](https://medium.com/data-science/understand-polars-lack-of-indexes-526ea75e413).
```

Polars `Series` are built on top of Apache Arrow arrays and support many similar
operations to Pandas `Series`.

(For interested readers, please see this extended reading on [Apache Arrow](https://www.datacamp.com/tutorial/apache-arrow))

```{code-cell} ipython3
s * 100
```

```{code-cell} ipython3
s.abs()
```

But `Series` provide more than basic arrays.

For example they have some additional (statistically oriented) methods

```{code-cell} ipython3
s.describe()
```

However the `pl.Series` object cannot be used in the same way as a `pd.Series` when pairing data with indices. 

For example, using a `pd.Series` you can do the following:

```{code-cell} ipython3
s = pd.Series(np.random.randn(4), name='daily returns')
s.index = ['AMZN', 'AAPL', 'MSFT', 'GOOG']
s
```

However, in Polars you will need to use the `DataFrame` object to do the same task.

This means you will use the `DataFrame` object more often when using polars if you
are interested in relationships between data 

Let's create a `pl.DataFrame` containing the equivalent data in the `pd.Series` .

```{code-cell} ipython3
df = pl.DataFrame({
    'company': ['AMZN', 'AAPL', 'MSFT', 'GOOG'],
    'daily returns': s.to_list()
})
df
```

To access specific values by company name, we can filter the DataFrame filtering on 
the `AMZN` ticker code and selecting the `daily returns`.

```{code-cell} ipython3
df.filter(pl.col('company') == 'AMZN').select('daily returns').item()
```

If we want to update `AMZN` return to 0, you can use the following chain of methods.


Here  `with_columns` is similar to `select` but adds columns to the same `DataFrame`

```{code-cell} ipython3
df = df.with_columns(
    pl.when(pl.col('company') == 'AMZN') # filter for AMZN in company column
    .then(0)                             # set values to 0
    .otherwise(pl.col('daily returns'))  # otherwise keep the original value
    .alias('daily returns')              # assign back to the column
)
df

You can check if a ticker code is in the company list

```{code-cell} ipython3
'AAPL' in df['company']
```

## DataFrames

```{index} single: Polars; DataFrames
```

While a `Series` is a single column of data, a `DataFrame` is several columns, one for each variable.

In essence, a `DataFrame` in polars is analogous to a (highly optimized) Excel spreadsheet.

Thus, it is a powerful tool for representing and analyzing data that are naturally organized into rows and columns.

Let's look at an example that reads data from the CSV file `pandas/data/test_pwt.csv`, 
which is taken from the [Penn World Tables](https://www.rug.nl/ggdc/productivity/pwt/pwt-releases/pwt-7.0).

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
URL = 'https://raw.githubusercontent.com/QuantEcon/lecture-python-programming/master/source/_static/lecture_specific/pandas/data/test_pwt.csv'
df = pl.read_csv(URL)
type(df)
```

Here is the content of `test_pwt.csv`

```{code-cell} ipython3
df
```

### Select Data by Position

In practice, one thing that we do all the time is to find, select and work with a 
subset of the data of our interests. 

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

In this case, `df.filter()` takes a boolean expression and only returns rows with the `True` values.

We can view this boolean mask as a table with the alias `meets_criteria`

```{code-cell} ipython3
df.select(
    pl.col('country'),  
    (pl.col('POP') >= 20000).alias('meets_criteria')
)
```

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

For example, we can use the conditioning to select the country with the largest 
household consumption - gdp share `cc`.

```{code-cell} ipython3
df.filter(pl.col('cc') == pl.col('cc').max())
```

When we only want to look at certain columns of a selected sub-dataframe, we can combine filter with select.

```{code-cell} ipython3
df.filter(
           (pl.col('cc') + pl.col('cg') >= 80) & (pl.col('POP') <= 20000)
           ).select(['country', 'year', 'POP'])
```

**Application: Subsetting Dataframe**

Real-world datasets can be very large.

It is sometimes desirable to work with a subset of data to enhance computational efficiency and reduce redundancy.

Let's imagine that we're only interested in the population (`POP`) and total GDP (`tcgdp`).

One way to strip the data frame `df` down to only these variables is to overwrite the `DataFrame` using the selection method described above

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

Here is an example using built-in functions to find the `max` value for each column

```{code-cell} ipython3
df.select([
    pl.col(['year', 'POP', 'XRAT', 'tcgdp', 'cc', 'cg']).max().name.suffix('_max')
])
```

For more complex operations, we can use `map_elements` (similar to pandas' apply):

```{code-cell} ipython3
df.select([
    pl.col('country'),
    pl.col('POP').map_elements(lambda x: x * 2).alias('POP_doubled')
])
```

However as you can see from the warning issued by Polars there is often a better way to achieve this using the Polars API.

```{code-cell} ipython3
df.select([
    pl.col('country'),
    (pl.col('POP') * 2).alias('POP_doubled')
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
    pl.when(pl.col('POP') >= 20000)          # when population >= 20000
    .then(pl.col('POP'))                     # keep the population value
    .otherwise(None)                         # otherwise set to null
    .alias('POP_filtered')                   # save results in POP_filtered
).select(['country', 'POP', 'POP_filtered']) # select the columns
```

**2.** We can modify specific values based on conditions

```{code-cell} ipython3
df_modified = df.with_columns(                     
    pl.when(pl.col('cg') == pl.col('cg').max())    # pick the largest cg value
    .then(None)                                    # set to null
    .otherwise(pl.col('cg'))                       # otherwise keep the value in the cg column
    .alias('cg')                                   # update the column with name cg
)
df_modified
```

**3.** We can use expressions to modify columns as a whole

```{code-cell} ipython3
df.with_columns([
    pl.when(pl.col('POP') <= 10000)          # when population is < 10,000
    .then(None)                              # set the value to null
    .otherwise(pl.col('POP'))                # otherwise keep the existing value
    .alias('POP'),                           # update the POP column
    (pl.col('XRAT') / 10).alias('XRAT')      # using the XRAT column, divide the value by 10 and update the column in-place
])
```

**4.** We can use in-built functions to modify all individual entries in specific columns by data type.

```{code-cell} ipython3
df.with_columns([
    pl.col(pl.Float64).round(2)   # round all Float64 columns to 2 decimal places
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

Here we fill `null` values with the column means

```{code-cell} ipython3
cols = ["cc", "tcgdp", "POP", "XRAT"]
df_with_nulls.with_columns([
    pl.col(cols).fill_null(pl.col(cols).mean()) 
])

Missing value imputation is a big area in data science involving various machine learning techniques.

There are also more [advanced tools](https://scikit-learn.org/stable/modules/impute.html) in python to impute missing values.

### Standardization and Visualization

Let's imagine that we're only interested in the population (`POP`) and total GDP (`tcgdp`).

One way to strip the data frame `df` down to only these variables is to overwrite the `DataFrame` using the selection method described above

```{code-cell} ipython3
df = df.select(['country', 'POP', 'tcgdp'])
df
```

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

Next, we're going to add a column showing real GDP per capita, multiplying by 1,000,000 as we go because total GDP is in millions.

```{note}
Polars (or Pandas) doesn't have a way of recording dimensional analysis units such as GDP represented in millions of dollars. This is left to the user to ensure they track their own units when undertaking analysis.
```

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

+++

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

```{note}
Many python packages will return Pandas DataFrames by default. In this example we use the `yfinance` package and convert the data to a polars DataFrame
```

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
    ticker_df = ticker_df.pivot(values='Close', index='Date', on='ticker')
    
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

# Convert to pandas for easier calculation, excluding Date column to avoid type errors
numeric_cols = [col for col in ticker.columns if col != 'Date']
first_pd = ticker.head(1).select(numeric_cols).to_pandas().iloc[0]
last_pd = ticker.tail(1).select(numeric_cols).to_pandas().iloc[0]

price_change = (last_pd - first_pd) / first_pd * 100
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
# Combine all yearly returns using concat and pivot approach
all_yearly_data = []

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
                          ((pl.col('last_price') - pl.col('first_price') + 1e-10) / (pl.col('first_price') + 1e-10)).alias('return')
                      )
                      .with_columns(pl.lit(indices_list[index_col]).alias('index_name'))
                      .select(['year', 'index_name', 'return']))
        
        all_yearly_data.append(yearly_data)

# Concatenate all data
combined_data = pl.concat(all_yearly_data)

# Pivot to get indices as columns
yearly_returns = combined_data.pivot(values='return', index='year', on='index_name')

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
