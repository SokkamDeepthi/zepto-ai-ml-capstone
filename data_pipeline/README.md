# Module 1 — Data Pipeline

## Overview

This module implements an end-to-end data pipeline that scrapes book data from a public website, cleans and transforms the data, converts prices from GBP to INR, stores the cleaned data in a normalized SQLite database, and executes SQL and pandas queries.

## Data Source

The data was scraped from:

https://books.toscrape.com/

The first 5 paginated catalogue pages were used.

The final dataset contains 100 books across multiple categories.

## Pipeline Steps

The pipeline follows these steps:

1. Scrape book data using `requests` and `BeautifulSoup`.
2. Extract title, price, star rating, availability, and category.
3. Clean and convert the scraped fields into proper data types.
4. Convert GBP prices to INR using the required fixed project rate.
5. Save the cleaned dataset as `books_cleaned.csv`.
6. Create a normalized SQLite database.
7. Store categories and books in separate tables.
8. Execute SQL queries including filtering, sorting, limiting, distinct values, and joins.
9. Read SQL results into pandas using `pd.read_sql()`.
10. Reproduce the JOIN result using `pd.merge()`.

## Data Cleaning

The following transformations were applied:

### Price

The GBP currency symbol was removed and the price was converted to a floating-point value in the `price_gbp` column.

### Rating

Text ratings such as `One`, `Two`, `Three`, `Four`, and `Five` were converted into integer values from 1 to 5.

### Availability

The availability text was converted into a boolean-style `in_stock` field.

### Currency Conversion

The project-defined fixed conversion rate was used:

`1 GBP = 105.50 INR`

The INR price was calculated as:

`price_inr = price_gbp * 105.50`

This is a fixed project baseline and does not use a live currency API.

## Database Design

SQLite was used as the relational database.

The database contains two normalized tables:

### categories

- `category_id` — Primary Key
- `category_name` — Unique category name

### books

- `book_id` — Primary Key
- `title`
- `price_gbp`
- `price_inr`
- `rating`
- `in_stock`
- `category_id` — Foreign Key referencing `categories.category_id`

The relationship is:

`categories.category_id` → `books.category_id`

This separates category information from book records and avoids storing the same category name repeatedly.

## Project Files

### `scrape_books.py`

Scrapes the book catalogue, cleans the data, performs the GBP to INR conversion, and saves the cleaned dataset.

### `books_cleaned.csv`

Contains the cleaned book dataset used as the input for the database pipeline.

### `database.py`

Creates the SQLite database and tables, inserts the cleaned data, and demonstrates a JOIN query.

### `books.db`

SQLite database containing the `categories` and `books` tables.

### `queries.py`

Contains SQL queries demonstrating:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- IN / BETWEEN
- JOIN

The query results are also read into pandas.

### `query_outputs.txt`

Contains the executed SQL query strings and their outputs.

## SQL and Pandas Analysis

At least five SQL queries were executed against the database.

The queries collectively demonstrate:

- Filtering records using `WHERE`
- Sorting records using `ORDER BY`
- Limiting results using `LIMIT`
- Finding unique categories using `DISTINCT`
- Filtering using `IN` or `BETWEEN`
- Combining books and categories using `JOIN`

Two query results were also loaded into pandas using `pd.read_sql()`.

The JOIN result was independently reproduced using `pd.merge()` on in-memory DataFrames.

## How to Run

### 1. Activate the virtual environment

From the project root:

```powershell
.venv\Scripts\Activate.ps1