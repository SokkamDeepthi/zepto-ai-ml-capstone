import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re

all_books = []

for page in range(1, 6):

    if page == 1:
        url = "https://books.toscrape.com/"
    else:
        url = f"https://books.toscrape.com/catalogue/page-{page}.html"

    response = requests.get(url)

    print("Page:", page, "| Status:", response.status_code)

    soup = BeautifulSoup(response.text, "html.parser")

    books = soup.select("article.product_pod")

    for book in books:

        title = book.h3.a["title"]

        price = book.select_one(".price_color").get_text(strip=True)

        rating = book.select_one(".star-rating")["class"][1]

        availability = book.select_one(
            ".availability"
        ).get_text(" ", strip=True)

        book_url = urljoin(url, book.h3.a["href"])

        detail_response = requests.get(book_url)

        detail_soup = BeautifulSoup(
            detail_response.text,
            "html.parser"
        )

        breadcrumb_items = detail_soup.select(
            "ul.breadcrumb li"
        )

        if len(breadcrumb_items) >= 3:
            category = breadcrumb_items[2].get_text(strip=True)
        else:
            category = "Unknown"

        all_books.append({
            "title": title,
            "price_gbp": price,
            "star_rating": rating,
            "availability": availability,
            "category": category
        })


# Convert list to DataFrame
df = pd.DataFrame(all_books)

# Clean price
df["price_gbp"] = (
    df["price_gbp"]
    .str.replace("£", "", regex=False)
    .str.replace("Â", "", regex=False)
    .astype(float)
)

# Convert rating text to number
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(rating_map)

# Convert availability to boolean
df["in_stock"] = df["availability"].str.contains(
    "In stock",
    case=False,
    na=False
)

# Fixed project exchange rate
GBP_TO_INR = 105.50

df["price_inr"] = df["price_gbp"] * GBP_TO_INR

print("\nTotal books:", len(df))

print("\nCleaned data:")
print(
    df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ].head()
)

print("\nData types:")
print(df.dtypes)

print("\nNumber of categories:", df["category"].nunique())

df.to_csv("data_pipeline/books_cleaned.csv", index=False)