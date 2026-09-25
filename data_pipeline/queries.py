import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect("data_pipeline/books.db")

# --------------------------------------------------
# QUERY 1: SELECT + WHERE
# Find books with rating 5
# --------------------------------------------------

query1 = """
SELECT title, price_gbp, rating
FROM books
WHERE rating = 5
"""

result1 = pd.read_sql(query1, conn)

print("\nQUERY 1 - Books with rating 5")
print(result1.head(10))


# --------------------------------------------------
# QUERY 2: ORDER BY + LIMIT
# Find 10 most expensive books
# --------------------------------------------------

query2 = """
SELECT title, price_gbp, price_inr
FROM books
ORDER BY price_gbp DESC
LIMIT 10
"""

result2 = pd.read_sql(query2, conn)

print("\nQUERY 2 - Top 10 most expensive books")
print(result2)


# --------------------------------------------------
# QUERY 3: DISTINCT
# Find all unique categories
# --------------------------------------------------

query3 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name
"""

result3 = pd.read_sql(query3, conn)

print("\nQUERY 3 - Unique categories")
print(result3)


# --------------------------------------------------
# QUERY 4: BETWEEN
# Find books priced between £20 and £30
# --------------------------------------------------

query4 = """
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 20 AND 30
ORDER BY price_gbp
"""

result4 = pd.read_sql(query4, conn)

print("\nQUERY 4 - Books priced between £20 and £30")
print(result4.head(10))


# --------------------------------------------------
# QUERY 5: JOIN
# Combine books and categories
# --------------------------------------------------

query5 = """
SELECT
    books.title,
    books.price_gbp,
    books.price_inr,
    books.rating,
    books.in_stock,
    categories.category_name
FROM books
JOIN categories
ON books.category_id = categories.category_id
"""

result5 = pd.read_sql(query5, conn)

print("\nQUERY 5 - Books with category using JOIN")
print(result5.head(10))


# --------------------------------------------------
# pandas.merge()
# Reproduce the same JOIN using pandas
# --------------------------------------------------

books_df = pd.read_sql("""
SELECT
    book_id,
    title,
    price_gbp,
    price_inr,
    rating,
    in_stock,
    category_id
FROM books
""", conn)

categories_df = pd.read_sql("""
SELECT
    category_id,
    category_name
FROM categories
""", conn)

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)

print("\nPANDAS MERGE - Equivalent to SQL JOIN")
print(
    merged_df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ].head(10)
)


# --------------------------------------------------
# Save query outputs
# --------------------------------------------------

with open("data_pipeline/query_outputs.txt", "w", encoding="utf-8") as file:

    file.write("QUERY 1 - Books with rating 5\n")
    file.write(query1)
    file.write("\nOUTPUT:\n")
    file.write(result1.to_string(index=False))

    file.write("\n\n" + "=" * 70 + "\n\n")

    file.write("QUERY 2 - Top 10 most expensive books\n")
    file.write(query2)
    file.write("\nOUTPUT:\n")
    file.write(result2.to_string(index=False))

    file.write("\n\n" + "=" * 70 + "\n\n")

    file.write("QUERY 3 - Unique categories\n")
    file.write(query3)
    file.write("\nOUTPUT:\n")
    file.write(result3.to_string(index=False))

    file.write("\n\n" + "=" * 70 + "\n\n")

    file.write("QUERY 4 - Books priced between £20 and £30\n")
    file.write(query4)
    file.write("\nOUTPUT:\n")
    file.write(result4.to_string(index=False))

    file.write("\n\n" + "=" * 70 + "\n\n")

    file.write("QUERY 5 - Books with category using JOIN\n")
    file.write(query5)
    file.write("\nOUTPUT:\n")
    file.write(result5.to_string(index=False))

    file.write("\n\n" + "=" * 70 + "\n\n")

    file.write("PANDAS MERGE - Equivalent to SQL JOIN\n")
    file.write(
        merged_df[
            [
                "title",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
                "category_name"
            ]
        ].head(10).to_string(index=False)
    )


print("\nQuery outputs saved to data_pipeline/query_outputs.txt")

conn.close()