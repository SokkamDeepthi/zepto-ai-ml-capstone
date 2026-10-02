import sqlite3
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).parent
DB_FILE = BASE_DIR / "books.db"
OUTPUT_FILE = BASE_DIR / "query_outputs.txt"


connection = sqlite3.connect(DB_FILE)


output = []


def add_output(text=""):
    print(text)
    output.append(str(text))


add_output("=" * 70)
add_output("ZEPTO DATA PIPELINE - SQL QUERY RESULTS")
add_output("=" * 70)


# =========================================================
# QUERY 1
# SELECT + WHERE
# =========================================================

query_1 = """
SELECT
    title,
    price_gbp,
    rating,
    in_stock
FROM books
WHERE rating >= 4;
"""

result_1 = pd.read_sql_query(
    query_1,
    connection
)

add_output("\nQUERY 1 - SELECT + WHERE")
add_output(query_1)
add_output(result_1.to_string(index=False))


# =========================================================
# QUERY 2
# ORDER BY + LIMIT
# =========================================================

query_2 = """
SELECT
    title,
    price_gbp,
    price_inr
FROM books
ORDER BY price_inr DESC
LIMIT 10;
"""

result_2 = pd.read_sql_query(
    query_2,
    connection
)

add_output("\nQUERY 2 - ORDER BY + LIMIT")
add_output(query_2)
add_output(result_2.to_string(index=False))


# =========================================================
# QUERY 3
# DISTINCT
# =========================================================

query_3 = """
SELECT DISTINCT
    category_name
FROM categories
ORDER BY category_name;
"""

result_3 = pd.read_sql_query(
    query_3,
    connection
)

add_output("\nQUERY 3 - DISTINCT")
add_output(query_3)
add_output(result_3.to_string(index=False))


# =========================================================
# QUERY 4
# IN
# =========================================================

query_4 = """
SELECT
    title,
    rating,
    price_gbp
FROM books
WHERE rating IN (4, 5)
ORDER BY rating DESC, price_gbp DESC;
"""

result_4 = pd.read_sql_query(
    query_4,
    connection
)

add_output("\nQUERY 4 - IN")
add_output(query_4)
add_output(result_4.to_string(index=False))


# =========================================================
# QUERY 5
# BETWEEN
# =========================================================

query_5 = """
SELECT
    title,
    price_gbp,
    price_inr
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp;
"""

result_5 = pd.read_sql_query(
    query_5,
    connection
)

add_output("\nQUERY 5 - BETWEEN")

add_output(query_5)

add_output(result_5.to_string(index=False))

# =========================================================
# QUERY 6
# JOIN
# =========================================================

query_6 = """
SELECT
    b.book_id,
    b.title,
    b.price_gbp,
    b.rating,
    c.category_name
FROM books b
JOIN categories c
    ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.title
LIMIT 10;
"""

result_6 = pd.read_sql_query(
    query_6,
    connection
)

add_output("\nQUERY 6 - JOIN")
add_output(query_6)
add_output(result_6.to_string(index=False))


# =========================================================
# PD.READ_SQL REQUIREMENT
# =========================================================

add_output("\n" + "=" * 70)
add_output("PANDAS pd.read_sql VERIFICATION")
add_output("=" * 70)


read_sql_1 = pd.read_sql_query(
    """
    SELECT title, rating
    FROM books
    WHERE rating = 5
    LIMIT 10;
    """,
    connection
)

read_sql_2 = pd.read_sql_query(
    """
    SELECT title, price_inr
    FROM books
    ORDER BY price_inr DESC
    LIMIT 10;
    """,
    connection
)


add_output("\nResult 1 using pd.read_sql:")
add_output(read_sql_1.to_string(index=False))


add_output("\nResult 2 using pd.read_sql:")
add_output(read_sql_2.to_string(index=False))


# =========================================================
# PD.MERGE EQUIVALENT JOIN
# =========================================================

add_output("\n" + "=" * 70)
add_output("PANDAS pd.merge JOIN VERIFICATION")
add_output("=" * 70)


books_df = pd.read_sql_query(
    """
    SELECT
        book_id,
        title,
        price_gbp,
        rating,
        category_id
    FROM books;
    """,
    connection
)


categories_df = pd.read_sql_query(
    """
    SELECT
        category_id,
        category_name
    FROM categories;
    """,
    connection
)


# Perform equivalent JOIN using pandas merge

merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)


merged_result = (
    merged_df[
        [
            "book_id",
            "title",
            "price_gbp",
            "rating",
            "category_name"
        ]
    ]
    .sort_values(
        by=["rating", "title"],
        ascending=[False, True]
    )
    .head(10)
    .reset_index(drop=True)
)


sql_join_result = (
    result_6
    .sort_values(
        by=["rating", "title"],
        ascending=[False, True]
    )
    .reset_index(drop=True)
)


add_output("\nSQL JOIN result:")
add_output(
    sql_join_result.to_string(index=False)
)


add_output("\nPandas pd.merge result:")
add_output(
    merged_result.to_string(index=False)
)


# =========================================================
# COMPARE SQL JOIN AND PD.MERGE
# =========================================================

columns_to_compare = [
    "book_id",
    "title",
    "price_gbp",
    "rating",
    "category_name"
]


sql_compare = sql_join_result[
    columns_to_compare
].copy()

merge_compare = merged_result[
    columns_to_compare
].copy()


sql_compare["price_gbp"] = (
    sql_compare["price_gbp"]
    .round(2)
)

merge_compare["price_gbp"] = (
    merge_compare["price_gbp"]
    .round(2)
)


join_match = sql_compare.equals(
    merge_compare
)


add_output(
    f"\nDo SQL JOIN and pd.merge match? "
    f"{join_match}"
)


# =========================================================
# SUMMARY
# =========================================================

add_output("\n" + "=" * 70)
add_output("QUERY SUMMARY")
add_output("=" * 70)

add_output("Query 1: SELECT + WHERE")
add_output("Query 2: ORDER BY + LIMIT")
add_output("Query 3: DISTINCT")
add_output("Query 4: IN")
add_output("Query 5: BETWEEN")
add_output("Query 6: JOIN")
add_output("Pandas: pd.read_sql_query used")
add_output("Pandas: pd.merge used")
add_output(f"SQL JOIN == pd.merge: {join_match}")


# =========================================================
# SAVE OUTPUT
# =========================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(output)
    )


connection.close()


print("\n" + "=" * 70)
print("QUERY EXECUTION COMPLETED")
print("=" * 70)

print(f"\nOutput saved to:")
print(OUTPUT_FILE)