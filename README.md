# Movie Recommender (TF-IDF + Cosine Similarity)

A content-based movie recommendation system built in Python. Type in a few keywords (genres, themes, plot ideas) and the system returns the five most similar movies from a dataset, along with their details and similarity scores.

Built as a final project for a BSCS course.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Results](#results)
- [Limitations](#limitations)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Possible Improvements](#possible-improvements)

---

## Project Overview

### Problem

With thousands of movies available, it is hard to find something that fits a specific mood or idea. Browsing by genre alone is too broad, and most people can't remember an exact title. They usually remember themes: *"a horror movie with a game at a wedding"*.

### Goal

Build a simple recommender that takes free-form keywords from the user and returns the movies whose content best matches those keywords.

### Approach

This is a **content-based** recommender. It does not use user ratings or viewing history. It looks only at the movies' own text (plot overview, genres, and keywords) and measures how closely each movie matches what the user typed.

### Features

- Free-form keyword search (e.g. `horror, game, suspense, family, wedding, escape`)
- Top 5 recommendations showing title, release year, runtime, genres, and overview
- A summary table of titles with similarity scores
- Interactive loop: keep searching until you enter `0`

### Tech Stack

- Python 3
- [pandas](https://pandas.pydata.org/) for data handling
- [scikit-learn](https://scikit-learn.org/) for `TfidfVectorizer` and `cosine_similarity`
- [tabulate](https://pypi.org/project/tabulate/)
- Jupyter Notebook

### Dataset

`cleaned_dataset.csv` contains the movie records used by the recommender. The notebook relies on these columns:

| Column         | Description            |
| -------------- | ---------------------- |
| `title`        | Movie title            |
| `overview`     | Plot summary           |
| `genres`       | Comma-separated genres |
| `keywords`     | Descriptive keywords   |
| `release_date` | Release year           |
| `runtime`      | Runtime in minutes     |

---

## Methodology

The pipeline has five steps.

### 1. Load the data

The cleaned dataset is loaded into a pandas DataFrame.

### 2. Build a text representation for each movie

Three text columns are concatenated into one field called `combined_text`:

```python
movies_df['combined_text'] = (
    movies_df['overview'] + ' ' +
    movies_df['genres'] + ' ' +
    movies_df['keywords']
)
```

Combining them means a query can match a movie through its plot, its genre, or its tags.

### 3. Convert text to TF-IDF vectors

`TfidfVectorizer(stop_words='english')` turns every movie's `combined_text` into a numeric vector.

**TF-IDF** (Term Frequency, Inverse Document Frequency) gives each word a weight:

- **TF**: a word counts for more if it appears often in a movie's text.
- **IDF**: a word counts for less if it appears in many movies. Common words carry little information, while rare words like "escape" or "wedding" are more distinctive.

English stopwords (*the*, *and*, *of*, ...) are removed so they don't add noise. The result is a sparse matrix with one row per movie and one column per vocabulary word.

### 4. Vectorize the user's query

The user's keywords are transformed with the **same fitted vectorizer** (`tfidf.transform`), so the query and the movies live in the same vector space. Words in the query that don't exist in the movie vocabulary are ignored.

### 5. Rank by cosine similarity

Cosine similarity measures the angle between the query vector and each movie vector:

```
cos(q, m) = (q · m) / (‖q‖ ‖m‖)
```

The score ranges from 0 (no shared weighted terms) to 1 (identical direction). Because it depends on direction and not length, long and short overviews can be compared fairly. The movies are sorted by score, and the top N (default 5) are returned.

```python
cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix).flatten()
top_indices = cosine_sim.argsort()[-top_n:][::-1]
```

### Output

For each recommendation the program prints the title, release year, runtime (converted from minutes to `Xh Ym`), genres, and overview. It then prints a summary table of titles and similarity scores.

---

## Results

### Sample query

```
Enter keywords (or type 0 to quit): horror, game, suspense, family, wedding, escape
```

### Recommendations

| Rank | Title        | Year | Runtime | Genres                  | Similarity |
| ---- | ------------ | ---- | ------- | ----------------------- | ---------- |
| 1    | Ready or Not | 2019 | 1h 35m  | Comedy, Horror, Thriller | 0.3950     |
| 2    | The Week Of  | 2018 | 1h 56m  | Comedy                  | 0.3662     |
| 3    | KING GAME    | 2010 | 1h 33m  | Drama, Thriller, Mystery | 0.3657     |
| 4    | Play or Die  | 2019 | 1h 29m  | Horror, Mystery         | 0.3528     |
| 5    | Sex After Six | 1972 | 1h 22m | Comedy                  | 0.3437     |

### Example console output

```
Recommended Movies:
Title: Ready or Not
Released Date: 2019
Runtime: 1h 35m
Genre: Comedy, Horror, Thriller
Overview: A bride's wedding night takes a sinister turn when her eccentric
new in-laws force her to take part in a terrifying game.
----------------------------------------
...

-----------------------------------------------------
| Title                          | Similarity Score |
-----------------------------------------------------
| Ready or Not                   | 0.3950           |
| The Week Of                    | 0.3662           |
| KING GAME                      | 0.3657           |
| Play or Die                    | 0.3528           |
| Sex After Six                  | 0.3437           |
-----------------------------------------------------
```

### Discussion

- **The top result fits well.** *Ready or Not* matches several parts of the query at once: horror, a deadly game, a wedding, and family (the in-laws). It is the only result that combines them, which is why it ranks first.
- **Other results match on part of the query.** *Play or Die* (an escape game, horror) and *KING GAME* (trapped in a room, forced to play a game) match the "game / escape / suspense" side. *The Week Of* and *Sex After Six* match mostly on "wedding" (and "game" for the latter), even though they are comedies and not horror.
- **Scores are moderate (about 0.34 to 0.40).** The query is short, while each movie's text is long, so even the best match shares only part of its weighted vocabulary with the query. What matters is the relative ranking, not the absolute value.
- **Scores are close together.** The top five span only about 0.05, so ranks 2 to 5 are not strongly separated.

This project was checked by running example queries and reading the results. It was not evaluated with a formal metric such as precision@k or user ratings.

---

## Limitations

- **Keyword matching, not meaning.** TF-IDF only matches the words used. A query like "scary" won't match a movie described only with "terrifying" or "horror" unless those exact words are in its text.
- **Partial matches can outrank better fits.** As seen above, a movie matching one strong keyword (like "wedding") can appear even if it misses the intended genre.
- **No personalization.** Every user gets the same results for the same query.
- **Unknown words are ignored.** If none of the query words are in the vocabulary, all scores are 0 and the results are effectively arbitrary.
- **No formal evaluation** (see above).

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ven-62/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn tabulate jupyter
```

### 3. Run the notebook

```bash
jupyter notebook FP_Lavapie_VenJohnRey_BSCS2A.ipynb
```

Run all cells in order. The last cell starts the interactive prompt. Type keywords and press Enter, or type `0` to quit.

---

## Project Structure

```
.
├── FP_Lavapie_VenJohnRey_BSCS2A.ipynb   # Main notebook
├── cleaned_dataset.csv                  # Movie dataset
└── README.md
```

---

## Possible Improvements

- Add a web interface (Streamlit or Flask)
- Support "movies similar to X" searches by title
- Try n-grams, or semantic embeddings (e.g. sentence transformers) to match meaning instead of exact words
- Filter results by genre, year, or runtime
- Handle empty or out-of-vocabulary queries gracefully
- Evaluate with a measurable metric or user feedback

---
