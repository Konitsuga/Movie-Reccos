import time
from config import OMDB_SECRET, TMDB_SECRET
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from config import TMDB_SECRET, OMDB_SECRET

# --- SECRET API ---
TMDB_API_KEY = TMDB_SECRET
OMDB_API_KEY = OMDB_SECRET

MIN_IMDB_SCORE = 7.7
MIN_RT_SCORE = 77

MY_LIST = [
    {"title": "Kung Fu Hustle", "type": "movie"},
    {"title": "Modern Love Chennai", "type": "tv"},
    {"title": "Khatta Meetha", "type": "movie"},
    {"title": "Sisu", "type": "movie"},
    {"title": "Are You There God? Its Me Margaret", "type": "movie"},
]


# --- RESILIENT HTTP SESSION ---
def create_retry_session():
    """Creates a requests session configured with retries and exponential backoff."""
    session = requests.Session()
    retries = Retry(
        total=5,  # Retry up to 5 times
        backoff_factor=1,  # Wait 1s, 2s, 4s, 8s, 16s between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on server errors or rate limits
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


http = create_retry_session()


# --- HELPER FUNCTIONS ---


def get_tmdb_id(title, media_type):
    """Searches TMDb to find the internal ID for a given movie/show title."""
    url = f"https://api.themoviedb.org/3/search/{media_type}"
    params = {"api_key": TMDB_API_KEY, "query": title}

    try:
        response = http.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            return results[0]["id"]
    except requests.exceptions.RequestException as e:
        print(f"\n[Warning] Failed to fetch TMDb ID for '{title}': {e}")

    return None


def get_tmdb_recommendations(tmdb_id, media_type):
    """Fetches TMDb's recommended media for a specific TMDb ID."""
    url = f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}/recommendations"
    params = {"api_key": TMDB_API_KEY}

    recs = []
    try:
        response = http.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])

        for item in results:
            recs.append(
                {
                    "title": item.get("title") or item.get("name"),
                    "media_type": media_type,
                    "release_date": item.get("release_date")
                    or item.get("first_air_date"),
                }
            )
    except requests.exceptions.RequestException as e:
        print(
            f"\n[Warning] Failed to fetch recommendations for ID {tmdb_id}: {e}"
        )

    return recs


def get_scores_from_omdb(title):
    """Fetches IMDb Rating and Rotten Tomatoes Score via OMDb API."""
    url = "http://www.omdbapi.com/"
    params = {"apikey": OMDB_API_KEY, "t": title}

    imdb_rating = None
    rt_rating = None

    try:
        response = http.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("Response") == "True":
            # Extract IMDb Rating
            try:
                imdb_rating = float(data.get("imdbRating", 0))
            except (ValueError, TypeError):
                imdb_rating = None

            # Extract Rotten Tomatoes Score
            ratings = data.get("Ratings", [])
            for rating in ratings:
                if rating.get("Source") == "Rotten Tomatoes":
                    rt_val = rating.get("Value", "").replace("%", "")
                    try:
                        rt_rating = int(rt_val)
                    except ValueError:
                        rt_rating = None

    except requests.exceptions.RequestException as e:
        print(f"\n[Warning] Failed to fetch OMDb scores for '{title}': {e}")

    return imdb_rating, rt_rating


# --- MAIN PIPELINE ---


def generate_filtered_recommendations(my_list):
    input_titles_lower = {item["title"].lower() for item in my_list}
    raw_recommendations = {}

    print("Step 1/3: Fetching recommendations from TMDb...")
    for item in tqdm(my_list):
        tmdb_id = get_tmdb_id(item["title"], item["type"])
        if not tmdb_id:
            continue

        recs = get_tmdb_recommendations(tmdb_id, item["type"])
        for rec in recs:
            title = rec["title"]
            if title and title.lower() not in input_titles_lower:
                if title not in raw_recommendations:
                    raw_recommendations[title] = {
                        "type": rec["media_type"],
                        "frequency": 1,
                    }
                else:
                    raw_recommendations[title]["frequency"] += 1

        time.sleep(0.15)  # Pace TMDb calls slightly

    print(
        f"\nFound {len(raw_recommendations)} unique recommended candidate titles."
    )
    print("Step 2/3: Fetching IMDb and Rotten Tomatoes scores via OMDb...")

    final_results = []
    sorted_candidates = sorted(
        raw_recommendations.items(),
        key=lambda x: x[1]["frequency"],
        reverse=True,
    )

    for title, meta in tqdm(sorted_candidates):
        imdb_score, rt_score = get_scores_from_omdb(title)

        if (
            imdb_score is not None
            and rt_score is not None
            and imdb_score >= MIN_IMDB_SCORE
            and rt_score >= MIN_RT_SCORE
        ):
            final_results.append(
                {
                    "Title": title,
                    "Type": meta["type"].upper(),
                    "IMDb Rating": imdb_score,
                    "Rotten Tomatoes": f"{rt_score}%",
                    "List Relevance Score": meta["frequency"],
                }
            )

        time.sleep(0.1)

    # Step 3: Format & Output
    df = pd.DataFrame(final_results)
    if not df.empty:
        df = df.sort_values(
            by=["List Relevance Score", "IMDb Rating"], ascending=False
        )
    return df


if __name__ == "__main__":
    df_recs = generate_filtered_recommendations(MY_LIST)

    print("\n--- Top Recommendations ---")
    if not df_recs.empty:
        print(df_recs.to_string(index=False))
        df_recs.to_csv("filtered_recommendations.csv", index=False)
        print("\nSaved recommendations to 'filtered_recommendations.csv'")
    else:
        print(
            "No recommendations met the strict threshold criteria or network errors occurred."
        )