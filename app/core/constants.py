"""
Immutable constants — domain names, schema field names, etc.
Import from here instead of scattering string literals around the codebase.
"""

# ── Domain tags (used in unified schema) ──────────────────────────────────────
DOMAIN_AMAZON    = "amazon"
DOMAIN_GOODREADS = "goodreads"
DOMAIN_YELP      = "yelp"          # reserved for future use

KNOWN_DOMAINS = {DOMAIN_AMAZON, DOMAIN_GOODREADS}


# ── Unified schema field names ─────────────────────────────────────────────────
# Keeping these as constants prevents typo-bugs across the pipeline.

F_USER_ID       = "user_id"
F_ITEM_ID       = "item_id"
F_RATING        = "rating"
F_REVIEW_TEXT   = "review_text"
F_TIMESTAMP     = "timestamp"
F_DOMAIN        = "domain"
F_CATEGORY      = "category"
F_SPLIT         = "split"          # "train" | "test"

# Item metadata sub-fields
F_ITEM_TITLE    = "title"
F_ITEM_DESC     = "description"
F_ITEM_AVG_RATING = "avg_rating"
F_ITEM_PRICE_TIER = "price_tier"

# User profile fields
F_PROFILE_AVG_RATING     = "avg_rating"
F_PROFILE_RATING_STD     = "rating_std"
F_PROFILE_REVIEW_LEN_AVG = "review_length_avg"
F_PROFILE_TOP_CATEGORIES = "top_categories"
F_PROFILE_SENTIMENT      = "sentiment_tendency"
F_PROFILE_SAMPLE_REVIEWS = "sample_reviews"


# ── Rating normalisation ───────────────────────────────────────────────────────
RATING_MIN = 1.0
RATING_MAX = 5.0

PRICE_TIER_BINS   = [0, 10, 30, 75, float("inf")]
PRICE_TIER_LABELS = ["budget", "mid", "premium", "luxury"]


# ── DuckDB table names ─────────────────────────────────────────────────────────
TABLE_REVIEWS       = "reviews"
TABLE_ITEMS         = "items"
TABLE_USER_PROFILES = "user_profiles"
TABLE_TRAIN         = "train"
TABLE_TEST          = "test"