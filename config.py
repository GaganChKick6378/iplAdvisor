
# API Keys
EXA_API_KEY = 'YOUR API KEY'
OPENAI_API_KEY = 'YOUR API KEY'
LANGCHAIN_API_KEY = 'YOUR API KEY'
LANGCHAIN_PROJECT = "ipl-advisor"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_TRACING='true'
# Search parameters
MAX_RESULTS = 10
MAX_CHARS = 5000

# Vector DB settings
VECTOR_DB_PATH = "cricket_data_store"

# OpenAI settings
MODEL_NAME = "gpt-3.5-turbo"

# Confidence thresholds
HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.6
LOW_CONFIDENCE = 0.4

LLM_COST_PER_1K_TOKENS  = 0.002    # $0.002 per 1k tokens for gpt-3.5-turbo
EXA_COST_PER_RESULT     = 0.0001   # $0.0001 per result ( taken from their website)