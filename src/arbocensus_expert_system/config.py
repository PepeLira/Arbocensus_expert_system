from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='./arbocensus_expert_system/my_parameters.env', encoding="utf-8") # Load environment variables from .env file

def get_env(key, default=None):
    value = os.getenv(key)
    if value is None and default is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value if value is not None else default