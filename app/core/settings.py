# app/core/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Google Cloud credentials
    GCP_PROJECT_ID: str = Field(..., env='GCP_PROJECT_ID')
    GCS_BUCKET_NAME: str = Field(..., env='GCS_BUCKET_NAME')

    # Supabase credentials
    SUPABASE_URL: str = Field(..., env='SUPABASE_URL')
    SUPABASE_SERVICE_ROLE_KEY: str = Field(...,
                                           env='SUPABASE_SERVICE_ROLE_KEY')

    # Default to 'production'
    ENVIRONMENT: str = Field('production', env='ENVIRONMENT')

    # Ngrok URL
    NGROK_URL: str = Field(..., env='NGROK_URL')

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


# Instantiate the settings
settings = Settings()
