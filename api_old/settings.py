from pathlib import Path

# Define paths
ROOT = Path('/home/ubuntu/LTM_Bot/LTM-Long-short_Term_Memory.Bot-API_0.00')
MODELS = ROOT / 'models'
GRAPHQL_SCHEMA = ROOT / 'schema.graphql'

# Server
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 8000

HEADERS = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Connection': 'keep-alive',
            'DNT': '1'
        }