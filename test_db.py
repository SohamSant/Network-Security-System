
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
load_dotenv()
import os

# uri = "mongodb+srv://sohamsant9_db_user:<db_password>@cluster0.kmb9bz7.mongodb.net/?appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(os.getenv("MONGO_DB_URL"), server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)