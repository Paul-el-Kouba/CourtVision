import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


# Get credentials
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
# Get a reference to the storage service
from firebase_admin import storage
storage_client = storage
# Reference the specific Firebase Storage bucket you want to access
bucket = storage_client.bucket("Database")
# List files in a specific folder
folder_name = "30-04-2024_10-27"
blobs = bucket.list_blobs(prefix=folder_name)
for blob in blobs:
if blob.name.endswith(‘.mp4’):
print(blob.name)

