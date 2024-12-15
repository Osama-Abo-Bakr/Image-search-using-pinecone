# Importing Main Library
import os
import timm
import torch
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from fastapi import HTTPException
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# Prepare Pinecone Index
_ = load_dotenv(override=True)
PINCONE_API_KEY = os.getenv('PINECONE_API_KEY')
pinecone = Pinecone(api_key=PINCONE_API_KEY)
index_name = 'image-search-live'
index = pinecone.Index(index_name)

# Creating Model
model_inference = timm.create_model('vgg19', pretrained=True)
model_inference = nn.Sequential(*list(model_inference.children())[:-1])
_ = model_inference.eval()

def download_image(image_url: str, folder_path: str):
    """
    Download Image
    
    Args:
    *****
        (image_url: str) --> The image url to download.
        (folder_path: str) --> The folder path to save the downloaded image.
    
    Returns:
    *******
        (image_path_local: str) --> The local path of the downloaded image.
    """
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image_base = os.path.basename(image_url).split('.')[0]
            
            # Prepare Image Path
            image_path_local = os.path.join(folder_path, f"{image_base}.jpg")
            
            # Download image
            with open(image_path_local, 'wb') as f:
                f.write(response.content)
            
            return image_path_local
        else:
            raise HTTPException(status_code=404, detail="Image Not Found")
    
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to Download Image")
    
    

# Extract Image Feature
def extract_image_feature(image_paths: list):
    """
    Extract Image Feature
    
    Args:
    *****
        (image_paths: list) --> The list of image paths.
    
    Returns:
    *******
        (batch_fearures: List) --> A List of image features.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    batch_fearures = []
    for image_path in image_paths:
        # Convert Image Path to Pillow
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        
        # Pass The Image to Model to Extract Feature.
        with torch.no_grad():
            conv_feature = model_inference(image)
            
            image_features = conv_feature.view(conv_feature.size(0), -1).tolist()[0]
        
        # append Feature
        batch_fearures.append(image_features)
    
    return batch_fearures


# ------------------------------------ Search using Pinecone ------------------------------------ #
def search_vectorDB(image_url: str, folder_path: str, top_k: int, threshold: float=None, class_type: str=None):
    '''
    This Function is to use the pinecone index to make a query and retrieve similar records.
    Args:
    *****
        (image_url: str) --> The image url to get similar records to it.
        (folder_path: str) --> The folder path to save the downloaded image.
        (top_k: int) --> The number required of similar records in descending order.
        (threshold: float) --> The threshold to filter the retrieved IDs based on it.
        (class_type: str) --> Which class to filter using it (class-a or class-b)
    
    Returns:
    *******
        (similar_ids: List) --> A List of IDs for similarity records.
    '''
    try:
        # Download Image
        image_local_path = download_image(image_url=image_url, folder_path=folder_path)
        
        # Extract Image Feature
        image_features = extract_image_feature(image_paths=[image_local_path])[0]
        
        if class_type in ['class-a', 'class-b']:
            # Search in Pinecone
            results = index.query(vector=[image_features], top_k=top_k, include_metadata=True, filter={'class': class_type})['matches']
        else:
            # Search in Pinecone
            results = index.query(vector=[image_features], top_k=top_k, include_metadata=True)['matches']
        
        
        if threshold:
            similarity_scores = [{'id': int(record['id']), 'score': float(record['score']), 'class': record['metadata']['class']} for record in results if float(record['score']) > threshold]
        else:
            similarity_scores = [{'id': int(record['id']), 'score': float(record['score']), 'class': record['metadata']['class']} for record in results]
        return similarity_scores
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to Search in Pinecone")
    
    

def Inserting_to_Pinecone(image_id: int, image_url: str, folder_path: str, class_type: str):
    """
    Upserting to Pinecone
    
    Args:
    *****
        (image_id: int) --> The image id to upsert.
        (image_url: str) --> The image url to upsert.
        (folder_path: str) --> The folder path to save the downloaded image.
        (text_id: int) --> The text id to upsert.
        (text: str) --> The text to upsert.
        (class_type: str) --> Which class to filter using it (class-a or class-b)
    
    Returns:
    *******
        (similar_ids: List) --> A List of IDs for similarity records.
    """
    try:
        # Download Image
        image_local_path = download_image(image_url=image_url, folder_path=folder_path)
        
        # Extract Image Feature
        image_features = extract_image_feature(image_paths=[image_local_path])[0]
        
        # Upsert in Pinecone
        to_upsert = [(str(image_id), image_features, {'class': class_type})]
        
        _ = index.upsert(vectors=to_upsert)
        
        return f'Upserting Done: Count Now is {index.describe_index_stats()["total_vector_count"]} vectors.'
    
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to Upsert in Pinecone")
    
    

def delete_vectorDB(image_id: int):
    """
    Delete from Pinecone
    
    Args:
    *****
        (image_id: int) --> The image id to delete.
    
    Returns:
    *******
        (message: str) --> A message to show the result.
    """
    try:
        _ = index.delete(ids=[str(image_id)])
        return f"Deleting Done: Count Now is {index.describe_index_stats()['total_vector_count']} vectors."
    
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to delete from pinecone vector DB.')