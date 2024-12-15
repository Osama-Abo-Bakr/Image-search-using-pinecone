from fastapi import FastAPI, Form, HTTPException
from utils import search_vectorDB, delete_vectorDB, Inserting_to_Pinecone
import os




# Initialize the FastAPI app
app = FastAPI(debug=True)


## Variable for the first endpoints
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_IMAGES_FOLDER_PATH = os.path.join(PROJECT_PATH, 'download-images')
os.makedirs(DOWNLOAD_IMAGES_FOLDER_PATH, exist_ok=True)


# ------------------ Endpoints for Image Search ----------------------------------
@app.post('/')
async def image_search(image_url: str = Form(...),
                       top_k: int = Form(...),
                       threshold: float = Form(None),
                       class_type: str = Form(..., description='Class-A or Class-B', enum=['ALL', 'class-a', 'class-b'])):
    
    try:
        ## Validation for top_k, and threshold
        if top_k <= 0 or not isinstance(top_k, int) or top_k > 10000 or top_k is None:
            raise HTTPException(status_code=400, detail="Bad Request: 'top_k' must be a positive integer and less than 10000.")
        
        elif threshold is not None and (threshold <= 0.0 or not isinstance(threshold, float) or threshold > 1.0):
            raise HTTPException(status_code=400, 
                                detail="Bad Request: 'threshold' must be a positive float greater than 0.0 and less than 1.0")
        else:        
            
            results = search_vectorDB(image_url=image_url, folder_path=DOWNLOAD_IMAGES_FOLDER_PATH,
                                           top_k=top_k, threshold=threshold, class_type=class_type)
            
            # After Gitting Response.
            _ = [os.remove(os.path.join(DOWNLOAD_IMAGES_FOLDER_PATH, image)) for image in os.listdir(DOWNLOAD_IMAGES_FOLDER_PATH)]
            
            return results
            
        
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to Search Data')
    

# https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/29708160-9b39-42d0-a5ed-4f8b9c85a267/labrador+retriever+dans+pet+care.jpeg?format=1500w
# https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/29708160-9b39-42d0-a5ed-4f8b9c85a267/labrador+retriever+dans+pet+care.jpeg?format=1500w

# ---------------------------- Endpoints For Updating Vector DB ----------------------------

## Variable for the Second endpoints
INDEXING_IMAGES_FOLDER_PATH = os.path.join(PROJECT_PATH, 'download-images')
os.makedirs(INDEXING_IMAGES_FOLDER_PATH, exist_ok=True)

@app.post('/updating_or_deleting')
async def index_or_delete(image_id: int = Form(...),
                          image_url: str = Form(None),
                          class_type: str = Form(None, description='Class-Type', enum=['class-a', 'class-b']),
                          case: str = Form(..., description='Case', enum=['Upsert', 'Delete'])):
    
    ## Validate the new_text is not None if the case=upsert
    if case == 'Upsert' and (not image_url or not class_type):
        raise HTTPException(status_code=400, detail='"image_url & class_type" is mandatory for case "upsert".')
    
    # Call Function (inserting_vector) from utils
    if case == 'Upsert':
        msg = Inserting_to_Pinecone(image_id=image_id, image_url=image_url, 
                                    folder_path=INDEXING_IMAGES_FOLDER_PATH,
                                    class_type=class_type)
    
    # Call Function (deleting_vector) from utils
    elif case == 'Delete':
        msg = delete_vectorDB(image_id=image_id)
    
    return {'message': msg}