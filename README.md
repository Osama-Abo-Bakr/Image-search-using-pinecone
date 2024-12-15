# Image Search Using Pinecone

This repository demonstrates a robust solution for image search using Pinecone Vector Database. The system leverages machine learning models to extract image features, store them in Pinecone's vector database, and retrieve the most similar images based on feature vectors. Built with FastAPI, it provides a flexible and efficient API for managing and querying image data.

## Repository Details
- **Repo Name**: [Image-search-using-pinecone](https://github.com/Osama-Abo-Bakr/Image-search-using-pinecone.git)

## Key Features
- **Efficient Vector Search**: Enables quick similarity searches for images.
- **API Endpoints**: Perform image search, add new images, or delete existing ones from the database.
- **Pinecone Integration**: Utilizes Pinecone’s scalable vector search technology for efficient storage and retrieval.
- **Feature Extraction**: Extracts deep features from images using a pre-trained VGG19 model from the Timm library.
- **Customizable Search**: Supports filtering by classes and threshold similarity scores.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip
- Pinecone API Key

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/Image-search-using-pinecone.git
   cd Image-search-using-pinecone
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the environment variables by creating a `.env` file:
   ```env
   PINECONE_API_KEY=<Your Pinecone API Key>
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

## Usage

### API Endpoints

#### 1. **Image Search**
- **Endpoint**: `/`
- **Method**: `POST`
- **Parameters**:
  - `image_url` (str, required): URL of the image to search for.
  - `top_k` (int, required): Number of top similar images to retrieve.
  - `threshold` (float, optional): Minimum similarity score to consider.
  - `class_type` (str, required): Class filter (`ALL`, `class-a`, `class-b`).
- **Response**: Returns a list of similar image IDs and their similarity scores.

#### 2. **Insert or Delete Image**
- **Endpoint**: `/updating_or_deleting`
- **Method**: `POST`
- **Parameters**:
  - `image_id` (int, required): Unique ID for the image.
  - `image_url` (str, optional): URL of the image (required for upsert).
  - `class_type` (str, optional): Class label (`class-a`, `class-b`, required for upsert).
  - `case` (str, required): Operation type (`Upsert`, `Delete`).
- **Response**: Success or error message indicating the operation result.

### Example Requests

#### Search Image
```bash
curl -X POST "http://127.0.0.1:8000/" \
     -F "image_url=https://example.com/sample-image.jpg" \
     -F "top_k=5" \
     -F "threshold=0.8" \
     -F "class_type=ALL"
```

#### Insert Image
```bash
curl -X POST "http://127.0.0.1:8000/updating_or_deleting" \
     -F "image_id=12345" \
     -F "image_url=https://example.com/sample-image.jpg" \
     -F "class_type=class-a" \
     -F "case=Upsert"
```

#### Delete Image
```bash
curl -X POST "http://127.0.0.1:8000/updating_or_deleting" \
     -F "image_id=12345" \
     -F "case=Delete"
```

## Implementation Details

### Core Components
1. **Feature Extraction**:
   - Uses a pre-trained VGG19 model to extract deep image features.
   - Feature vectors are normalized and resized for efficient processing.

2. **Vector Database**:
   - Pinecone stores feature vectors and provides real-time similarity search.

3. **Utilities**:
   - Functions for downloading images, extracting features, and managing vector data.

### File Structure
```
Image-search-using-pinecone/
├── main.py          # FastAPI server and endpoints
├── utils.py         # Utility functions for feature extraction and Pinecone operations
├── requirements.txt # Required Python packages
├── .env             # Environment variables
├── README.md        # Project documentation
```

## Contributing
Contributions are welcome! If you'd like to improve this project, please:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed explanation of your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or collaboration, please reach out to:
- **Name**: Osama Abo Bakr
- **GitHub**: [Osama-Abo-Bakr](https://github.com/Osama-Abo-Bakr)