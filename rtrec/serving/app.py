from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any, Dict
import logging

from rtrec.models import Fast_SLIM_MSE as SlimMSE
from .dependencies import resolve_account_id

# Create a FastAPI instance
app = FastAPI()

# CORS Middleware (if necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust based on your requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = SlimMSE(alpha=0.1, beta=1.0, lambda1=0.0002, lambda2=0.0001, min_value=-5, max_value=10)

# Define input models for requests
class Interaction(BaseModel):
    user: Any
    item: Any
    timestamp: float
    rating: float

class RecommendationRequest(BaseModel):
    user: Any
    top_k: int = 10
    filter_interacted: bool = True

# Response model for recommendations
class RecommendationResponse(BaseModel):
    user: Any
    recommendations: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {"message": "Recommender System API is running"}

# Fit endpoint to train the recommender system
@app.post("/fit")
async def fit(interactions: List[Interaction], account_id: str = Depends(resolve_account_id)):
    try:
        user_interactions = [
            (interaction.user, interaction.item, interaction.timestamp, interaction.rating)
            for interaction in interactions
        ]
        recommender.fit(user_interactions)
        return {"message": "Training successful"}
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail="Training failed")

# Recommend endpoint to get recommendations for a user
@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest, account_id: str = Depends(resolve_account_id)):
    try:
        recommendations = recommender.recommend(
            user=request.user, top_k=request.top_k, filter_interacted=request.filter_interacted
        )

        # Format recommendations as a list of dictionaries with metadata
        response = {
            "user": request.user,
            "recommendations": [{"item": item} for item in recommendations]
        }
        return response
    except Exception as e:
        logging.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Recommendation failed")

# Run the app if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
