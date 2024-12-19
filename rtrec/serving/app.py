from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
import logging
import os

from rtrec.models import SLIM

# Default secret token for testing
DEFAULT_SECRET_TOKEN = "fake_secret_token"
SECRET_TOKEN = os.getenv("X_TOKEN", DEFAULT_SECRET_TOKEN)

# Initialize logging
logging.basicConfig(level=logging.INFO)

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
    recommendations: List[Any]

def create_app() -> FastAPI:
    """Factory function to create a FastAPI instance."""
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize the recommender instance
    recommender = SLIM(min_value=-5, max_value=10, decay_in_days=365)

    @app.get("/")
    def read_root():
        return {"message": "Recommender System API is running"}

    # Fit endpoint to train the recommender system
    @app.post("/fit")
    async def fit(interactions: List[Interaction], x_token: str = Header()):
        if x_token != SECRET_TOKEN:
            raise HTTPException(status_code=400, detail="Invalid X-Token header")

        try:
            user_interactions = [
                (interaction.user, interaction.item, interaction.timestamp, interaction.rating)
                for interaction in interactions
            ]
            recommender.fit(user_interactions, progress_bar=False)
            return {"message": "Training successful"}
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise HTTPException(status_code=500, detail="Training failed")

    # Recommend endpoint to get recommendations for a user
    @app.post("/recommend", response_model=RecommendationResponse)
    async def recommend(request: RecommendationRequest, x_token: str = Header()):
        if x_token != SECRET_TOKEN:
            raise HTTPException(status_code=400, detail="Invalid X-Token header")

        try:
            recommendations = recommender.recommend(
                user=request.user, top_k=request.top_k, filter_interacted=request.filter_interacted
            )

            # Format recommendations as a list of dictionaries with metadata
            response = {
                "user": request.user,
                "recommendations": recommendations
            }
            return response
        except Exception as e:
            logging.error(f"Recommendation failed: {e}")
            raise HTTPException(status_code=500, detail="Recommendation failed")

    return app

# Run the app if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
