from fastapi import HTTPException, Request
import httpx
import os

AUTH_ENDPOINT = os.getenv("AUTH_ENDPOINT", "https://your-auth-service/validate")

# Function to verify XSRF token and resolve account_id
async def resolve_account_id(request: Request) -> str:
    xsrf_token = request.cookies.get("XSRF-TOKEN")  # Adjust cookie name as necessary
    if not xsrf_token:
        raise HTTPException(status_code=403, detail="Missing XSRF token")

    # Make a request to the external authentication service to validate the token
    async with httpx.AsyncClient() as client:
        response = await client.get(AUTH_ENDPOINT, params={"xsrf_token": xsrf_token})

        if response.status_code != 200:
            raise HTTPException(status_code=403, detail="Invalid XSRF token")

        # Assuming the service returns the account_id
        account_id = response.json().get("account_id")
        if not account_id:
            raise HTTPException(status_code=403, detail="Account ID not found")

    return account_id
