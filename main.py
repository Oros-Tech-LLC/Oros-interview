from datetime import datetime, timezone
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
from routes.interview import router as qa_router
from jose import jwt, jwk
from jose.utils import base64url_decode
import requests
import uvicorn
import os
import json


description = """
API for OROS QA Generator
"""

app = FastAPI(title="OROS Backend API", description=description, version="vbeta")

def get_jwks():
    url = os.getenv("COGNITO_URL")
    response = requests.get(url)
    return response.json()

async def check_auth_header(request: Request):
    if urlparse(str(request.url)).path != "/":
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    
        try:
            # Split the token based on the provided format: type=user_token;token=<your-token>
            token_type, token_part = auth_header.split(';')
            token = token_part.split('=')[1] 
            jwks = get_jwks()
            headers = jwt.get_unverified_headers(token)
            kid = headers.get('kid')  
            key_index = -1
    
            for i in range(len(jwks['keys'])):
                if kid == jwks['keys'][i]['kid']:
                    key_index = i
                    break
                
            if key_index == -1:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token expired")
    
            # Construct the public key
            public_key = jwk.construct(jwks['keys'][key_index])
    
            # Verify the token signature
            try:
                claims = jwt.decode(
                    token, 
                    public_key, 
                    algorithms=['RS256'], 
                    audience=os.getenv("AUDIENCE"),
                    options={"verify_exp": True}
                )
            except jwt.ExpiredSignatureError:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Token expired")
            except jwt.JWTError as jwt_error:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"JWT Error: {jwt_error}")
    
        except ValueError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format")

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    try:
        await check_auth_header(request)
    except HTTPException as auth_error:
        return JSONResponse(status_code=auth_error.status_code, content={"detail": auth_error.detail})
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    return await call_next(request)
    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(qa_router, tags=["QA with topics"])

@app.get("/")
def read_root():
    return {"message": "App Healthy!!!"}



#if __name__ == "__main__":
# uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9001")))

