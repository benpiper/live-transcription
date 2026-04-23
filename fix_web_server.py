import re

with open('web_server.py', 'r') as f:
    content = f.read()

# Make get_current_user global
search = """def create_app(boot_callback=None, input_callback=None) -> FastAPI:"""

replace = """from fastapi.security import OAuth2PasswordBearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username

def create_app(boot_callback=None, input_callback=None) -> FastAPI:"""

content = content.replace(search, replace)

# Add CORS with conditional allow_credentials
search_cors = """    cors_origins = get_security_setting("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )"""

replace_cors = """    cors_origins = get_security_setting("cors_origins", ["*"])
    if "*" in cors_origins:
        allow_credentials = False
    else:
        allow_credentials = True
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )"""

content = content.replace(search_cors, replace_cors)

# Remove the internal get_current_user
search_internal_auth = """    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

    async def get_current_user(token: str = Depends(oauth2_scheme)):
        payload = decode_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=401,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username

    @app.post("/api/login", response_model=TokenResponse, tags=["auth"])"""

replace_internal_auth = """    @app.post("/api/login", response_model=TokenResponse, tags=["auth"])"""

content = content.replace(search_internal_auth, replace_internal_auth)

with open('web_server.py', 'w') as f:
    f.write(content)
