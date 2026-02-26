import os
import re
import httpx

from dataclasses import dataclass
from pydantic import Field
from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv
from httpx import AsyncClient
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.deepseek import DeepSeekProvider
from typing import Optional

load_dotenv()

############################ Model ########################

api_key_1: Optional[str] = os.getenv("DEEPSEEK_API_KEY")

model = OpenAIChatModel(
    "deepseek-chat",
    provider=DeepSeekProvider(api_key=f"{api_key_1}")
)

@dataclass
class Deps:
    client: httpx.AsyncClient = Field(default_factory=AsyncClient)
    github_token: Optional[str] = Field(default_factory=lambda: os.getenv("GITHUB_TOKEN"))

github_agent = Agent[Deps](
    model=model,
     system_prompt = """ You are a coding expert with access to GitHub to help the user
manage their repositories and get information about them. Your only job is to assist with
this and you don't answer other questions besides describing what you are able to do.
Don't ask the user before taking an action, just do it. Always make sure you look at
the repository with the provided tools before answering.
When answering a question about the repo, slways ster you answer with the full repo URL.""",
    deps_type=Deps,
    retries=2,
)

@github_agent.tool
async def get_repo_info(ctx: RunContext[Deps], github_url: str) -> str:
    """Get repository information from GitHub API.
    
    Args:
        ctx (RunContext[Deps]): The run context containing dependencies.
        github_url (str): The URL of the GitHub repository.
        
    Returns:
        str: A summary of the repository information.
    """
    match = re.search(r"github\.com/([^/]+)/([^/]+)", github_url)
    if not match:
        return "Invalid GitHub URL format. Please provide a URL like"
    owner, repo = match.groups()
    headers = {"Authorization": f"token {ctx.deps.github_token}"} if ctx.deps.github_token else {}
    response = await ctx.deps.client.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers)
    if response.status_code != 200:
        return f"Failed to fetch repository information: {response.status_code} - {response.text}"
    data = response.json()
    size_mb = data["size"] / 1024
    return (
        f"Repository '{data['full_name']}' has {data['stargazers_count']} stars, "
        f"{data['forks_count']} forks, and is {size_mb:.2f} MB in size. "
        f"URL: {data['html_url']}"
    )

@github_agent.tool
async def get_repo_structure(ctx: RunContext[Deps], github_url: str) -> str:
    """Get the directory structure of a GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.

    Returns:
        str: Directory structure as a formatted string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1',
        headers=headers
    )
    
    if response.status_code != 200:
        # Try with master branch if main fails
        response = await ctx.deps.client.get(
            f'https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1',
            headers=headers
        )
        if response.status_code != 200:
            return f"Failed to get repository structure: {response.text}"
    
    data = response.json()
    tree = data['tree']
    
    # Build directory structure
    structure = []
    for item in tree:
        if not any(excluded in item['path'] for excluded in ['.git/', 'node_modules/', '__pycache__/']):
            structure.append(f"{'📁 ' if item['type'] == 'tree' else '📄 '}{item['path']}")
    
    return "\n".join(structure)

@github_agent.tool
async def get_file_content(ctx: RunContext[Deps], github_url: str, file_path: str) -> str:
    """Get the content of a specific file from the GitHub repository.

    Args:
        ctx: The context.
        github_url: The GitHub repository URL.
        file_path: Path to the file within the repository.

    Returns:
        str: File content as a string.
    """
    match = re.search(r'github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$', github_url)
    if not match:
        return "Invalid GitHub URL format"
    
    owner, repo = match.groups()
    headers = {'Authorization': f'token {ctx.deps.github_token}'} if ctx.deps.github_token else {}
    
    response = await ctx.deps.client.get(
        f'https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}',
        headers=headers
    )
    
    if response.status_code != 200:
        # Try with master branch if main fails
        response = await ctx.deps.client.get(
            f'https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}',
            headers=headers
        )
        if response.status_code != 200:
            return f"Failed to get file content: {response.text}"
    
    return response.text