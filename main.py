#!/usr/bin/env python3
"""
YouTrack MCP Server - A Model Context Protocol server for JetBrains YouTrack.
"""
import argparse
import logging
import os
import signal
import sys
from typing import Dict, Any, Optional
import json
from contextlib import asynccontextmanager
import uuid

# Try importing nest_asyncio but don't fail if it's not available
try:
    import nest_asyncio
    nest_asyncio.apply()
    logger = logging.getLogger(__name__)
    logger.info("Successfully applied nest_asyncio")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("nest_asyncio not available, event loop nesting may cause issues")

# App version - easy to find and update
APP_VERSION = os.getenv("APP_VERSION", "0.3.7")


from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from youtrack_mcp.config import Config, config
from youtrack_mcp.server import YouTrackMCPServer
from youtrack_mcp.tools.loader import load_all_tools

from fastapi.responses import StreamingResponse
import asyncio
from typing import AsyncGenerator
import time
from fastapi import Response

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global server and tools instances
server = None
tools = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI application."""
    global tools, server
    
    # Load configuration
    load_config()
    
    # Initialize MCP server with HTTP transport
    server = YouTrackMCPServer(transport="http")
    
    # Load all tools
    all_tools = load_all_tools()
    tools = all_tools
    
    # Register the tools with the server
    server.register_loaded_tools(all_tools)
    
    logger.info(f"HTTP server started with {len(all_tools)} tools")
    
    yield
    
    # Cleanup when the application is shutting down
    logger.info("Shutting down HTTP server")

# FastAPI app for HTTP mode
app = FastAPI(
    title="YouTrack MCP Server",
    description="MCP Server for JetBrains YouTrack",
    version=APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class ToolRequest(BaseModel):
    name: str = Field(..., description="The name of the tool to execute")
    arguments: Dict[str, Any] = Field(default={}, description="Arguments for the tool")

class ToolResponse(BaseModel):
    result: Any = Field(..., description="Result of the tool execution")

# MCP Protocol Models
class MCPToolCall(BaseModel):
    tool: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default={}, description="Arguments for the tool")

class MCPListToolsRequest(BaseModel):
    """MCP request to list available tools."""
    pass

class MCPCallToolRequest(BaseModel):
    """MCP request to call a tool."""
    method: str = Field(..., description="Method name (e.g., 'tools/call')")
    params: Dict[str, Any] = Field(..., description="Request parameters")

class MCPResponse(BaseModel):
    """MCP response format."""
    result: Any = Field(None, description="Result data")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information")

# MCP Endpoints for Langflow compatibility
@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """
    Main MCP endpoint for Langflow integration.
    Handles MCP protocol messages including tools/list and tools/call.
    """
    try:
        body = await request.json()
        method = body.get("method", "")
        params = body.get("params", {})
        
        logger.info(f"MCP request: method={method}, params={params}")
        
        if method == "tools/list":
            # Return list of available tools in MCP format
            mcp_tools = []
            for name, tool_func in tools.items():
                tool_schema = {
                    "name": name,
                    "description": tool_func.__doc__ or "No description available",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                
                # Extract parameter information if available
                if hasattr(tool_func, "tool_definition") and isinstance(tool_func.tool_definition, dict):
                    definition = tool_func.tool_definition
                    tool_schema["description"] = definition.get("description", tool_schema["description"])
                    
                    # Add parameter schema
                    params_info = definition.get("parameters") or definition.get("parameter_descriptions")
                    if isinstance(params_info, dict):
                        for param_name, param_desc in params_info.items():
                            tool_schema["inputSchema"]["properties"][param_name] = {
                                "type": "string",
                                "description": str(param_desc)
                            }
                            if param_name not in tool_schema["inputSchema"]["required"]:
                                tool_schema["inputSchema"]["required"].append(param_name)
                    elif isinstance(params_info, list):
                        for param in params_info:
                            if isinstance(param, dict) and "name" in param:
                                tool_schema["inputSchema"]["properties"][param["name"]] = {
                                    "type": param.get("type", "string"),
                                    "description": param.get("description", "")
                                }
                                if param.get("required", False):
                                    tool_schema["inputSchema"]["required"].append(param["name"])
                
                mcp_tools.append(tool_schema)
            
            return {"result": {"tools": mcp_tools}}
            
        elif method == "tools/call":
            # Call a specific tool
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                return {
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: missing tool name"
                    }
                }
            
            if tool_name not in tools:
                return {
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found"
                    }
                }
            
            try:
                # Execute tool
                logger.info(f"Executing MCP tool: {tool_name} with arguments: {arguments}")
                result = tools[tool_name](**arguments)
                
                return {
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": str(result)
                            }
                        ]
                    }
                }
            except Exception as e:
                logger.exception(f"Error executing MCP tool {tool_name}")
                return {
                    "error": {
                        "code": -32603,
                        "message": f"Tool execution failed: {str(e)}"
                    }
                }
        else:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found"
                }
            }
            
    except Exception as e:
        logger.exception("Error processing MCP request")
        return {
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }

@app.get("/mcp/tools")
async def mcp_list_tools():
    """
    MCP-compatible endpoint to list all available tools.
    Returns tools in the format expected by MCP clients like Langflow.
    """
    try:
        mcp_tools = []
        for name, tool_func in tools.items():
            tool_schema = {
                "name": name,
                "description": tool_func.__doc__ or "No description available",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Extract parameter information if available
            if hasattr(tool_func, "tool_definition") and isinstance(tool_func.tool_definition, dict):
                definition = tool_func.tool_definition
                tool_schema["description"] = definition.get("description", tool_schema["description"])
                
                # Add parameter schema
                params_info = definition.get("parameters") or definition.get("parameter_descriptions")
                if isinstance(params_info, dict):
                    for param_name, param_desc in params_info.items():
                        tool_schema["inputSchema"]["properties"][param_name] = {
                            "type": "string",
                            "description": param_desc
                        }
                        tool_schema["inputSchema"]["required"].append(param_name)
                elif isinstance(params_info, list):
                    for param in params_info:
                        if isinstance(param, dict) and "name" in param:
                            tool_schema["inputSchema"]["properties"][param["name"]] = {
                                "type": param.get("type", "string"),
                                "description": param.get("description", "")
                            }
                            if param.get("required", False):
                                tool_schema["inputSchema"]["required"].append(param["name"])
            
            mcp_tools.append(tool_schema)
        
        return {"tools": mcp_tools}
    except Exception as e:
        logger.exception("Error listing MCP tools")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list tools: {str(e)}"}
        )

@app.post("/api/tools/{tool_name}")
async def execute_tool(tool_name: str, request: Request):
    """
    Execute a specific tool by name.
    
    Args:
        tool_name: Name of the tool to execute
        request: The request object containing tool arguments
        
    Returns:
        Tool execution result
    """
    try:
        # Get tool from registry
        if tool_name not in tools:
            return JSONResponse(
                status_code=404,
                content={"error": f"Tool '{tool_name}' not found"}
            )
        
        # Parse request body
        body = await request.json()
        arguments = body.get("arguments", {})
        
        # Execute tool
        logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")
        result = tools[tool_name](**arguments)
        
        return {"result": result}
    except Exception as e:
        logger.exception(f"Error executing tool {tool_name}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/tools")
async def list_tools():
    """
    List all available tools.
    Returns:
        List of available tools with their definitions (MCP-compatible array)
    """
    tool_list = []
    for name, tool_func in tools.items():
        tool_info = {"name": name}
        param_list = []
        if hasattr(tool_func, "tool_definition") and isinstance(tool_func.tool_definition, dict):
            definition = tool_func.tool_definition
            # Описание
            tool_info["description"] = definition.get("description", tool_func.__doc__ or "No description available")
            # Параметры
            params = definition.get("parameters") or definition.get("parameter_descriptions")
            if isinstance(params, dict):
                param_list = [
                    {"name": k, "description": v, "type": "string"} for k, v in params.items()
                ]
            elif isinstance(params, list):
                param_list = [p for p in params if isinstance(p, dict) and "name" in p and "description" in p]
        else:
            tool_info["description"] = tool_func.__doc__ or "No description available"
        tool_info["parameters"] = param_list
        tool_list.append(tool_info)
    return {"tools": tool_list}


# --- SSE message helper ---
def make_sse_message(data, event=None):
    msg = ""
    if event:
        msg += f"event: {event}\n"
    msg += f"data: {data}\n\n"
    return msg

@app.get("/sse")
@app.post("/sse")
@app.head("/sse")
async def sse_endpoint(request: Request):
    """
    SSE endpoint for MCP integration with Langflow.
    Provides server-sent events stream with MCP server information and tools.
    Supports GET, POST, and HEAD methods for compatibility with different clients.
    """
    # Handle HEAD requests quickly
    if request.method == "HEAD":
        return Response(
            content="",
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
            }
        )
    
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            logger.info("SSE event_generator: start")
            # Сначала отправляем endpoint
            session_id = str(uuid.uuid4())
            endpoint = f"/messages/?session_id={session_id}"
            logger.info(f"SSE send: endpoint: {endpoint}")
            yield make_sse_message(endpoint, event="endpoint")
            # Далее стандартные MCP события
            try:
                init_event = {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {'protocolVersion': '2024-11-05', 'capabilities': {}}}
                logger.info(f"SSE send: initialize: {init_event}")
                yield make_sse_message(json.dumps(init_event), event="initialize")
            except Exception as e:
                logger.exception(f"SSE error on yield initialize: {e}")
                raise

            try:
                capabilities = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": config.MCP_SERVER_NAME,
                            "version": APP_VERSION
                        }
                    }
                }
                logger.info(f"SSE send: capabilities: {capabilities}")
                yield make_sse_message(json.dumps(capabilities), event="capabilities")
            except Exception as e:
                logger.exception(f"SSE error on yield capabilities: {e}")
                raise

            logger.info(f"SSE tools dict: {len(tools)} tools: {list(tools.keys())}")
            try:
                if tools:
                    mcp_tools = []
                    for name, tool_func in tools.items():
                        tool_schema = {
                            "name": name,
                            "description": tool_func.__doc__ or "No description available",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                                "required": []
                            }
                        }
                        if hasattr(tool_func, "tool_definition") and isinstance(tool_func.tool_definition, dict):
                            definition = tool_func.tool_definition
                            tool_schema["description"] = definition.get("description", tool_schema["description"])
                            params_info = definition.get("parameters") or definition.get("parameter_descriptions")
                            if isinstance(params_info, dict):
                                for param_name, param_desc in params_info.items():
                                    tool_schema["inputSchema"]["properties"][param_name] = {
                                        "type": "string",
                                        "description": str(param_desc)
                                    }
                                    tool_schema["inputSchema"]["required"].append(param_name)
                            elif isinstance(params_info, list):
                                for param in params_info:
                                    if isinstance(param, dict) and "name" in param:
                                        tool_schema["inputSchema"]["properties"][param["name"]] = {
                                            "type": param.get("type", "string"),
                                            "description": param.get("description", "")
                                        }
                                        if param.get("required", False):
                                            tool_schema["inputSchema"]["required"].append(param["name"])
                        mcp_tools.append(tool_schema)
                    tools_list_response = {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "result": {
                            "tools": mcp_tools
                        }
                    }
                    logger.info(f"SSE send: tools: {tools_list_response}")
                    yield make_sse_message(json.dumps(tools_list_response), event="tools")
            except Exception as e:
                logger.exception(f"SSE error on yield tools: {e}")
                raise

            heartbeat_count = 0
            while True:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                heartbeat_count += 1
                heartbeat = {
                    "jsonrpc": "2.0",
                    "method": "notifications/ping",
                    "params": {
                        "timestamp": time.time(),
                        "count": heartbeat_count
                    }
                }
                try:
                    logger.info(f"SSE send: heartbeat: {heartbeat}")
                    yield make_sse_message(json.dumps(heartbeat), event="heartbeat")
                except Exception as e:
                    logger.exception(f"SSE error on yield heartbeat: {e}")
                    raise
                
        except asyncio.CancelledError:
            logger.info("SSE client disconnected")
            return
        except Exception as e:
            logger.exception(f"Error in SSE event generator: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
            yield make_sse_message(json.dumps(error_response), event="error")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )


@app.post("/sse/execute/{tool_name}")
async def sse_execute_tool(tool_name: str, request: Request):
    """
    Execute a specific tool via SSE interface.
    This endpoint is designed for n8n MCP integration.

    Args:
        tool_name: Name of the tool to execute
        request: The request object containing tool arguments

    Returns:
        Tool execution result in n8n-compatible format
    """
    try:
        # Check if tool exists
        if tool_name not in tools:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": list(tools.keys())
                }
            )

        # Parse request body
        body = await request.json()

        # Extract arguments - n8n might send them in different formats
        arguments = body.get("arguments", body.get("args", body.get("params", {})))

        # Log for debugging
        logger.info(f"SSE Execute: {tool_name} with arguments: {arguments}")

        # Execute tool
        try:
            result = tools[tool_name](**arguments)

            # Format response for n8n
            response = {
                "success": True,
                "tool": tool_name,
                "result": result,
                "timestamp": int(time.time())
            }

            # If result is a string that looks like JSON, try to parse it
            if isinstance(result, str) and result.strip().startswith('{'):
                try:
                    parsed_result = json.loads(result)
                    response["result"] = parsed_result
                except json.JSONDecodeError:
                    # Keep original string if parsing fails
                    pass

            return response

        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "tool": tool_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )

    except Exception as e:
        logger.exception(f"Error in SSE execute endpoint")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/sse/tools")
async def sse_list_tools():
    """
    List all available tools in n8n-compatible format.
    """
    tool_list = []

    for name, tool_func in tools.items():
        tool_info = {
            "name": name,
            "description": tool_func.__doc__ or "No description available"
        }

        # Add tool definition if available - safely
        if hasattr(tool_func, "tool_definition"):
            definition = tool_func.tool_definition
            if isinstance(definition, dict):
                # Only include specific safe fields
                if "parameters" in definition and isinstance(definition["parameters"], dict):
                    tool_info["parameters"] = definition["parameters"]
                if "parameter_descriptions" in definition and isinstance(definition["parameter_descriptions"], dict):
                    tool_info["parameter_descriptions"] = definition["parameter_descriptions"]
                if "description" in definition and isinstance(definition["description"], str):
                    tool_info["description"] = definition["description"]

        # Try to extract parameter info from docstring
        if hasattr(tool_func, "__wrapped__"):
            original_func = tool_func.__wrapped__
            if hasattr(original_func, "__doc__") and original_func.__doc__:
                # Extract first line as description
                lines = original_func.__doc__.strip().split('\n')
                if lines and lines[0]:
                    tool_info["description"] = lines[0].strip()

        tool_list.append(tool_info)

    return {
        "tools": tool_list,
        "count": len(tool_list),
        "server": {
            "name": config.MCP_SERVER_NAME,
            "version": APP_VERSION
        }
    }


@app.options("/sse")
async def sse_options():
    """Handle preflight CORS requests for SSE endpoint."""
    return Response(
        content="",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
            "Access-Control-Max-Age": "86400",
        }
    )


@app.options("/sse/execute/{tool_name}")
async def sse_execute_options(tool_name: str):
    """Handle OPTIONS request for SSE execute endpoint."""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )


@app.post("/sse/call")
async def sse_call_tool(request: Request):
    """
    SSE endpoint for calling MCP tools.
    Handles tool execution requests from Langflow MCP client.
    """
    try:
        body = await request.json()
        method = body.get("method", "")
        params = body.get("params", {})
        request_id = body.get("id", 1)
        
        logger.info(f"SSE tool call: method={method}, params={params}")
        
        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: missing tool name"
                    }
                })
            
            if tool_name not in tools:
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{tool_name}' not found"
                    }
                })
            
            try:
                # Execute tool
                logger.info(f"Executing SSE tool: {tool_name} with arguments: {arguments}")
                result = tools[tool_name](**arguments)
                
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": str(result)
                            }
                        ]
                    }
                })
            except Exception as e:
                logger.exception(f"Error executing SSE tool {tool_name}")
                return JSONResponse({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": f"Tool execution failed: {str(e)}"
                    }
                })
        elif method == "tools/list":
            # Return list of available tools
            mcp_tools = []
            for name, tool_func in tools.items():
                tool_schema = {
                    "name": name,
                    "description": tool_func.__doc__ or "No description available",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
                
                # Extract parameter information if available
                if hasattr(tool_func, "tool_definition") and isinstance(tool_func.tool_definition, dict):
                    definition = tool_func.tool_definition
                    tool_schema["description"] = definition.get("description", tool_schema["description"])
                    
                    # Add parameter schema
                    params_info = definition.get("parameters") or definition.get("parameter_descriptions")
                    if isinstance(params_info, dict):
                        for param_name, param_desc in params_info.items():
                            tool_schema["inputSchema"]["properties"][param_name] = {
                                "type": "string",
                                "description": str(param_desc)
                            }
                            if param_name not in tool_schema["inputSchema"]["required"]:
                                tool_schema["inputSchema"]["required"].append(param_name)
                    elif isinstance(params_info, list):
                        for param in params_info:
                            if isinstance(param, dict) and "name" in param:
                                tool_schema["inputSchema"]["properties"][param["name"]] = {
                                    "type": param.get("type", "string"),
                                    "description": param.get("description", "")
                                }
                                if param.get("required", False):
                                    if param["name"] not in tool_schema["inputSchema"]["required"]:
                                        tool_schema["inputSchema"]["required"].append(param["name"])
                
                mcp_tools.append(tool_schema)
            
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": mcp_tools
                }
            })
        else:
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found"
                }
            })
            
    except Exception as e:
        logger.exception("Error processing SSE call request")
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }, status_code=500)


@app.post("/messages/")
async def messages_endpoint(request: Request, session_id: str = Query(None)):
    """
    MCP-compatible endpoint for POST /messages/?session_id=... (SSE MCP clients).
    Возвращает ответы строго в формате JSON-RPC 2.0 (jsonrpc, id, result/error).
    """
    try:
        body = await request.json()
        method = body.get("method", "")
        params = body.get("params", {})
        req_id = body.get("id", None)
        logger.info(f"/messages/ request: method={method}, params={params}, session_id={session_id}, id={req_id}")
        # MCP initialize
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": config.MCP_SERVER_NAME, "version": APP_VERSION}
            }
            response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            logger.info(f"/messages/ response: {response}")
            return JSONResponse(response)
        # MCP tools/list
        elif method == "tools/list":
            mcp_tools = []
            for name, tool_func in tools.items():
                tool_schema = {
                    "name": name,
                    "description": tool_func.__doc__ or "No description available",
                    "inputSchema": {"type": "object", "properties": {}, "required": []}
                }
                if hasattr(tool_func, "tool_definition") and isinstance(tool_func.tool_definition, dict):
                    definition = tool_func.tool_definition
                    tool_schema["description"] = definition.get("description", tool_schema["description"])
                    params_info = definition.get("parameters") or definition.get("parameter_descriptions")
                    if isinstance(params_info, dict):
                        for param_name, param_desc in params_info.items():
                            tool_schema["inputSchema"]["properties"][param_name] = {"type": "string", "description": str(param_desc)}
                            tool_schema["inputSchema"]["required"].append(param_name)
                    elif isinstance(params_info, list):
                        for param in params_info:
                            if isinstance(param, dict) and "name" in param:
                                tool_schema["inputSchema"]["properties"][param["name"]] = {"type": param.get("type", "string"), "description": param.get("description", "")}
                                if param.get("required", False):
                                    tool_schema["inputSchema"]["required"].append(param["name"])
                mcp_tools.append(tool_schema)
            response = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": mcp_tools}}
            logger.info(f"/messages/ response: {response}")
            return JSONResponse(response)
        # MCP tools/call
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            if not tool_name:
                response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Invalid params: missing tool name"}}
                logger.info(f"/messages/ response: {response}")
                return JSONResponse(response)
            if tool_name not in tools:
                response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Tool '{tool_name}' not found"}}
                logger.info(f"/messages/ response: {response}")
                return JSONResponse(response)
            try:
                logger.info(f"/messages/ executing tool: {tool_name} with arguments: {arguments}")
                tool_func = tools[tool_name]
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**arguments)
                else:
                    result = tool_func(**arguments)
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": str(result)}]}}
                logger.info(f"/messages/ response: {response}")
                return JSONResponse(response)
            except Exception as e:
                logger.exception(f"Error executing MCP tool {tool_name}")
                response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": f"Tool execution failed: {str(e)}"}}
                logger.info(f"/messages/ response: {response}")
                return JSONResponse(response)
        # Неизвестный метод
        else:
            response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method '{method}' not found"}}
            logger.info(f"/messages/ response: {response}")
            return JSONResponse(response)
    except Exception as e:
        logger.exception("Error processing /messages/ request")
        response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": f"Internal error: {str(e)}"}}
        logger.info(f"/messages/ response: {response}")
        return JSONResponse(response)


@app.get("/tools")
async def tools_alias():
    return await list_tools()

@app.get("/mcp/tools")
async def mcp_tools_alias():
    return await list_tools()


def load_config():
    """Load configuration from environment variables or file."""
    # Environment variables have higher priority than config file
    env_config = {}
    
    # Extract config variables from environment
    for key in dir(Config):
        if key.isupper() and not key.startswith("_"):
            env_key = f"YOUTRACK_MCP_{key}"
            if env_key in os.environ:
                env_value = os.environ[env_key]
                # Convert string booleans to actual booleans
                if env_value.lower() in ("true", "false"):
                    env_value = env_value.lower() == "true"
                env_config[key] = env_value
    
    # Create config instance from environment variables
    if env_config:
        logger.info("Loading configuration from environment variables")
        Config.from_dict(env_config)
    
    # Ensure token is properly formatted for YouTrack Cloud
    if config.YOUTRACK_API_TOKEN and not config.YOUTRACK_API_TOKEN.startswith(("perm:", "perm-")):
        # Check if we need to add the perm- prefix
        if "." in config.YOUTRACK_API_TOKEN and "=" in config.YOUTRACK_API_TOKEN:
            config.YOUTRACK_API_TOKEN = f"perm-{config.YOUTRACK_API_TOKEN}"
            logger.info("Added 'perm-' prefix to the API token")
        else:
            # For traditional tokens
            config.YOUTRACK_API_TOKEN = f"perm:{config.YOUTRACK_API_TOKEN}"
            logger.info("Added 'perm:' prefix to the API token")
    
    # Force YouTrack URL to be properly formatted
    if config.YOUTRACK_URL and config.YOUTRACK_URL.endswith("/"):
        config.YOUTRACK_URL = config.YOUTRACK_URL.rstrip("/")
        logger.info("Removed trailing slash from YouTrack URL")
    
    # Initialize configuration from environment variables
    config.validate()
    
    # Use environment variable for URL if specified instead of auto-detection
    if os.getenv("YOUTRACK_URL") and not config.YOUTRACK_URL:
        logger.info(f"Using URL from environment: {os.getenv('YOUTRACK_URL')}")
        config.YOUTRACK_URL = os.getenv("YOUTRACK_URL")
    
    # Log configuration status
    if config.YOUTRACK_URL:
        logger.info(f"Configured for YouTrack instance at: {config.YOUTRACK_URL}")
    else:
        logger.info("Configured for YouTrack Cloud instance")
    
    logger.info(f"SSL verification: {'Enabled' if config.VERIFY_SSL else 'Disabled'}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YouTrack MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to (HTTP mode only)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to (HTTP mode only)")
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--youtrack-url", 
        help="YouTrack instance URL (not required for YouTrack Cloud)"
    )
    parser.add_argument(
        "--api-token", 
        help="YouTrack API token for authentication"
    )
    parser.add_argument(
        "--verify-ssl",
        action="store_true",
        default=None,
        help="Verify SSL certificates (default: True)"
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_false",
        dest="verify_ssl",
        help="Disable SSL certificate verification"
    )
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="stdio",
        help="Transport mode: 'stdio' for Claude integration (default), 'http' for API server"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Display version information and exit"
    )
    
    return parser.parse_args()

def apply_cli_args(args):
    """Apply command line arguments to configuration."""
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Apply YouTrack configuration
    config_dict = {}
    
    if args.youtrack_url:
        config_dict["YOUTRACK_URL"] = args.youtrack_url
    
    if args.api_token:
        config_dict["YOUTRACK_API_TOKEN"] = args.api_token
    
    if args.verify_ssl is not None:
        config_dict["VERIFY_SSL"] = args.verify_ssl
    
    if config_dict:
        Config.from_dict(config_dict)

def handle_signal(signum: int, frame) -> None:
    """
    Handle termination signals.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logging.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Run the MCP server."""
    args = parse_args()
    
    # Check if version information was requested
    if args.version:
        print(f"YouTrack MCP Server v{APP_VERSION}")
        sys.exit(0)
    
    # Apply command line arguments
    apply_cli_args(args)
    
    # Load configuration
    load_config()
    
    # Log version information
    logger.info(f"Starting YouTrack MCP Server v{APP_VERSION}")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Check if running in HTTP mode
    if args.transport == "http":
        logger.info(f"Starting HTTP server on {args.host}:{args.port}")
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    else:
        # Initialize MCP server with stdio transport
        server = YouTrackMCPServer(transport="stdio")
        
        # Load all tools just once
        all_tools = load_all_tools()
        
        # Register the tools with the server
        server.register_loaded_tools(all_tools)
        
        # Run the server directly in stdio mode
        logger.info("Starting in stdio mode for Cursor/Claude integration")
        server.run()

if __name__ == "__main__":
    main() 