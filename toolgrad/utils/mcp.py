# from langchain.tools import StructuredTool
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None
from langchain_core.tools import StructuredTool, ToolException
from typing import Any, Callable, List, Optional
import asyncio
import random
import os


def get_default_mcp_dict() -> dict:
  filesystem_path = get_example_filesystem_dir()
  return {
      "filesystem": {
          "command": "npx",
          "args": [
              "-y",
              "@modelcontextprotocol/server-filesystem",
              filesystem_path,
          ],
          "transport": "stdio",
      },
  }


def get_example_filesystem_dir() -> str:
  import toolgrad
  toolgrad_root = toolgrad.__path__[0]
  filesystem_path = os.path.join(toolgrad_root, "utils", "filesystem")
  return filesystem_path


# Read-only access
ALLOWED_APIS = [
    'read_file', 'list_directory', 'read_text_file', 'directory_tree',
    'read_multiple_files'
]


class JSONWrappingTool(StructuredTool):

  def run(self, tool_input, config=None, **kwargs):
    try:
      result = super().run(tool_input, config, **kwargs)
      # Handle MCP results that may be arrays of content blocks
      if isinstance(result, list):
        # Extract text content from MCP response format
        content = "\n".join([
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in result
        ])
        return {"error": "", "response": content}
      return {"error": "", "response": str(result)}
    except ToolException as te:
      return {"error": str(te), "response": ""}
    except Exception as e:
      return {"error": str(e), "response": ""}

  async def arun(self, tool_input, config=None, **kwargs):
    try:
      result = await super().arun(tool_input, config=config, **kwargs)
      # Handle MCP results that may be arrays of content blocks
      if isinstance(result, list):
        # Extract text content from MCP response format
        content = "\n".join([
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in result
        ])
        return {"error": "", "response": content}
      return {"error": "", "response": str(result)}
    except ToolException as te:
      return {"error": str(te), "response": ""}
    except Exception as e:
      return {"error": str(e), "response": ""}


def _wrap_with_path_prefix(tool: StructuredTool, base_path: str) -> StructuredTool:
  """Wrap a tool's function to automatically prepend base_path to 'path' or 'paths' arguments if they are relative."""
  
  original_func = tool.func
  original_coroutine = tool.coroutine if hasattr(tool, 'coroutine') else None

  def wrap_args(kwargs):
    new_kwargs = kwargs.copy()
    if 'path' in new_kwargs and isinstance(new_kwargs['path'], str):
      if not os.path.isabs(new_kwargs['path']):
        new_kwargs['path'] = os.path.join(base_path, new_kwargs['path'])
    if 'paths' in new_kwargs and isinstance(new_kwargs['paths'], list):
       new_kwargs['paths'] = [os.path.join(base_path, p) if not os.path.isabs(p) else p for p in new_kwargs['paths']]
    return new_kwargs

  if original_func is None and original_coroutine is not None:
    # Async-only MCP tools have no sync callable. Run the coroutine
    # synchronously (worker thread when already inside an event loop) so
    # sync ``invoke()`` paths work too.
    import asyncio as _asyncio
    import threading as _threading

    def new_func(**kwargs):
      wrapped = wrap_args(kwargs)

      async def _call():
        return await original_coroutine(**wrapped)

      try:
        _asyncio.get_running_loop()
      except RuntimeError:
        return _asyncio.run(_call())
      box: dict = {}

      def _runner():
        try:
          box["result"] = _asyncio.run(_call())
        except BaseException as exc:  # noqa: BLE001 — re-raised below
          box["error"] = exc

      thread = _threading.Thread(target=_runner, daemon=True)
      thread.start()
      thread.join()
      if "error" in box:
        raise box["error"]
      return box.get("result")
  else:
    def new_func(**kwargs):
      return original_func(**wrap_args(kwargs))

  if original_coroutine:
    async def new_coroutine(**kwargs):
      return await original_coroutine(**wrap_args(kwargs))
  else:
    new_coroutine = None

  return StructuredTool(
      name=tool.name,
      description=tool.description,
      func=new_func,
      coroutine=new_coroutine,
      args_schema=tool.args_schema,
  )


def discover_mcp_tools(mcp_dict: dict) -> List[StructuredTool]:
  """Discover MCP tools: connect, filter to ``ALLOWED_APIS``, wrap each to
  prepend the example filesystem path. Returns the full tool list (no
  sampling); shared by ``get_mcp_apis`` and external callers that need the
  catalog (e.g. to build a tool knowledge graph before sampling)."""
  client = MultiServerMCPClient(mcp_dict)
  tools = asyncio.run(client.get_tools())
  tools = [tool for tool in tools if tool.name in ALLOWED_APIS]

  # Wrap tools to prepend filesystem path
  filesystem_path = get_example_filesystem_dir()
  tools = [_wrap_with_path_prefix(tool, filesystem_path) for tool in tools]
  print(f"Found {len(tools)} APIs in MCP.")
  return tools


def get_mcp_apis(
    mcp_dict: dict,
    num_apis: int = 5,
    seed: int = 42,
    sampler: Optional[Callable[[List[StructuredTool], int],
                               List[StructuredTool]]] = None,
) -> list[StructuredTool]:
  """Discover MCP tools and return a sample of ``num_apis`` of them.

  Args:
    sampler: Optional override for the sampling step. Receives the full
      discovered (and path-prefixed) tool list and ``num_apis``, and returns
      the chosen subset. When None (default), uniform ``random.sample``
      with ``seed`` is used, preserving the original behavior.
  """
  tools = discover_mcp_tools(mcp_dict)

  if num_apis > len(tools) or num_apis <= 0:
    raise ValueError(
        f"num_apis should be between 1 and {len(tools)}, got {num_apis}.")
  if sampler is not None:
    return sampler(tools, num_apis)
  random.seed(seed)
  return random.sample(tools, num_apis)
