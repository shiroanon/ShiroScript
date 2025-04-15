#!/usr/bin/env python3

import json
import re
import requests
import sys
import traceback
import argparse
from typing import Dict, Any, List, Optional, Tuple, Set

# --- Configuration ---
DEFAULT_SHIROUI_URL = "https://359b94cb65d4e8c57a5472c2980d25e9.loophole.site" # Default ComfyUI/ShiroUI port

# --- Node Info Fetching ---# --- Node Info Fetching (Identical to the original script) ---

import time
import logging
from urllib.parse import urljoin

# --- Configuration ---

# Configure logging (you might want to adjust this in a larger application)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    stream=sys.stderr  # Log to stderr like the original print statements
)
logger = logging.getLogger(__name__)

# Retry settings
MAX_RETRIES = 3  # Number of retries after the initial attempt
INITIAL_BACKOFF_SECONDS = 1.0  # Initial delay before the first retry
REQUEST_TIMEOUT_SECONDS = 15  # Timeout for each individual request attempt

# --- Helper Function (Optional but good practice) ---

def _is_retryable_error(exception: Exception) -> bool:
    """Checks if an exception indicates a potentially transient error worth retrying."""
    if isinstance(exception, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exception, requests.exceptions.HTTPError):
        # Retry on server errors (5xx), but not client errors (4xx)
        return 500 <= exception.response.status_code < 600
    # Could potentially add other specific transient errors if needed
    # For example, some RequestException subtypes might be retryable
    # if isinstance(exception, requests.exceptions.ChunkedEncodingError):
    #     return True
    return False

# --- Main Function ---

def fetch_nodes_info(shiro_url: str) -> Optional[Dict[str, Any]]:
    """
    Fetches node schema information from the RUNNING ShiroUI backend with retries.

    Args:
        shiro_url: The base URL of the ShiroUI/ComfyUI backend (e.g., "http://127.0.0.1:8188").

    Returns:
        A dictionary containing the node information if successful, otherwise None.
    """
    # 1. Normalize and validate URL
    if not isinstance(shiro_url, str) or not shiro_url.strip():
        logger.error("Invalid ShiroUI URL provided: URL must be a non-empty string.")
        return None

    original_url = shiro_url
    if not shiro_url.startswith(("http://", "https://")):
        shiro_url = f"http://{shiro_url}"
        logger.warning(f"URL scheme missing for '{original_url}', assuming http://. Using: {shiro_url}")

    # Ensure trailing slash for urljoin robustness
    if not shiro_url.endswith('/'):
        shiro_url += '/'

    try:
        # Use urljoin for safer path combination
        object_info_url = urljoin(shiro_url, "object_info")
    except ValueError as e:
        logger.error(f"Invalid base URL format '{shiro_url}': {e}")
        return None

    logger.info(f"Attempting to fetch node info from: {object_info_url}")

    current_delay = INITIAL_BACKOFF_SECONDS
    last_exception: Optional[Exception] = None

    # Use a session for potential connection reuse and configuration
    with requests.Session() as session:
        for attempt in range(MAX_RETRIES + 1): # Initial attempt + MAX_RETRIES
            is_last_attempt = (attempt == MAX_RETRIES)
            log_prefix = f"Attempt {attempt + 1}/{MAX_RETRIES + 1}"

            try:
                response = session.get(object_info_url, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

                # Success path
                data = response.json()
                logger.info(f"{log_prefix}: Successfully fetched and decoded node info.")
                return data

            except requests.exceptions.Timeout as e:
                logger.warning(f"{log_prefix}: Timeout connecting to {object_info_url} after {REQUEST_TIMEOUT_SECONDS}s.")
                last_exception = e
                if is_last_attempt or not _is_retryable_error(e): break

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"{log_prefix}: Connection error to {shiro_url}. Is it running? Details: {e}")
                last_exception = e
                if is_last_attempt or not _is_retryable_error(e): break

            except requests.exceptions.HTTPError as e:
                if _is_retryable_error(e):
                    logger.warning(f"{log_prefix}: Received server error ({e.response.status_code}) from {object_info_url}. Retrying...")
                    last_exception = e
                    # Fall through to retry delay if not last attempt
                else:
                    # Non-retryable HTTP error (e.g., 404 Not Found, 401 Unauthorized)
                    logger.error(f"{log_prefix}: HTTP Client Error: {e.response.status_code} fetching from {object_info_url}.")
                    try:
                        logger.error(f"Response body: {e.response.text}")
                    except Exception: pass # Avoid errors if text isn't available/decodable
                    return None # Definite failure, do not retry

                if is_last_attempt: break # Exit loop after logging warning/error

            except requests.exceptions.RequestException as e:
                # Catch other potential request exceptions
                logger.warning(f"{log_prefix}: Network request failed: {e}")
                last_exception = e
                # Decide if generic RequestExceptions are retryable (can be broad)
                # Let's retry them by default unless it's the last attempt
                if is_last_attempt: break

            except json.JSONDecodeError as e:
                # Server responded, but with invalid JSON. Unlikely to be fixed by retry.
                status_code = getattr(locals().get('response'), 'status_code', 'N/A')
                logger.error(f"{log_prefix}: Failed to decode JSON response from {object_info_url} (Status: {status_code}). Error: {e}")
                try:
                     # Only log response text if response variable exists and is not None
                     if 'response' in locals() and response is not None:
                         logger.error(f"Response text: {response.text[:500]}...") # Log snippet
                except Exception: pass
                return None # Definite failure

            except Exception as e:
                # Catch truly unexpected errors during the request/parsing phase
                logger.error(f"{log_prefix}: An unexpected error occurred: {e}", exc_info=False) # Set exc_info=True for full traceback if desired
                traceback.print_exc(file=sys.stderr) # Keep explicit traceback for unexpected errors
                last_exception = e
                return None # Unexpected error, stop trying

            # --- Retry Logic ---
            if not is_last_attempt:
                logger.info(f"Waiting {current_delay:.2f} seconds before next attempt...")
                time.sleep(current_delay)
                current_delay *= 2  # Exponential backoff (e.g., 1s, 2s, 4s, ...)

    # If the loop completes without returning, all attempts failed
    logger.error(f"Failed to fetch node info from {object_info_url} after {MAX_RETRIES + 1} attempts.")
    if last_exception:
        logger.error(f"Last encountered error: {type(last_exception).__name__}: {last_exception}")

    return None

# --- Parser Class ---

class ShiroParserError(Exception):
    """Custom exception for parsing errors, including line number."""
    def __init__(self, message: str, line_number: int):
        super().__init__(f"Error on line {line_number}: {message}")
        self.line_number = line_number

class ShiroDirectParser:
    """Parses .shiro text format directly into ShiroUI/ComfyUI JSON."""

    NODE_START_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s*=\s*([a-zA-Z_]\w*)\s*\(")
    ARG_SPLIT_RE = re.compile(r"\s*([a-zA-Z_]\w*)\s*=\s*(.+?)\s*(?:,|$|\))")
    NUMBER_RE = re.compile(r"^-?(\d+\.\d+|\.\d+|\d+\.?)$")
    BOOL_RE = re.compile(r"^(true|false)$")
    STRING_RE = re.compile(r'^"(.*)"$')
    NAMED_CONN_RE = re.compile(r"^([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)$")
    SIMPLE_CONN_RE = re.compile(r"^([a-zA-Z_]\w*)$")

    def __init__(self, nodes_info: Dict[str, Any]):
        if not nodes_info:
            raise ValueError("nodes_info cannot be empty.")
        self.nodes_info = nodes_info
        self.node_class_types: Set[str] = set(nodes_info.keys())

        self._json_nodes: Dict[str, Dict[str, Any]] = {}
        self._variable_map: Dict[str, Tuple[str, str]] = {} # var_name -> (node_id, node_type)
        self._node_id_counter: int = 0
        self._current_line_num: int = 0

    def _error(self, message: str) -> ShiroParserError:
        return ShiroParserError(message, self._current_line_num)

    def _get_new_node_id(self) -> str:
        node_id = str(self._node_id_counter)
        self._node_id_counter += 1
        return node_id

    def _get_output_slot_index(self, node_type: str, output_name: str) -> int:
        node_info = self.nodes_info.get(node_type)
        if not node_info: raise self._error(f"Internal error: Node info not found for type '{node_type}' during slot lookup.")
        available_names = node_info.get('output_name') or node_info.get('output', [])
        if not available_names: raise self._error(f"Node type '{node_type}' has no defined outputs in schema.")
        try: return available_names.index(output_name)
        except ValueError: raise self._error(f"Node type '{node_type}' has no output named '{output_name}'. Available outputs: {available_names}") from None

    def _parse_value_string(self, value_str: str, input_key: str) -> Any:
        value_str = value_str.strip()
        if (match := self.NUMBER_RE.match(value_str)): num_str = match.group(1); return int(num_str) if '.' not in num_str else float(num_str)
        if (match := self.BOOL_RE.match(value_str)): return True if match.group(1) == "true" else False
        if (match := self.STRING_RE.match(value_str)): return match.group(1).encode('utf-8').decode('unicode_escape')
        if (match := self.NAMED_CONN_RE.match(value_str)):
            var_name, output_name = match.groups()
            if var_name not in self._variable_map: raise self._error(f"Undefined variable '{var_name}' used in connection for input '{input_key}'.")
            source_node_id, source_node_type = self._variable_map[var_name]; slot_index = self._get_output_slot_index(source_node_type, output_name); return [source_node_id, slot_index]
        if (match := self.SIMPLE_CONN_RE.match(value_str)):
            var_name = match.group(1)
            if var_name not in self._variable_map: raise self._error(f"Undefined variable '{var_name}' used in connection for input '{input_key}'.")
            source_node_id, source_node_type = self._variable_map[var_name]; node_info = self.nodes_info.get(source_node_type)
            if not node_info: raise self._error(f"Internal error: Node info missing for source node type '{source_node_type}'.")
            num_outputs = len(node_info.get('output', []))
            if num_outputs == 1: return [source_node_id, 0]
            elif num_outputs == 0: raise self._error(f"Variable '{var_name}' (type '{source_node_type}') has no outputs and cannot be used as an input connection for '{input_key}'.")
            else: output_names = node_info.get('output_name') or node_info.get('output', []); raise self._error(f"Ambiguous connection: Variable '{var_name}' (type '{source_node_type}') has {num_outputs} outputs ({output_names or 'unnamed'}). Use '.OUTPUT_NAME' syntax for input '{input_key}'.")
        raise self._error(f"Cannot parse value '{value_str}' for input '{input_key}'. Expected number, boolean, \"string\", or connection (var / var.OUTPUT).")

    def _parse_arguments(self, args_str: str, node_type: str, defining_var: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}; schema_info = self.nodes_info[node_type]; schema_inputs = schema_info.get('input', {})
        # valid_input_keys = set(schema_inputs.get('required', {}).keys()) | set(schema_inputs.get('optional', {}).keys())
        last_match_end = 0
        for match in self.ARG_SPLIT_RE.finditer(args_str):
            gap_text = args_str[last_match_end:match.start()].strip()
            if gap_text and gap_text != ',': raise self._error(f"Invalid syntax near '{gap_text}' within arguments for node '{defining_var}'. Check commas.")
            key, value_str = match.groups()
            # if key not in valid_input_keys: print(f"Warning: Line {self._current_line_num}: Input key '{key}' not found in schema for node type '{node_type}'.", file=sys.stderr)
            if key in inputs: raise self._error(f"Duplicate input key '{key}' specified for node '{defining_var}'.")
            try: inputs[key] = self._parse_value_string(value_str, key)
            except ShiroParserError as e: raise e
            except Exception as e: raise self._error(f"Unexpected error parsing value for input '{key}': {e}") from e
            last_match_end = match.end()
        trailing_text = args_str[last_match_end:].strip()
        if trailing_text: raise self._error(f"Invalid trailing text '{trailing_text}' in arguments for node '{defining_var}'.")
        return inputs

    # --- CORRECTED parse method ---
    def parse(self, shiro_code: str) -> Dict[str, Any]:
        """Parses the entire .shiro code string and returns the JSON graph."""
        self._json_nodes = {}
        self._variable_map = {}
        self._node_id_counter = 0
        self._current_line_num = 0

        lines = shiro_code.splitlines()
        line_idx = 0
        while line_idx < len(lines):
            self._current_line_num = line_idx + 1 # Track line number for errors
            line = lines[line_idx].strip()

            if not line or line.startswith('#'):
                line_idx += 1
                continue # Skip blank lines and comments

            # Check if this line starts a node definition
            match = self.NODE_START_RE.match(line)
            if not match:
                if line: raise self._error(f"Invalid syntax. Expected 'variable = NodeType(...)' or comment/blank line. Found: '{line}'")
                else: line_idx += 1; continue # Should not happen if line is stripped, but safety

            var_name, node_type = match.groups()
            start_paren_pos = line.find('(')
            start_line_num_for_node = self._current_line_num

            # --- Corrected Multi-line Handling ---
            args_content_list = [line[start_paren_pos + 1:]] # Start content after '('
            paren_level = line.count('(') - line.count(')') # Initial level on first line
            found_end = False
            current_scan_line_idx = line_idx # Start scan from the current line

            # Check if complete on the first line
            if paren_level == 0 and line.rstrip().endswith(')'):
                 closing_paren_pos = line.rfind(')')
                 args_content_list[0] = args_content_list[0][:closing_paren_pos] # Trim trailing ')'
                 found_end = True
                 line_idx += 1 # Consume this line
            else:
                 # Multi-line definition or syntax error on first line
                 if paren_level < 0: # Closing paren without opening on first line
                      raise ShiroParserError(f"Mismatched closing parenthesis ')'.", self._current_line_num)

                 # Start scanning from the *next* line
                 current_scan_line_idx += 1
                 while current_scan_line_idx < len(lines):
                     self._current_line_num = current_scan_line_idx + 1 # Update line num for errors below
                     current_line = lines[current_scan_line_idx]

                     # Basic parenthesis counting (ignores strings/comments)
                     open_count = current_line.count('(')
                     close_count = current_line.count(')')
                     paren_level += open_count - close_count

                     if paren_level < 0:
                          raise ShiroParserError(f"Mismatched closing parenthesis ')'.", self._current_line_num)

                     if paren_level == 0:
                         # Found the potential end line
                         closing_paren_pos = current_line.rfind(')')
                         if closing_paren_pos != -1:
                             # Add content up to *but not including* the closing paren
                             args_content_list.append(current_line[:closing_paren_pos])
                             found_end = True
                             line_idx = current_scan_line_idx + 1 # Consume this line and move past it
                             break # Exit the inner while loop
                         else:
                             # Level is 0 but no ')'? Could be syntax error, add line and let arg parser fail
                             args_content_list.append(current_line)
                     else:
                         # Paren level > 0, still inside definition
                         args_content_list.append(current_line)

                     current_scan_line_idx += 1 # Move to next line

                 if not found_end:
                     raise ShiroParserError(f"Unmatched opening parenthesis '(' for node '{var_name}' starting on line {start_line_num_for_node}. Reached end of file.", start_line_num_for_node)
            # --- End parenthesis finding ---

            # Join lines and parse arguments
            args_str = " ".join(args_content_list).strip()

            # --- Perform validations and processing (reset line num context) ---
            # Use start line for definition errors, use end line for arg errors?
            self._current_line_num = start_line_num_for_node # Set context for validation errors below

            if node_type not in self.node_class_types:
                 raise self._error(f"Unknown node type '{node_type}'.")
            if var_name in self._variable_map:
                 raise self._error(f"Variable name '{var_name}' redefined.")

            try:
                # Parse the collected argument string
                # Argument parser will use the latest _current_line_num set inside the multi-line loop if applicable
                inputs = self._parse_arguments(args_str, node_type, var_name)
            except ShiroParserError as e:
                 raise e # Propagate error with correct line number
            except Exception as e:
                 # Use the line number where the multi-line block ended
                 self._current_line_num = line_idx # Or current_scan_line_idx if error was inside?
                 raise self._error(f"Unexpected error parsing arguments for '{var_name}': {e}") from e


            # Add node to results
            node_id = self._get_new_node_id()
            self._variable_map[var_name] = (node_id, node_type)
            self._json_nodes[node_id] = {
                "class_type": node_type,
                "inputs": inputs
            }

            # Ensure outer loop continues *after* the processed block
            # line_idx is already updated correctly above

        return self._json_nodes
def parse_code_str(url:str, shiro_code:str )->Dict[str, Any]:
    nodes_info=fetch_nodes_info(url)
    if not nodes_info: sys.exit(1)

    print("Parsing .shiro file...", file=sys.stderr)
    try:
        shiro_parser = ShiroDirectParser(nodes_info)
        generated_json = shiro_parser.parse(shiro_code)
        print("Parsing successful.", file=sys.stderr)
        return generated_json
    except ShiroParserError as e: print(f"\nParsing Error: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"\nAn unexpected error occurred during parsing:", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)
    
# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Parse a .shiro workflow file into ShiroUI/ComfyUI JSON.")
    parser.add_argument("shiro_file", help="Path to the .shiro workflow file.")
    parser.add_argument("--url", default=DEFAULT_SHIROUI_URL, help=f"URL of the running ShiroUI/ComfyUI backend (default: {DEFAULT_SHIROUI_URL}).")
    parser.add_argument("-o", "--output", help="Optional path to save the output JSON file.")
    args = parser.parse_args()

    nodes_info = fetch_nodes_info(args.url)
    if not nodes_info: sys.exit(1)

    try:
        with open(args.shiro_file, 'r', encoding='utf-8') as f: shiro_code = f.read()
        print(f"Read workflow from: {args.shiro_file}", file=sys.stderr)
    except FileNotFoundError: print(f"Error: Input file not found: '{args.shiro_file}'", file=sys.stderr); sys.exit(1)
    except IOError as e: print(f"Error reading file '{args.shiro_file}': {e}", file=sys.stderr); sys.exit(1)

    print("Parsing .shiro file...", file=sys.stderr)
    try:
        shiro_parser = ShiroDirectParser(nodes_info)
        generated_json = shiro_parser.parse(shiro_code)
        print("Parsing successful.", file=sys.stderr)
    except ShiroParserError as e: print(f"\nParsing Error: {e}", file=sys.stderr); sys.exit(1)
    except Exception as e: print(f"\nAn unexpected error occurred during parsing:", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.exit(1)

    json_output_string = json.dumps(generated_json, indent=2)

    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f: f.write(json_output_string)
            print(f"Generated JSON saved to: {args.output}", file=sys.stderr)
        except IOError as e: print(f"Error writing output file '{args.output}': {e}", file=sys.stderr); print("\n--- Generated ShiroUI JSON ---"); print(json_output_string); sys.exit(1)
    else:
        print("\n--- Generated ShiroUI JSON ---"); print(json_output_string)

    print("\n--- Conversion Complete ---", file=sys.stderr)

if __name__ == "__main__":
    main()