#!/usr/bin/env python3

import json
import re
import requests
import sys
import traceback
import argparse
import keyword
from typing import Dict, Any, List, Optional, Tuple, Set

# --- Configuration ---
DEFAULT_SHIROUI_URL = "https://359b94cb65d4e8c57a5472c2980d25e9.loophole.site" # Default ComfyUI/ShiroUI URL (added http://)

# --- Node Info Fetching (Identical to the original script) ---

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

# --- Shiro Generator Class ---

class ShiroGeneratorError(Exception):
    """Custom exception for generation errors."""
    pass

class JsonToShiroGenerator:
    """Converts ComfyUI JSON workflow to .shiro text format."""

    def __init__(self, nodes_info: Dict[str, Any]):
        if not nodes_info:
            raise ValueError("nodes_info cannot be empty.")
        self.nodes_info = nodes_info
        self.node_class_types: Set[str] = set(nodes_info.keys())

        # Internal state for generation
        self._workflow_json: Dict[str, Dict[str, Any]] = {}
        self._node_id_to_var_name: Dict[str, str] = {}
        self._generated_vars: Set[str] = set()
        self._var_name_counters: Dict[str, int] = {}
        self._shiro_lines: List[str] = []
        self._processed_nodes: Set[str] = set()

    def _sanitize_name(self, name: str) -> str:
        """Converts a string into a valid Python/Shiro variable name."""
        # Remove invalid characters (keep letters, numbers, underscores)
        s = re.sub(r'\W|^(?=\d)', '_', name)
        # Prepend underscore if it starts with a number (already handled by regex but belt-and-suspenders)
        if s and s[0].isdigit():
            s = '_' + s
        # Handle Python keywords
        if keyword.iskeyword(s):
            s += '_'
        # Make lowercase and replace spaces/dashes with underscores
        s = s.lower().replace(' ', '_').replace('-', '_')
        # Remove consecutive underscores
        s = re.sub(r'_+', '_', s).strip('_')
        # Ensure it's not empty
        if not s:
            return "var"
        return s

    def _get_unique_variable_name(self, node_id: str, node_data: Dict[str, Any]) -> str:
        """Generates a unique variable name for a node."""
        base_name = "unnamed_node"
        # Try using title from _meta if available
        if "_meta" in node_data and "title" in node_data["_meta"]:
             base_name = self._sanitize_name(node_data["_meta"]["title"])
        # Fallback to class_type if title wasn't useful or available
        if base_name == "var" or not base_name: # check if sanitize resulted in empty or default
            base_name = self._sanitize_name(node_data.get("class_type", "UnknownType"))

        # Ensure uniqueness
        if base_name not in self._var_name_counters:
            self._var_name_counters[base_name] = 0
            var_name = base_name
        else:
            self._var_name_counters[base_name] += 1
            var_name = f"{base_name}_{self._var_name_counters[base_name]}"

        # Final check to prevent accidental collisions (though unlikely with counters)
        while var_name in self._generated_vars:
             self._var_name_counters[base_name] += 1
             var_name = f"{base_name}_{self._var_name_counters[base_name]}"

        self._generated_vars.add(var_name)
        return var_name

    def _get_output_name_by_index(self, node_type: str, index: int) -> Optional[str]:
        """Finds the name of an output slot given its index."""
        node_info = self.nodes_info.get(node_type)
        if not node_info:
            print(f"Warning: Schema info not found for node type '{node_type}' during output name lookup.", file=sys.stderr)
            return None
        # ComfyUI schema uses 'output_name' primarily, fallback to 'output'
        output_names = node_info.get('output_name') or node_info.get('output', [])
        if index < 0 or index >= len(output_names):
            print(f"Warning: Output index {index} out of bounds for node type '{node_type}' (outputs: {output_names}).", file=sys.stderr)
            return None
        return output_names[index]

    def _format_value(self, value: Any, input_key: str) -> str:
        """Formats a Python value into its .shiro string representation."""
        if isinstance(value, str):
            # Use json.dumps for proper string escaping (handles quotes, backslashes etc.)
            return json.dumps(value)
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list) and len(value) == 2:
            # This represents a connection [source_node_id, source_output_slot_index]
            source_node_id, source_slot_index = map(str, value) # Ensure IDs are strings

            if source_node_id not in self._node_id_to_var_name:
                # This should not happen if topological sort is correct
                raise ShiroGeneratorError(f"Internal Error: Source node ID '{source_node_id}' for input '{input_key}' not yet processed.")

            source_var_name = self._node_id_to_var_name[source_node_id]
            source_node_data = self._workflow_json.get(source_node_id)
            if not source_node_data:
                 raise ShiroGeneratorError(f"Internal Error: Source node data missing for ID '{source_node_id}'.")

            source_node_type = source_node_data.get("class_type")
            if not source_node_type:
                 raise ShiroGeneratorError(f"Internal Error: Source node type missing for ID '{source_node_id}'.")

            # Check source node's outputs using the schema
            source_node_info = self.nodes_info.get(source_node_type)
            num_outputs = 0
            if source_node_info:
                num_outputs = len(source_node_info.get('output_name', []) or source_node_info.get('output', []))
            else:
                print(f"Warning: Schema not found for source node type '{source_node_type}' (ID: {source_node_id}). Cannot determine optimal connection syntax.", file=sys.stderr)

            if num_outputs <= 1:
                 # If 0 or 1 output, just use the variable name (Shiro convention)
                 # Even if index is technically > 0 (shouldn't happen), this is safer
                 return source_var_name
            else:
                # Multiple outputs, need the specific output name
                output_name = self._get_output_name_by_index(source_node_type, int(source_slot_index))
                if output_name:
                    # Check if output name is simple enough (like a valid identifier)
                    if re.fullmatch(r'[a-zA-Z_]\w*', output_name):
                        return f"{source_var_name}.{output_name}"
                    else:
                        # Handle potentially complex output names if needed, though uncommon
                        print(f"Warning: Output name '{output_name}' for {source_node_type} might require special handling if not a simple identifier. Using it directly.", file=sys.stderr)
                        return f"{source_var_name}.{output_name}" # Assume it works
                else:
                    # Fallback if name lookup failed
                    print(f"Warning: Could not determine output name for index {source_slot_index} of node type '{source_node_type}'. Using index-based fallback (may not be standard .shiro).", file=sys.stderr)
                    # Fallback to a less standard format, though not ideal for .shiro
                    # return f"{source_var_name}[{source_slot_index}]" # Or raise error? Let's stick to dot notation if possible.
                    # A better fallback might be to just guess based on common patterns or raise clearly
                    # Let's try a default name based on index if name is missing
                    return f"{source_var_name}.output_{source_slot_index}" # Default guess

        # Handle other types if necessary (e.g., None, complex objects)
        # For now, assume basic types or connections
        raise ShiroGeneratorError(f"Unsupported value type '{type(value).__name__}' for input '{input_key}': {value!r}")


    def generate(self, workflow_json: Dict[str, Any]) -> str:
        """Generates the .shiro code from the loaded JSON workflow."""
        self._workflow_json = workflow_json
        self._node_id_to_var_name = {}
        self._generated_vars = set()
        self._var_name_counters = {}
        self._shiro_lines = []
        self._processed_nodes = set()

        # --- Topological Sort Logic ---
        nodes_to_process = set(self._workflow_json.keys())
        processing_order: List[str] = []

        while nodes_to_process:
            processed_in_pass = set()
            nodes_ready_this_pass = set()

            for node_id in list(nodes_to_process): # Iterate over a copy
                node_data = self._workflow_json[node_id]
                dependencies_met = True
                inputs = node_data.get("inputs", {})

                for input_value in inputs.values():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        source_node_id = str(input_value[0])
                        # Check if the dependency is in the original workflow
                        # and if it hasn't been processed yet (i.e., added to processing_order)
                        if source_node_id in self._workflow_json and source_node_id not in self._processed_nodes:
                             dependencies_met = False
                             break # Stop checking inputs for this node

                if dependencies_met:
                    nodes_ready_this_pass.add(node_id)

            if not nodes_ready_this_pass:
                 # If no nodes were processed and there are still nodes left, there's a cycle or broken link
                 remaining_ids = ", ".join(nodes_to_process)
                 raise ShiroGeneratorError(f"Could not determine processing order. Cyclic dependency or broken link suspected. Remaining nodes: {remaining_ids}")

            # Process nodes ready in this pass (add to order, mark as processed for next pass)
            # Sort ready nodes numerically for deterministic output (optional but nice)
            sorted_ready_nodes = sorted(list(nodes_ready_this_pass), key=lambda x: int(x) if x.isdigit() else float('inf'))

            for node_id in sorted_ready_nodes:
                processing_order.append(node_id)
                self._processed_nodes.add(node_id) # Mark as processed *logically*
                nodes_to_process.remove(node_id)  # Remove from the set of nodes needing processing

        # --- Generate .shiro lines based on the determined order ---
        for node_id in processing_order:
            node_data = self._workflow_json[node_id]
            class_type = node_data.get("class_type")
            if not class_type:
                print(f"Warning: Node ID '{node_id}' is missing 'class_type'. Skipping.", file=sys.stderr)
                continue

            if class_type not in self.node_class_types:
                print(f"Warning: Node type '{class_type}' (ID: {node_id}) not found in fetched schema. Generating anyway.", file=sys.stderr)

            # Generate variable name for this node
            var_name = self._get_unique_variable_name(node_id, node_data)
            self._node_id_to_var_name[node_id] = var_name

            # Format arguments
            args_list = []
            inputs = node_data.get("inputs", {})
            # Sort inputs alphabetically by key for consistent output
            sorted_input_keys = sorted(inputs.keys())

            for key in sorted_input_keys:
                value = inputs[key]
                try:
                    formatted_value = self._format_value(value, key)
                    args_list.append(f"{key} = {formatted_value}")
                except ShiroGeneratorError as e:
                    # Add context to the error
                    raise ShiroGeneratorError(f"Error processing input '{key}' for node {var_name} (ID: {node_id}, Type: {class_type}): {e}") from e
                except Exception as e:
                     raise ShiroGeneratorError(f"Unexpected error formatting input '{key}' for node {var_name} (ID: {node_id}, Type: {class_type}): {e}") from e


            # Assemble the .shiro line
            args_str = ", ".join(args_list)
            # Basic multi-line formatting for readability if args are long
            if len(args_str) > 100 and len(args_list) > 1: # Arbitrary threshold
                args_str = ",\n    ".join(args_list)
                shiro_line = f"{var_name} = {class_type}(\n    {args_str}\n)"
            else:
                shiro_line = f"{var_name} = {class_type}({args_str})"

            # Add comments from _meta title if present
            if "_meta" in node_data and "title" in node_data["_meta"]:
                 title = node_data["_meta"]["title"].replace('\n', ' ').strip()
                 self._shiro_lines.append(f"# {title} (Node ID: {node_id})")

            self._shiro_lines.append(shiro_line)
            self._shiro_lines.append("") # Add a blank line for separation

        return "\n".join(self._shiro_lines)

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Convert a ComfyUI/ShiroUI JSON workflow to .shiro format.")
    parser.add_argument("json_file", help="Path to the input JSON workflow file.")
    parser.add_argument("--url", default=DEFAULT_SHIROUI_URL, help=f"URL of the running ComfyUI/ShiroUI backend (default: {DEFAULT_SHIROUI_URL}). Needed for output names.")
    parser.add_argument("-o", "--output", help="Optional path to save the output .shiro file.")
    args = parser.parse_args()

    # 1. Fetch Node Info
    nodes_info = fetch_nodes_info(args.url)
    if not nodes_info:
        print("Exiting due to failure fetching node information.", file=sys.stderr)
        sys.exit(1)

    # 2. Load Input JSON
    try:
        print(f"Reading workflow from: {args.json_file}", file=sys.stderr)
        with open(args.json_file, 'r', encoding='utf-8') as f:
            workflow_json = json.load(f)
        if not isinstance(workflow_json, dict):
             raise ValueError("Workflow JSON is not a valid dictionary (object).")
        # Basic validation: check if keys look like node IDs and values are dicts
        for key, value in workflow_json.items():
            if not isinstance(value, dict):
                 raise ValueError(f"Invalid format: Value for key '{key}' is not a dictionary.")
            # Could add more checks here (e.g., presence of 'class_type')

    except FileNotFoundError:
        print(f"Error: Input JSON file not found: '{args.json_file}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{args.json_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except (IOError, ValueError) as e:
        print(f"Error reading or validating file '{args.json_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the JSON file:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


    # 3. Generate .shiro Code
    print("Generating .shiro code...", file=sys.stderr)
    try:
        generator = JsonToShiroGenerator(nodes_info)
        shiro_code = generator.generate(workflow_json)
        print("Generation successful.", file=sys.stderr)
    except ShiroGeneratorError as e:
        print(f"\nGeneration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during generation:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # 4. Output Results
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(shiro_code)
            print(f"Generated .shiro code saved to: {args.output}", file=sys.stderr)
        except IOError as e:
            print(f"Error writing output file '{args.output}': {e}", file=sys.stderr)
            print("\n--- Generated .shiro Code ---")
            print(shiro_code)
            sys.exit(1)
    else:
        print("\n--- Generated .shiro Code ---")
        print(shiro_code)

    print("\n--- Conversion Complete ---", file=sys.stderr)

if __name__ == "__main__":
    main()