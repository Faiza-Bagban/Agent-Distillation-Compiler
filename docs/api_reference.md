# API Reference -- Agent Distillation Compiler

## Base URL
`http://localhost:8000`

## Endpoints

### GET /health
Returns server status.

**Response:**
```json
{"status": "ok", "model_loaded": true}
```

### POST /generate
Generates Python code for a given problem.

**Request body:**
```json
{
  "problem": "Write a function that reverses a string.",
  "max_new_tokens": 512
}
```

**Response:**
```json
{
  "code": "def reverse_string(s):\n    return s[::-1]",
  "route": "student",
  "latency_seconds": 4.2
}
```

**Fields:**
- `problem` (str, required): Natural language coding problem description.
- `max_new_tokens` (int, optional, default 512): Max tokens to generate.
- `code` (str): Generated Python code.
- `route` (str): Which model answered -- `"student"` or `"teacher"`.
- `latency_seconds` (float): Wall-clock time for the request.

## Error codes
- `400` -- Empty problem string
- `503` -- Model not loaded yet (retry in a few seconds after startup)
