# Model Loader

The Model Loader is a service that provides a deterministic way to load and manage AI models. It exposes a FastAPI for programmatic access and a Typer-based CLI for interactive use.

## Features

- 📜 **Model Registry**: centrally managed in `configs/model_registry.json`.
- 🚀 **FastAPI Service**: for listing, selecting, and verifying models.
- 💻 **CLI**: for interacting with the model loader from the command line.
- 🔒 **Version Locking**: to ensure that the correct version of a model is used.
- 🔍 **Vision Probing**: to verify that a model has vision capabilities.

## API Reference

The Model Loader API is a FastAPI application that exposes the following endpoints:

### `GET /v1/models`

Lists all the models in the registry.

**Response:**

```json
[
  {
    "name": "string",
    "backend": "string",
    "capabilities": {},
    "locked": false,
    "vision_ok": true,
    "display_name": "string"
  }
]
```

### `POST /v1/select`

Selects a model from the registry.

**Request:**

```json
{
  "name": "string",
  "require_capabilities": [
    "string"
  ]
}
```

**Response:**

```json
{
  "endpoint": "string",
  "backend": "string",
  "internal_model_id": "string",
  "capabilities": {}
}
```

### `POST /v1/verify`

Verifies a model in the registry.

**Request:**

```json
{
  "name": "string"
}
```

**Response:**

```json
{
  "status": "ok",
  "aggregate_sha256": "string"
}
```

### `POST /v1/probe/vision`

Probes a model to see if it has vision capabilities.

**Request:**

```json
{
  "name": "string",
  "image_path": "string"
}
```

**Response:**

The response is the result of the vision probe.

### `GET /v1/models/{name}/metrics`

Gets the metrics for a model.

**Response:**

```json
{
  "rolling_samples": 0
}
```

## CLI Reference

The Model Loader CLI is a Typer-based application that provides the following commands:

### `list`

Lists all the models in the registry.

**Usage:**

```bash
imageworks-loader list [OPTIONS]
```

**Options:**

- `--role TEXT`: Filter models advertising the specified functional role.

### `select`

Selects a model from the registry.

**Usage:**

```bash
imageworks-loader select NAME [OPTIONS]
```

**Options:**

- `--require-vision`: Require vision capability.

### `verify`

Verifies a model in the registry.

**Usage:**

```bash
imageworks-loader verify NAME
```

### `lock`

Locks a model in the registry.

**Usage:**

```bash
imageworks-loader lock NAME [OPTIONS]
```

**Options:**

- `--set-expected`: Set expected hash from current artifacts if empty.

### `unlock`

Unlocks a model in the registry.

**Usage:**

```bash
imageworks-loader unlock NAME
```

### `probe-vision`

Probes a model to see if it has vision capabilities.

**Usage:**

```bash
imageworks-loader probe-vision NAME IMAGE
```

### `metrics`

Gets the metrics for a model.

**Usage:**

```bash
imageworks-loader metrics [NAME]
```
