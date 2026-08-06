"""Path shim so the MIMIC source data can live locally or in S3.

Only ``build_mimic_sample`` (via ``datasets/mimic_meds.py``) uses this. The
evaluation itself always reads the small local sample file, never S3 — see the
note in ``mimic_config.py``.

A path is treated as S3 when it starts with ``s3://``. Everything else falls
through to plain local filesystem calls, so existing local runs are unaffected.

boto3 is imported lazily and is an optional dependency (``pip install
'.[s3]'``), so nothing here breaks a local-only install or the CPU-only tests.
"""

import gzip
import io
import os
from concurrent.futures import ThreadPoolExecutor

S3_PREFIX = "s3://"

# Retries matter: the discharge gzip is ~1.1 GB read in a single streaming pass,
# long enough that a transient reset is a real possibility.
_MAX_ATTEMPTS = int(os.environ.get("MIMIC_S3_MAX_ATTEMPTS", "10"))

# Parallel GETs for the ~600 small label objects. Sequential round trips would
# make listing alone take ~30-60 s.
_LABEL_WORKERS = int(os.environ.get("MIMIC_S3_WORKERS", "16"))

# Set via MIMIC_AWS_PROFILE or build_mimic_sample --aws-profile.
_PROFILE = os.environ.get("MIMIC_AWS_PROFILE") or None


def set_profile(profile):
    """Pin the AWS profile used for every subsequent call."""
    global _PROFILE, _client
    _PROFILE = profile or None
    _client = None


def is_s3(path):
    return isinstance(path, str) and path.startswith(S3_PREFIX)


def split_s3(uri):
    """``s3://bucket/a/b.csv`` -> ``("bucket", "a/b.csv")``."""
    if not is_s3(uri):
        raise ValueError(f"not an s3 uri: {uri!r}")
    rest = uri[len(S3_PREFIX):]
    bucket, _, key = rest.partition("/")
    if not bucket:
        raise ValueError(f"s3 uri has no bucket: {uri!r}")
    return bucket, key


def join(base, *parts):
    """Join path components for either scheme."""
    if is_s3(base):
        return "/".join([base.rstrip("/")] + [p.strip("/") for p in parts])
    return os.path.join(base, *parts)


_client = None


def client():
    """Cached boto3 S3 client with a retry policy."""
    global _client
    if _client is None:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(
                "reading from s3:// needs boto3 — install it with:\n"
                "    pip install 'boto3'      (or: uv sync --extra s3)"
            ) from e
        cfg = Config(retries={"max_attempts": _MAX_ATTEMPTS, "mode": "standard"})
        session = boto3.Session(profile_name=_PROFILE) if _PROFILE else boto3.Session()
        _client = session.client("s3", config=cfg)
    return _client


def list_csv(directory):
    """Sorted paths of every ``*.csv`` directly under `directory`.

    Works for a local dir or an ``s3://bucket/prefix``. S3 listing is paginated,
    so it is not capped at 1,000 keys.
    """
    if not is_s3(directory):
        import glob
        return sorted(glob.glob(os.path.join(directory, "*.csv")))

    bucket, prefix = split_s3(directory)
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    keys = []
    paginator = client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv") and not key.endswith("/"):
                keys.append(key)
    return sorted(f"{S3_PREFIX}{bucket}/{k}" for k in keys)


def read_text(path, encoding="utf-8"):
    """Whole-object text read. Used for the small label CSVs."""
    if not is_s3(path):
        with open(path, encoding=encoding, newline="") as f:
            return f.read()
    bucket, key = split_s3(path)
    body = client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return body.decode(encoding)


def read_text_many(paths, encoding="utf-8"):
    """``{path: text}`` for many paths, fetching S3 objects in parallel.

    Local paths are read sequentially (already fast). Any failure propagates —
    a missing label file is a real error, not something to paper over.
    """
    paths = list(paths)
    s3_paths = [p for p in paths if is_s3(p)]
    out = {p: read_text(p, encoding) for p in paths if not is_s3(p)}
    if s3_paths:
        client()  # build the client once, before threads race for it
        with ThreadPoolExecutor(max_workers=_LABEL_WORKERS) as pool:
            for path, text in zip(
                s3_paths, pool.map(lambda p: read_text(p, encoding), s3_paths)
            ):
                out[path] = text
    return out


def open_gzip_text(path, encoding="utf-8", newline=""):
    """Streaming text handle over a gzipped CSV, local or in S3.

    The S3 body is decompressed on the fly, so the 1.1 GB discharge file is
    never written to local disk. Returns a context manager.
    """
    if not is_s3(path):
        return gzip.open(path, "rt", encoding=encoding, newline=newline)

    bucket, key = split_s3(path)
    body = client().get_object(Bucket=bucket, Key=key)["Body"]
    # StreamingBody supports read(); GzipFile needs nothing more than that.
    raw = gzip.GzipFile(fileobj=body, mode="rb")
    return io.TextIOWrapper(raw, encoding=encoding, newline=newline)


def exists(path):
    if not is_s3(path):
        return os.path.exists(path)
    bucket, key = split_s3(path)
    try:
        client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:  # noqa: BLE001 - 404/403 both mean "cannot read it"
        return False


def describe(path):
    """Short human-readable location, for run logs."""
    return f"{path} (s3)" if is_s3(path) else f"{path} (local)"
