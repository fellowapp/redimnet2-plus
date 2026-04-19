#!/usr/bin/env python3
"""Download a password-protected Baidu share with a logged-in cookie jar.

Example:
  conda run -n asv python scripts/download_baidu_share.py \
    --cookie-file /tmp/baidu_user.cookies \
    --share-url https://pan.baidu.com/s/1VaHniaQDVht30TZg3RAwCA \
    --password b6fy \
    --target-dir /data/additional/DATASETS/audio/kespeech
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import hmac
import http.cookiejar
import json
import os
import pathlib
import queue
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable


USER_AGENT = "Mozilla/5.0"
DEVICE_ID = "BB91C9B818963851F99A99261A70E37E|VUFQKX5JL"
APP_SIGNATURE_CERT_MD5 = "ae5821440fab5e1a61a025f014bd8972"
APP_VERSION = "11.10.4"
HMAC_DYNAMIC_KEY = "B8ec24caf34ef7227c66767d29ffd3fb"
DEFAULT_APP_ID = "250528"


class RC4:
    def __init__(self, key: bytes) -> None:
        state = list(range(256))
        j = 0
        for i in range(256):
            j = (j + state[i] + key[i % len(key)]) % 256
            state[i], state[j] = state[j], state[i]
        self.state = state

    def crypt(self, data: bytes) -> bytes:
        state = self.state[:]
        i = 0
        j = 0
        out = bytearray()
        for byte in data:
            i = (i + 1) % 256
            j = (j + state[i]) % 256
            state[i], state[j] = state[j], state[i]
            out.append(byte ^ state[(state[i] + state[j]) % 256])
        return bytes(out)


@dataclass
class ShareFile:
    fs_id: str
    path: str
    name: str
    size: int
    md5: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cookie-file", required=True, help="Netscape cookie file")
    parser.add_argument("--share-url", required=True, help="Baidu share URL")
    parser.add_argument("--password", required=True, help="Share extraction code")
    parser.add_argument("--target-dir", required=True, help="Download destination")
    parser.add_argument(
        "--app-id",
        default=DEFAULT_APP_ID,
        help=f"Baidu app id (default: {DEFAULT_APP_ID})",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only download files matching this shell-style pattern. Repeatable.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Skip files matching this shell-style pattern. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered files without downloading them.",
    )
    parser.add_argument(
        "--backend",
        choices=("python", "curl", "wget"),
        default="python",
        help="Download backend to use for file transfers.",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=8,
        help="Per-read chunk size in MiB for Python downloads.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=30,
        help="Progress print interval in seconds.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="Socket timeout in seconds for API and download requests.",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=15,
        help="Base retry delay in seconds between failed download attempts.",
    )
    parser.add_argument(
        "--max-attempts-per-file",
        type=int,
        default=0,
        help="Retries per file. 0 means retry forever.",
    )
    parser.add_argument(
        "--manifest-name",
        default=".baidu_share_manifest.json",
        help="Manifest filename to write into target dir.",
    )
    return parser.parse_args()


def extract_shorturl(share_url: str) -> str:
    match = re.search(r"/s/1([A-Za-z0-9_-]+)", share_url)
    if not match:
        raise ValueError(f"Could not parse short URL from {share_url}")
    return "1" + match.group(1)


def decode_seckey(seckey: str) -> str:
    return seckey.replace("-", "+").replace("~", "=").replace("_", "/")


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes:.1f}B"


def logical_and_allocated_size(path: pathlib.Path) -> tuple[int, int]:
    stat = path.stat()
    allocated = getattr(stat, "st_blocks", 0) * 512
    return stat.st_size, allocated


def is_retryable_request_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 425, 429, 500, 502, 503, 504}
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, socket.gaierror):
            return True
        if isinstance(reason, TimeoutError | socket.timeout):
            return True
        if isinstance(reason, OSError):
            errno = getattr(reason, "errno", None)
            return errno in {-3, -2, 101, 103, 104, 110, 111, 113}
        if isinstance(reason, str):
            lowered = reason.lower()
            return "temporary failure in name resolution" in lowered or "timed out" in lowered
    return isinstance(exc, TimeoutError | socket.timeout)


class BaiduShareClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        cookie_path = pathlib.Path(args.cookie_file)
        if not cookie_path.exists():
            raise FileNotFoundError(f"Cookie file not found: {cookie_path}")

        self.cookie_jar = http.cookiejar.MozillaCookieJar(str(cookie_path))
        self.cookie_jar.load(ignore_discard=True, ignore_expires=True)
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookie_jar)
        )

        self.shorturl = extract_shorturl(args.share_url)
        self.shareid = ""
        self.share_uk = ""
        self.seckey = ""
        self.root_path = ""
        self.uid = ""
        self.true_sk = ""
        self.bduss_sha1 = ""

    def request_text(
        self,
        url: str,
        *,
        data: str | bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> str:
        request_headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
        if headers:
            request_headers.update(headers)
        if isinstance(data, str):
            data = data.encode()
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            request = urllib.request.Request(url, data=data, headers=request_headers)
            try:
                with self.opener.open(
                    request, timeout=timeout or self.args.request_timeout
                ) as resp:
                    return resp.read().decode("utf-8", errors="replace")
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if attempt == max_attempts or not is_retryable_request_error(exc):
                    raise
                sleep_for = min(120, 5 * attempt)
                print(
                    f"request attempt {attempt} failed for {url}: {exc}; retrying in {sleep_for}s",
                    flush=True,
                )
                time.sleep(sleep_for)
        raise RuntimeError(f"exhausted request retries for {url}")

    def request_json(
        self,
        url: str,
        *,
        data: str | bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> dict:
        return json.loads(
            self.request_text(url, data=data, headers=headers, timeout=timeout)
        )

    def initialize_share(self) -> None:
        form = urllib.parse.urlencode(
            {"pwd": self.args.password, "root": "1", "shorturl": self.shorturl}
        )
        response = self.request_json(
            "https://pan.baidu.com/share/wxlist?channel=weixin&version=2.2.2&clienttype=25&web=1",
            data=form,
            headers={
                "Referer": self.args.share_url,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        if response.get("errno") != 0:
            raise RuntimeError(f"wxlist failed: {response}")

        data = response["data"]
        if not data.get("list"):
            raise RuntimeError(f"Share is empty or inaccessible: {response}")

        self.shareid = str(data["shareid"])
        self.share_uk = str(data["uk"])
        self.seckey = decode_seckey(data["seckey"])
        self.root_path = data["list"][0]["path"]
        print(
            f"share initialized: shareid={self.shareid} uk={self.share_uk} root={self.root_path}",
            flush=True,
        )

    def initialize_account(self) -> None:
        response = self.request_json("https://tieba.baidu.com/mo/q/sync")
        self.uid = str(response["data"]["user_id"])

        sk_response = self.request_json(
            "https://pan.baidu.com/api/report/user?"
            + urllib.parse.urlencode(
                {
                    "action": "ANDROID_ACTIVE_BACKGROUND_UPLOAD_AND_DOWNLOAD",
                    "clienttype": "1",
                    "needrookie": "1",
                    "timestamp": str(int(time.time())),
                    "bind_uid": self.uid,
                    "channel": "android",
                }
            )
        )
        encrypted_sk = sk_response.get("uinfo")
        if sk_response.get("errno") != 0 or not encrypted_sk:
            raise RuntimeError(f"Failed to fetch encrypted SK: {sk_response}")

        bduss = next((cookie.value for cookie in self.cookie_jar if cookie.name == "BDUSS"), "")
        if not bduss:
            raise RuntimeError("BDUSS cookie is required but missing from the cookie file")

        self.true_sk = RC4(self.uid.encode()).crypt(base64.b64decode(encrypted_sk)).decode(
            "latin1"
        )
        self.bduss_sha1 = hashlib.sha1(bduss.encode()).hexdigest()
        print(f"account initialized: uid={self.uid}", flush=True)

    def list_dir(self, dir_path: str, page: int = 1, num: int = 200) -> dict:
        params = urllib.parse.urlencode(
            {
                "uk": self.share_uk,
                "shareid": self.shareid,
                "page": page,
                "num": num,
                "dir": dir_path,
                "order": "other",
                "desc": 1,
                "showempty": 0,
                "web": 1,
                "channel": "chunlei",
                "clienttype": 0,
                "app_id": self.args.app_id,
            }
        )
        response = self.request_json(
            "https://pan.baidu.com/share/list?" + params,
            headers={"Referer": self.args.share_url},
        )
        if response.get("errno") != 0:
            raise RuntimeError(f"share/list failed for {dir_path}: {response}")
        return response

    def walk_files(self) -> list[ShareFile]:
        file_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        file_queue.put(self.root_path)
        files: list[ShareFile] = []

        while not file_queue.empty():
            current_dir = file_queue.get()
            page = 1
            while True:
                response = self.list_dir(current_dir, page=page)
                entries = response.get("list", [])
                for entry in entries:
                    is_dir = bool(int(entry.get("isdir", 0)))
                    if is_dir:
                        file_queue.put(entry["path"])
                        continue
                    files.append(
                        ShareFile(
                            fs_id=str(entry["fs_id"]),
                            path=entry["path"],
                            name=entry["server_filename"],
                            size=int(entry["size"]),
                            md5=entry.get("md5", ""),
                        )
                    )

                has_more = bool(response.get("has_more"))
                if not has_more and len(entries) < 200:
                    break
                page += 1
        files.sort(key=lambda item: item.path)
        return files

    def build_dlink(self, fs_id: str) -> str:
        timestamp_ms = int(time.time() * 1000)
        timestamp = timestamp_ms // 1000
        rand_payload = (
            self.bduss_sha1
            + self.uid
            + self.true_sk
            + str(timestamp_ms)
            + DEVICE_ID
            + APP_VERSION
            + APP_SIGNATURE_CERT_MD5
        )
        rand = hashlib.sha1(rand_payload.encode("latin1")).hexdigest()
        extra = json.dumps({"sekey": self.seckey}, separators=(",", ":"))
        post_body = (
            f'encrypt=0&uk={self.share_uk}&product=share&primaryid={self.shareid}'
            f'&fid_list=["{fs_id}"]&extra={extra}'
        )
        sign = hmac.new(
            HMAC_DYNAMIC_KEY.encode(),
            f"{post_body}_{DEVICE_ID}_{timestamp}".encode(),
            hashlib.sha1,
        ).hexdigest()
        query = urllib.parse.urlencode(
            {
                "sign": sign,
                "timestamp": timestamp,
                "rand": rand,
                "time": timestamp_ms,
                "devuid": DEVICE_ID,
                "channel": "android",
                "clienttype": "1",
                "version": APP_VERSION,
            }
        )
        response = self.request_json(
            "https://pan.baidu.com/api/sharedownload?" + query,
            data=post_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.get("errno") != 0 or not response.get("list"):
            raise RuntimeError(f"sharedownload failed for {fs_id}: {response}")
        return response["list"][0]["dlink"]


class ShareDownloader:
    def __init__(self, client: BaiduShareClient, args: argparse.Namespace) -> None:
        self.client = client
        self.args = args
        self.target_dir = pathlib.Path(args.target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(1, args.chunk_size_mb) * 1024 * 1024

    def relative_target(self, share_path: str) -> pathlib.Path:
        share_posix = pathlib.PurePosixPath(share_path)
        root_posix = pathlib.PurePosixPath(self.client.root_path)
        try:
            relative = share_posix.relative_to(root_posix)
        except ValueError:
            relative = pathlib.PurePosixPath(share_posix.name)
        return pathlib.Path(*relative.parts)

    def matches_filters(self, file: ShareFile) -> bool:
        if self.args.include and not any(
            fnmatch.fnmatch(file.name, pattern) or fnmatch.fnmatch(file.path, pattern)
            for pattern in self.args.include
        ):
            return False
        if self.args.exclude and any(
            fnmatch.fnmatch(file.name, pattern) or fnmatch.fnmatch(file.path, pattern)
            for pattern in self.args.exclude
        ):
            return False
        return True

    def write_manifest(self, files: Iterable[ShareFile]) -> None:
        manifest_path = self.target_dir / self.args.manifest_name
        payload = [
            {
                "name": file.name,
                "path": file.path,
                "relative_target": str(self.relative_target(file.path)),
                "size": file.size,
                "fs_id": file.fs_id,
                "md5": file.md5,
            }
            for file in files
        ]
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote manifest: {manifest_path}", flush=True)

    def clean_sparse_part_if_needed(self, part_path: pathlib.Path) -> None:
        if not part_path.exists():
            return
        logical, allocated = logical_and_allocated_size(part_path)
        if logical == 0 or allocated >= logical:
            return
        backup_path = part_path.with_suffix(part_path.suffix + ".sparse")
        if backup_path.exists():
            backup_path.unlink()
        print(
            "detected sparse partial file, moving it aside before resume: "
            f"{part_path} -> {backup_path}",
            flush=True,
        )
        part_path.rename(backup_path)

    def download_once(self, file: ShareFile, dlink: str, part_path: pathlib.Path) -> None:
        self.clean_sparse_part_if_needed(part_path)
        if self.args.backend == "python":
            self.download_once_python(file, dlink, part_path)
        elif self.args.backend == "curl":
            self.download_once_curl(file, dlink, part_path)
        else:
            self.download_once_wget(file, dlink, part_path)

    def download_once_python(self, file: ShareFile, dlink: str, part_path: pathlib.Path) -> None:
        start_offset = part_path.stat().st_size if part_path.exists() else 0
        headers = {"User-Agent": "pan.baidu.com", "Accept-Encoding": "identity"}
        if start_offset:
            headers["Range"] = f"bytes={start_offset}-"

        request = urllib.request.Request(dlink, headers=headers)
        with urllib.request.urlopen(request, timeout=self.args.request_timeout) as response:
            status = getattr(response, "status", response.getcode())
            if start_offset and status != 206:
                print(
                    f"server ignored resume for {file.name}, restarting from byte 0",
                    flush=True,
                )
                start_offset = 0

            if start_offset and status == 206:
                mode = "ab"
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start_offset}-"):
                    raise RuntimeError(
                        f"unexpected Content-Range for {file.name}: {content_range}"
                    )
            else:
                mode = "wb"

            part_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = start_offset
            start_time = time.time()
            last_report = start_time
            last_report_bytes = downloaded

            with part_path.open(mode) as handle:
                while True:
                    chunk = response.read(self.chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_report >= self.args.progress_interval:
                        delta = downloaded - last_report_bytes
                        rate = delta / max(1e-6, now - last_report)
                        print(
                            f"{file.name}: {format_bytes(downloaded)}/{format_bytes(file.size)} "
                            f"at {format_bytes(rate)}/s",
                            flush=True,
                        )
                        last_report = now
                        last_report_bytes = downloaded

            final_size = part_path.stat().st_size
            if final_size != file.size:
                raise RuntimeError(
                    f"incomplete download for {file.name}: got {final_size}, expected {file.size}"
                )

    def run_backend_command(self, file: ShareFile, part_path: pathlib.Path, command: list[str]) -> None:
        part_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)
        final_size = part_path.stat().st_size
        if final_size != file.size:
            raise RuntimeError(
                f"incomplete download for {file.name}: got {final_size}, expected {file.size}"
            )

    def download_once_curl(self, file: ShareFile, dlink: str, part_path: pathlib.Path) -> None:
        if not shutil.which("curl"):
            raise RuntimeError("curl is not installed")
        command = [
            "curl",
            "-L",
            "--http1.1",
            "-A",
            "pan.baidu.com",
            "-C",
            "-",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            str(self.args.retry_delay),
            "--output",
            str(part_path),
            dlink,
        ]
        self.run_backend_command(file, part_path, command)

    def download_once_wget(self, file: ShareFile, dlink: str, part_path: pathlib.Path) -> None:
        if not shutil.which("wget"):
            raise RuntimeError("wget is not installed")
        command = [
            "wget",
            "--continue",
            "--tries=1",
            "--timeout",
            str(self.args.request_timeout),
            "--user-agent",
            "pan.baidu.com",
            "--output-document",
            str(part_path),
            dlink,
        ]
        self.run_backend_command(file, part_path, command)

    def move_completed_part(self, part_path: pathlib.Path, final_path: pathlib.Path) -> None:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part_path, final_path)

    def should_retry(self, attempt: int) -> bool:
        return self.args.max_attempts_per_file == 0 or attempt < self.args.max_attempts_per_file

    def download_file(self, file: ShareFile) -> None:
        relative_target = self.relative_target(file.path)
        final_path = self.target_dir / relative_target
        part_path = final_path.with_suffix(final_path.suffix + ".part")

        if final_path.exists() and final_path.stat().st_size == file.size:
            print(f"skip complete: {relative_target}", flush=True)
            return

        attempt = 0
        while True:
            attempt += 1
            try:
                dlink = self.client.build_dlink(file.fs_id)
                print(
                    f"download start: {relative_target} (attempt {attempt}, size={format_bytes(file.size)})",
                    flush=True,
                )
                self.download_once(file, dlink, part_path)
                self.move_completed_part(part_path, final_path)
                print(f"download done: {relative_target}", flush=True)
                return
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"attempt {attempt} failed for {relative_target}: {exc}", flush=True)
                if not self.should_retry(attempt):
                    raise
                sleep_for = min(300, self.args.retry_delay * attempt)
                print(f"sleeping {sleep_for}s before retry", flush=True)
                time.sleep(sleep_for)


def main() -> int:
    args = parse_args()
    socket.setdefaulttimeout(args.request_timeout)

    client = BaiduShareClient(args)
    client.initialize_share()
    client.initialize_account()
    all_files = client.walk_files()
    downloader = ShareDownloader(client, args)

    selected_files = [file for file in all_files if downloader.matches_filters(file)]
    downloader.write_manifest(selected_files)

    if args.dry_run:
        for file in selected_files:
            print(f"{file.path} ({format_bytes(file.size)})")
        return 0

    for file in selected_files:
        downloader.download_file(file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
