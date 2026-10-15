#!/usr/bin/env python3
"""Standalone calibrated optical MP4 sender (Python 3.12+).

Install: python3 -m pip install opencv-contrib-python==5.0.0.93 numpy==2.5.4 imageio-ffmpeg==0.6.0
Usage: python3 optical_encoder.py INPUT OUTPUT.mp4
Dependencies include the FFmpeg binary via imageio-ffmpeg. No receiver code.
"""
import numpy as np
fec_K = 768
fec_N = 1536
fec_BLOCKS = 10
fec_RAW_BYTES = 960
fec_CODE_BYTES = 1920
fec_ROWS = np.arange(fec_K)
fec_A = np.column_stack((fec_ROWS, (5 * fec_ROWS + 17) % fec_K, (13 * fec_ROWS + 43) % fec_K))
fec_EDGES = np.column_stack((fec_A, fec_K + fec_ROWS, fec_K + fec_ROWS - 1))
fec_EDGES[0, 4] = fec_N
fec_ACTIVE = fec_EDGES != fec_N

def fec_encode(raw):
    """Encode ten independent 96-byte messages into 1920 codeword bytes."""
    if len(raw) != fec_RAW_BYTES:
        raise ValueError('LDPC input must be 960 bytes')
    information = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(fec_BLOCKS, fec_K)
    checks = np.bitwise_xor.reduce(information[:, fec_A], axis=2)
    parity = np.bitwise_xor.accumulate(checks, axis=1)
    return np.packbits(np.concatenate((information, parity), axis=1)).tobytes()
import hashlib
import struct
import zlib
from dataclasses import dataclass
CAPACITY = 1957
HEADER = struct.Struct('>4sB16sIIQH32s')
PAYLOAD = fec_RAW_BYTES - HEADER.size - 4
MAX_FILE_BYTES = 100 * 1024 * 1024

@dataclass(frozen=True)
class Packet:
    """One indexed file chunk; digest also determines the transfer identity."""
    index: int
    count: int
    size: int
    digest: bytes
    data: bytes

def encode(packet):
    """Serialize a chunk with a protected header, payload and zero padding."""
    head = HEADER.pack(b'OTV1', 1, packet.digest[:16], packet.index, packet.count, packet.size, len(packet.data), packet.digest)
    body = head + packet.data
    if len(packet.data) > PAYLOAD:
        raise ValueError('Payload too large')
    raw = body + struct.pack('>I', zlib.crc32(body))
    return fec_encode(raw.ljust(fec_RAW_BYTES, b'\x00')).ljust(CAPACITY, b'\x00')

def file_digest(path):
    """Compute SHA-256 without loading an entire file into memory."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').digest()

def packets(path):
    """Read file chunks sequentially, carrying complete metadata in each."""
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError('Experimental limit: 100 MiB')
    digest = file_digest(path)
    count = max(1, (size + PAYLOAD - 1) // PAYLOAD)
    with path.open('rb') as stream:
        for index in range(count):
            yield Packet(index, count, size, digest, stream.read(PAYLOAD))
import cv2
import numpy as np
WIDTH, HEIGHT = (1920, 1080)
X, Y, BLOCK = (90, 90, 30)
COLS, ROWS = (58, 30)
MARKERS = ((10, 10), (1850, 10), (1850, 1010), (10, 1010))
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def render(raw, marker_base=0):
    """Render exactly one fixed-size packet into an aligned BGR image."""
    if len(raw) != CAPACITY:
        raise ValueError('Invalid packet size')
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    triples = np.pad(bits, (0, ROWS * COLS * 9 - bits.size)).reshape(ROWS, COLS, 3, 3)
    rgb = triples @ np.array([4, 2, 1], dtype=np.uint8)
    rgb = np.rint(rgb * (255.0 / 7)).astype(np.uint8)
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    image[Y:Y + ROWS * BLOCK, X:X + COLS * BLOCK] = rgb[:, :, ::-1].repeat(BLOCK, 0).repeat(BLOCK, 1)
    for identity, (left, top) in enumerate(MARKERS):
        marker = cv2.aruco.generateImageMarker(DICTIONARY, identity + marker_base, 60)
        image[top:top + 60, left:left + 60] = marker[:, :, None]
    return image
import cv2
import numpy as np
SYMBOLS = np.arange(512)
RGB = SYMBOLS[:, None] >> np.array([6, 3, 0]) & 7
PALETTE = np.rint(64 + RGB * (160.0 / 7)).astype(np.uint8)
CAL_SYMBOLS = ((np.arange(ROWS * COLS) * 173 + 37) % 512).reshape(ROWS, COLS)
PILOT_SYMBOLS = np.array([r * 64 + g * 8 + b for r in (0, 2, 5, 7) for g in (0, 2, 5, 7) for b in (0, 2, 5, 7)] + [q * 73 for q in range(8)])
PILOT_X = 168 + np.arange(len(PILOT_SYMBOLS)) * 22
PILOT_Y = (25, 1015)
BIT_LABELS = (SYMBOLS[:, None] >> np.arange(8, -1, -1) & 1).astype(bool)

def _pilots(image):
    """Paint identical reference colors above and below the payload area."""
    strip = PALETTE[PILOT_SYMBOLS, ::-1].repeat(22, axis=0)
    for top in PILOT_Y:
        image[top:top + 40, 168:168 + strip.shape[0]] = strip[None, :, :]
    return image

def render_data(raw):
    """Keep the 58x30 payload grid and add profile-2 marker IDs and pilots."""
    image = render(raw, marker_base=20)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    bits = np.pad(bits, (0, ROWS * COLS * 9 - bits.size)).reshape(ROWS, COLS, 9)
    symbols = bits @ (1 << np.arange(8, -1, -1))
    image[Y:Y + ROWS * BLOCK, X:X + COLS * BLOCK] = PALETTE[symbols, ::-1].repeat(BLOCK, 0).repeat(BLOCK, 1)
    return _pilots(image)

def render_calibration():
    """Render all 512 known symbols at least three times in a mixed pattern."""
    image = render(bytes(CAPACITY), marker_base=10)
    pixels = PALETTE[CAL_SYMBOLS, ::-1].repeat(BLOCK, 0).repeat(BLOCK, 1)
    image[Y:Y + ROWS * BLOCK, X:X + COLS * BLOCK] = pixels
    return _pilots(image)
import os
import tempfile
from pathlib import Path
import cv2
CALIBRATION_HOLD = 30
TAIL_HOLD = 30
REPEAT_CYCLES = 2
CALIBRATION_INTERVAL = 32

def send(source, destination):
    """Write a calibrated optical AVI, publishing only a completed transfer."""
    source, destination = (Path(source), Path(destination))
    if destination.exists():
        raise FileExistsError(destination)
    if destination.suffix.lower() != '.avi':
        raise ValueError('Experimental video output must use .avi')
    iterator = packets(source)
    first = next(iterator)
    iterator.close()
    descriptor, name = tempfile.mkstemp(prefix='.optical-send-', suffix='.avi', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        physical = _write_video(source, temporary)
        if file_digest(source) != first.digest:
            raise ValueError('Input file changed during transmission')
        _publish(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {'logical_frames': first.count, 'physical_frames': physical, 'bytes': first.size, 'sha256': first.digest.hex(), 'profile': 'calibrated-2', 'repeat_cycles': REPEAT_CYCLES, 'duration_seconds': physical / 30}

def _write_video(source, destination):
    """Insert periodic calibration and repeat the complete file twice."""
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'FFV1'), 30, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError('FFV1 writer unavailable')
    calibration = render_calibration()
    physical = 0
    try:
        for _ in range(REPEAT_CYCLES):
            for packet in packets(source):
                if packet.index % CALIBRATION_INTERVAL == 0:
                    _repeat(writer, calibration, CALIBRATION_HOLD)
                    physical += CALIBRATION_HOLD
                _write_packet(writer, packet)
                physical += 4
        _repeat(writer, calibration, TAIL_HOLD)
        physical += TAIL_HOLD
    finally:
        writer.release()
    return physical

def _repeat(writer, image, count):
    """Hold one image for an exact number of physical frames."""
    for _ in range(count):
        writer.write(image)

def _write_packet(writer, packet):
    """Render once and repeat identical images four times."""
    _repeat(writer, render_data(encode(packet)), 4)

def _publish(temporary, destination):
    """Use Windows no-replace rename, or exclusive hard linking on POSIX."""
    if os.name == 'nt':
        os.rename(temporary, destination)
    else:
        os.link(temporary, destination)
import os
from pathlib import Path
import subprocess
import tempfile
import cv2
import imageio_ffmpeg

def describe(path):
    """Read dimensions, fps, codec tag and frame count from a video."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('動画を読み込めません: ' + str(path))
        tag = int(capture.get(cv2.CAP_PROP_FOURCC))
        return {'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 'fps': capture.get(cv2.CAP_PROP_FPS), 'frames': int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 'codec_tag': ''.join((chr(tag >> 8 * i & 255) for i in range(4)))}
    finally:
        capture.release()

def _validate_input(source, destination):
    """Require the project's full-HD 30fps AVI and a fresh MP4 output path."""
    if destination.exists():
        raise FileExistsError('出力ファイルは既に存在します: ' + str(destination))
    if source.suffix.lower() != '.avi':
        raise ValueError('変換元にはAVIファイルを指定してください')
    if destination.suffix.lower() != '.mp4':
        raise ValueError('変換先には.mp4を指定してください')
    info = describe(source)
    if (info['width'], info['height']) != (1920, 1080):
        raise ValueError('変換元は1920×1080の動画に限ります')
    if abs(info['fps'] - 30) > 0.01:
        raise ValueError('変換元は30fpsの動画に限ります')
    if info['frames'] < 1:
        raise ValueError('変換元にフレームがありません')
    return info

def _transcode(source, destination):
    """Run bundled FFmpeg at high quality, with no console or shell invocation."""
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-noautorotate', '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn', '-map_metadata', '-1', '-c:v', 'libx264', '-preset', 'medium', '-crf', '10', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level:v', '4.1', '-fps_mode', 'passthrough', '-movflags', '+faststart', str(destination)]
    flags = 0
    if os.name == 'nt':
        flags = subprocess.CREATE_NO_WINDOW
    with tempfile.TemporaryFile() as error_log:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=error_log, creationflags=flags, check=False)
        if result.returncode:
            error_log.seek(0, os.SEEK_END)
            length = error_log.tell()
            error_log.seek(max(0, length - 8192))
            reason = error_log.read().decode('utf-8', errors='replace')
            raise RuntimeError('MP4変換に失敗しました: ' + reason)

def convert(source, destination):
    """Stage conversion, validate output metadata, then publish a new MP4."""
    source, destination = (Path(source).resolve(), Path(destination).resolve())
    original = _validate_input(source, destination)
    descriptor, name = tempfile.mkstemp(prefix='.optical-mp4-', suffix='.mp4', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        _transcode(source, temporary)
        converted = describe(temporary)
        for key in ('width', 'height', 'frames'):
            if original[key] != converted[key]:
                raise ValueError('変換後の動画属性が一致しません: ' + key)
        if abs(converted['fps'] - 30) > 0.01:
            raise ValueError('変換後のfpsが一致しません')
        _publish(temporary, destination)
        return {'output': str(destination), 'codec': 'H.264', 'pixel_format': 'yuv420p', **converted, 'bytes': destination.stat().st_size, 'message': '変換完了。通信データの一致は「動画からファイルを復元」で確認してください。'}
    finally:
        temporary.unlink(missing_ok=True)

def create_mp4(source, destination):
    """Generate a temporary calibrated AVI, convert to MP4, and remove staging files."""
    destination = Path(destination).resolve()
    if destination.suffix.lower() != '.mp4':
        raise ValueError('保存先には.mp4を指定してください。')
    if destination.exists():
        raise FileExistsError('出力ファイルは既に存在します。別の保存名を指定してください。')
    with tempfile.TemporaryDirectory(prefix='.optical-sender-', dir=destination.parent) as folder:
        intermediate = Path(folder) / 'staging.avi'
        transfer = send(source, intermediate)
        conversion = convert(intermediate, destination)
    return {'message': 'MP4の作成が完了しました。', 'output': str(destination), 'input_bytes': transfer['bytes'], 'input_sha256': transfer['sha256'], 'profile': transfer['profile'], 'duration_seconds': transfer['duration_seconds'], 'frames': transfer['physical_frames'], 'mp4_bytes': conversion['bytes']}
import argparse
import json
import sys

def main():
    """Parse two paths, generate MP4 and report a nonzero exit on failure."""
    parser = argparse.ArgumentParser(description='Calibrated optical file-to-MP4 sender')
    parser.add_argument('input', type=Path, help='Input data file')
    parser.add_argument('output', type=Path, help='New output MP4 path')
    args = parser.parse_args()
    try:
        result = create_mp4(args.input, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
if __name__ == '__main__':
    sys.exit(main())
