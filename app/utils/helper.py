
import random
from PIL import Image
import base64
import gc
import io
import os
from typing import Optional, Dict, Tuple
import cv2
from PIL import Image, ImageOps, PngImagePlugin
from loguru import logger
import numpy as np
import requests
import torch


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def save_image_bytes(image_bytes, output_dir, filename):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct the output file path
    output_path = os.path.join(output_dir, filename)

    # Save the image bytes to the output file
    with open(output_path, 'wb') as f:
        f.write(image_bytes)

    print(f"Image saved to: {output_path}")


def numpy_to_bytes(image_numpy: np.ndarray, ext: str) -> bytes:
    data = cv2.imencode(
        f".{ext}",
        image_numpy,
        [int(cv2.IMWRITE_JPEG_QUALITY), 100, int(
            cv2.IMWRITE_PNG_COMPRESSION), 0],
    )[1]
    image_bytes = data.tobytes()
    return image_bytes


def load_img(img_bytes, gray: bool = False, return_info: bool = False):
    alpha_channel = None
    image = Image.open(io.BytesIO(img_bytes))

    if return_info:
        infos = image.info

    try:
        image = ImageOps.exif_transpose(image)
    except:
        pass

    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    if return_info:
        return np_img, alpha_channel, infos
    return np_img, alpha_channel


def download_image(url: str) -> Optional[np.array]:
    response = requests.get(url)
    if response.status_code == 200:
        # Convert the downloaded image bytes to NumPy array
        image = np.array(Image.open(io.BytesIO(response.content)))
        return image
    else:
        return None


def encode_pil_to_base64(image: Image, quality: int, infos: Dict) -> bytes:
    img_bytes = pil_to_bytes(
        image,
        "png",
        quality=quality,
        infos=infos,
    )
    return base64.b64encode(img_bytes)


def pil_to_bytes(pil_img, ext: str, quality: int = 95, infos={}) -> bytes:
    with io.BytesIO() as output:
        kwargs = {k: v for k, v in infos.items() if v is not None}
        if ext == "jpg":
            ext = "jpeg"
        if "png" == ext.lower() and "parameters" in kwargs:
            pnginfo_data = PngImagePlugin.PngInfo()
            pnginfo_data.add_text("parameters", kwargs["parameters"])
            kwargs["pnginfo"] = pnginfo_data

        pil_img.save(output, format=ext, quality=quality, **kwargs)
        image_bytes = output.getvalue()
    return image_bytes


def compress_image(image_np, target_size_mb=1):
    # Convert image to PIL Image
    image_pil = Image.fromarray(image_np)

    # Define compression quality
    quality = 95

    # Initialize compression loop parameters
    min_quality = 1
    max_quality = 100
    step = 5

    # Initialize compressed image size
    compressed_size = target_size_mb * 1024 * 1024

    # Perform binary search for the optimal quality
    while True:
        # Save image to memory buffer with the current compression quality
        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG", quality=quality)

        # Calculate the size of the compressed image
        compressed_image_size = buffer.tell()

        # Adjust quality based on the comparison with the target size
        if compressed_image_size > compressed_size:
            max_quality = quality
            quality -= step
        else:
            min_quality = quality
            quality += step

        # If the difference between min and max quality is less than the step,
        # or if the quality is already at the minimum or maximum, break the loop
        if max_quality - min_quality < step or quality <= min_quality or quality >= max_quality:
            break

    # Convert the PIL Image back to NumPy array
    compressed_image_np = cv2.imdecode(np.frombuffer(
        buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

    return compressed_image_np


def compress_image_png(image_np, target_size_mb=1):
    # Convert image to PIL Image
    image_pil = Image.fromarray(image_np)

    # Define compression level
    compression_level = 6  # Adjust the compression level as needed

    # Initialize compression loop parameters
    min_compression = 0  # Minimum compression level
    max_compression = 9  # Maximum compression level

    # Initialize compressed image size
    compressed_size = target_size_mb * 1024 * 1024

    # Perform binary search for the optimal compression level
    while True:
        # Save image to memory buffer with the current compression level
        buffer = io.BytesIO()
        image_pil.save(buffer, format="PNG", compress_level=compression_level)

        # Calculate the size of the compressed image
        compressed_image_size = buffer.tell()

        # Adjust compression level based on the comparison with the target size
        if compressed_image_size > compressed_size:
            max_compression = compression_level
            compression_level = (min_compression + compression_level) // 2
        else:
            min_compression = compression_level
            compression_level = (max_compression + compression_level) // 2

        # If the difference between min and max compression is less than 1,
        # or if the compression level is already at the minimum or maximum, break the loop
        if max_compression - min_compression < 1 or compression_level <= min_compression or compression_level >= max_compression:
            break

    # Convert the PIL Image back to NumPy array
    compressed_image_np = cv2.imdecode(np.frombuffer(
        buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)

    return compressed_image_np


def decode_base64_to_image(encoding: str, gray: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    if encoding.startswith("data:image/") or encoding.startswith("data:application/octet-stream;base64,"):
        encoding = encoding.split(";")[1].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(encoding)))

    alpha_channel = None
    try:
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        logger.warning(f"Exif transpose failed: {e}")

    infos = image.info
    if gray:
        image = image.convert("L")
        np_img = np.array(image)
    else:
        if image.mode == "RGBA":
            np_img = np.array(image)
            alpha_channel = np_img[:, :, -1]
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)
        else:
            image = image.convert("RGB")
            np_img = np.array(image)

    return np_img, alpha_channel, infos


def concat_alpha_channel(rgb_np_img: np.ndarray, alpha_channel: Optional[np.ndarray]) -> np.ndarray:
    if alpha_channel is not None:
        if alpha_channel.shape[:2] != rgb_np_img.shape[:2]:
            alpha_channel = cv2.resize(alpha_channel, dsize=(
                rgb_np_img.shape[1], rgb_np_img.shape[0]))
        rgb_np_img = np.concatenate(
            (rgb_np_img, alpha_channel[:, :, np.newaxis]), axis=-1)
    return rgb_np_img


def pil_to_bytes(pil_img: Image.Image, ext: str, quality: int = 95, infos: Dict = {}) -> bytes:
    with io.BytesIO() as output:
        kwargs = {k: v for k, v in infos.items() if v is not None}
        if ext == "jpg":
            ext = "jpeg"
        if "png" == ext.lower() and "parameters" in kwargs:
            pnginfo_data = PngImagePlugin.PngInfo()
            pnginfo_data.add_text("parameters", kwargs["parameters"])
            kwargs["pnginfo"] = pnginfo_data

        pil_img.save(output, format=ext, quality=quality, **kwargs)
        image_bytes = output.getvalue()
    return image_bytes


def resize_and_position_watermark(image: Image.Image, watermark_path: str) -> Image.Image:
    # Load the watermark image
    watermark = Image.open(watermark_path).convert("RGBA")

    # Resize the watermark to 10% of the image width while maintaining aspect ratio
    image_width, image_height = image.size
    watermark_width = int(image_width * 0.3)
    aspect_ratio = watermark.height / watermark.width
    watermark_height = int(watermark_width * aspect_ratio)
    watermark = watermark.resize(
        (watermark_width, watermark_height), Image.LANCZOS)

    # Define margins and possible positions
    margin = 10
    positions = [
        (margin, margin),  # Top-left corner
        (image_width - watermark_width - margin, margin),  # Top-right corner
        (margin, image_height - watermark_height - margin),  # Bottom-left corner
        (image_width - watermark_width - margin, image_height -
         watermark_height - margin)  # Bottom-right corner
    ]

    # Choose a random position for the watermark
    position = random.choice(positions)

    # Paste the watermark onto the image
    image.paste(watermark, position, watermark)
    return image


def apply_watermark(image: np.ndarray, margin: int = 20) -> np.ndarray:
    # Open the original image and watermark image
    original_image = Image.fromarray(image)
    watermark = Image.open("app/media/sefar_ai_logo.png").convert("RGBA")

    # Calculate the new size for the watermark (20% of the original image width)
    original_width, original_height = original_image.size
    new_watermark_width = int(original_width * 0.3)
    watermark_aspect_ratio = watermark.size[1] / watermark.size[0]
    new_watermark_height = int(new_watermark_width * watermark_aspect_ratio)

    # Resize the watermark
    watermark = watermark.resize(
        (new_watermark_width, new_watermark_height), Image.Resampling.LANCZOS)

    # Print the resized watermark size for verification
    print(f"Resized watermark size: {watermark.size}")

    # Calculate the position to place the watermark (bottom-right corner with margin)
    position = (original_width - new_watermark_width - margin,
                original_height - new_watermark_height - margin)

    # Add watermark to the original image
    transparent = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
    transparent.paste(original_image, (0, 0))
    transparent.paste(watermark, position, mask=watermark)
    # Remove alpha for saving in jpg format
    watermarked_image = transparent.convert('RGB')

    return np.array(watermarked_image)


def resize_and_reduce_quality(image: np.ndarray, max_side: int = 500, quality: int = 85) -> np.ndarray:
    # Convert the image to PIL format
    pil_image = Image.fromarray(image)

    # Calculate the new size while maintaining the aspect ratio
    original_width, original_height = pil_image.size
    if original_width > original_height:
        new_width = max_side
        new_height = int((max_side / original_width) * original_height)
    else:
        new_height = max_side
        new_width = int((max_side / original_height) * original_width)

    # Resize the image
    resized_image = pil_image.resize(
        (new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized image to a BytesIO object with reduced quality
    output = io.BytesIO()
    resized_image.save(output, format='JPEG', quality=quality)
    output.seek(0)

    # Read the image back from the BytesIO object
    reduced_quality_image = Image.open(output)

    return np.array(reduced_quality_image)
