"""Image blocks are counted by dimensions, not by base64 length."""

import base64
import struct

from cliffcompaction.engine import estimate_tokens
import pytest

from cliffcompaction.images import (
    DEFAULT_IMAGE_TOKENS,
    MAX_IMAGE_TOKENS,
    dimensions,
    image_payloads,
    tokens_for_payload,
)


def png_bytes(w: int, h: int, payload_bytes: int = 0) -> bytes:
    head = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", w, h)
    return head + b"\x08\x06\x00\x00\x00" + b"\x00" * payload_bytes


def jpeg_bytes(w: int, h: int, exif_bytes: int = 0) -> bytes:
    exif = b"\xff\xe1" + struct.pack(">H", exif_bytes + 2) + b"\x00" * exif_bytes
    sof = b"\xff\xc0" + struct.pack(">H", 17) + b"\x08" + struct.pack(">HH", h, w)
    return b"\xff\xd8" + exif + sof + b"\x00" * 12


def data_uri(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


# --------------------------------------------------------------- headers


def test_reads_png_dimensions():
    assert dimensions(png_bytes(1536, 1024)) == (1536, 1024)


def test_reads_jpeg_dimensions_past_an_exif_block():
    assert dimensions(jpeg_bytes(800, 600, exif_bytes=4000)) == (800, 600)


def test_reads_gif_dimensions():
    raw = b"GIF89a" + struct.pack("<HH", 320, 240) + b"\x00" * 8
    assert dimensions(raw) == (320, 240)


def test_reads_webp_vp8x_dimensions():
    raw = (
        b"RIFF" + b"\x00" * 4 + b"WEBP" + b"VP8X" + b"\x00" * 8
        + (1023).to_bytes(3, "little") + (767).to_bytes(3, "little")
    )
    assert dimensions(raw) == (1024, 768)


def test_unreadable_bytes_give_no_dimensions():
    assert dimensions(b"not an image at all") is None
    assert dimensions(b"") is None


# ----------------------------------------------------------- token cost


@pytest.mark.parametrize(
    "width,height,expected",
    [
        # Every row of the published table, high-resolution tier.
        (200, 200, 64),
        (1000, 1000, 1296),
        (1092, 1092, 1521),
        (1920, 1080, 2691),
        (2000, 1500, 3888),
        (3840, 2160, MAX_IMAGE_TOKENS),  # 10,764 patches, capped
    ],
)
def test_matches_the_published_visual_token_table(width, height, expected):
    assert tokens_for_payload(data_uri(png_bytes(width, height))) == expected


def test_large_image_is_capped_because_providers_downscale():
    huge = tokens_for_payload(data_uri(png_bytes(8000, 8000)))
    assert huge == MAX_IMAGE_TOKENS
    # A 64MP image must not cost 60x a 1MP one.
    assert huge < 4 * tokens_for_payload(data_uri(png_bytes(1024, 1024)))


def test_payload_length_does_not_change_the_estimate():
    """The whole bug: two identical pictures, wildly different file sizes."""
    small = data_uri(png_bytes(1536, 1024, payload_bytes=1_000))
    large = data_uri(png_bytes(1536, 1024, payload_bytes=2_000_000))
    assert len(large) > 100 * len(small)
    assert tokens_for_payload(small) == tokens_for_payload(large)


def test_hosted_url_falls_back_to_the_default():
    assert tokens_for_payload("https://example.com/cat.png") == DEFAULT_IMAGE_TOKENS


def test_undecodable_payload_falls_back_to_the_default():
    assert tokens_for_payload("data:image/png;base64,!!!!") == DEFAULT_IMAGE_TOKENS


# ------------------------------------------------------ dialect shapes


def test_finds_payloads_in_all_three_dialect_shapes():
    body = {
        "messages": [
            {"content": [{"type": "image", "source": {"data": "AAAA"}}]},
            {"content": [{"type": "input_image", "image_url": "data:image/png;base64,BBBB"}]},
            {"content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]},
        ]
    }
    assert list(image_payloads(body)) == [
        "AAAA",
        "data:image/png;base64,BBBB",
        "https://x/y.png",
    ]


def test_documents_keep_the_chars_estimate():
    body = {"content": [{"type": "document", "source": {"data": "x" * 5000}}]}
    assert list(image_payloads(body)) == []


# --------------------------------------------------------- end to end


def test_estimate_is_no_longer_dominated_by_base64():
    """The observed Codex/Blender case: one 1536x1024 screenshot, ~2.7M chars."""
    uri = data_uri(png_bytes(1536, 1024, payload_bytes=2_000_000))
    body = {
        "model": "m",
        "input": [
            {"content": [{"type": "input_text", "text": "hello " * 100}]},
            {"content": [{"type": "input_image", "image_url": uri}]},
        ],
    }
    est = estimate_tokens(body)
    assert len(uri) > 2_600_000  # chars/4 alone would call this ~665k tokens
    assert est < 5_000
    # The text is still counted.
    assert est > tokens_for_payload(uri)


def test_text_only_bodies_are_unchanged():
    body = {"model": "m", "input": [{"content": [{"type": "input_text", "text": "x" * 40_000}]}]}
    import json

    assert estimate_tokens(body) == len(json.dumps(body, ensure_ascii=False)) // 4


def test_many_screenshots_stay_in_a_sane_range():
    """32 images, as in the reproduced session: tens of thousands, not millions.

    chars/4 called this 11M. At ~2k tokens per 1536x1024 screenshot it lands
    near 65k -- under a 200k threshold, so no compaction is warranted, which
    is what the live session should have done and did not.
    """
    uri = data_uri(png_bytes(1536, 1024, payload_bytes=1_000_000))
    body = {"input": [{"content": [{"type": "input_image", "image_url": uri}]} for _ in range(32)]}
    est = estimate_tokens(body)
    assert 32 * 2_000 < est < 32 * MAX_IMAGE_TOKENS
    assert est < 200_000
