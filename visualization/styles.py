"""Theme styling configurations for VisionGPT architecture visualization.

Defines the ThemeStyle dataclass and registers default theme styles
for light, dark, monochrome, and presentation settings.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ThemeStyle:
    """Style attributes defining a rendering theme for diagrams."""

    name: str
    bg_color: str
    text_primary: str
    text_secondary: str
    border_color: str
    arrow_color: str
    arrow_width: float
    font_family: str
    title_font_size: int
    header_font_size: int
    body_font_size: int

    # Block color code categories
    input_color: str
    encoder_color: str
    fusion_color: str
    decoder_color: str
    output_color: str


# Preset configurations
THEMES: Dict[str, ThemeStyle] = {
    "light": ThemeStyle(
        name="light",
        bg_color="#ffffff",
        text_primary="#1f2937",  # Slate 800
        text_secondary="#4b5563",  # Slate 600
        border_color="#d1d5db",  # Slate 300
        arrow_color="#4b5563",
        arrow_width=1.5,
        font_family="sans-serif",
        title_font_size=16,
        header_font_size=12,
        body_font_size=9,
        input_color="#f3f4f6",  # Gray 100
        encoder_color="#dbeafe",  # Blue 100
        fusion_color="#dcfce7",  # Green 100
        decoder_color="#f3e8ff",  # Purple 100
        output_color="#fee2e2",  # Red 100
    ),
    "dark": ThemeStyle(
        name="dark",
        bg_color="#0f172a",  # Slate 900
        text_primary="#f8fafc",  # Slate 50
        text_secondary="#94a3b8",  # Slate 400
        border_color="#334155",  # Slate 700
        arrow_color="#94a3b8",
        arrow_width=1.5,
        font_family="sans-serif",
        title_font_size=16,
        header_font_size=12,
        body_font_size=9,
        input_color="#1e293b",  # Slate 800
        encoder_color="#1e3a8a",  # Blue 900
        fusion_color="#064e3b",  # Green 900
        decoder_color="#581c87",  # Purple 900
        output_color="#7f1d1d",  # Red 900
    ),
    "monochrome": ThemeStyle(
        name="monochrome",
        bg_color="#ffffff",
        text_primary="#000000",
        text_secondary="#333333",
        border_color="#000000",
        arrow_color="#000000",
        arrow_width=1.2,
        font_family="serif",
        title_font_size=14,
        header_font_size=11,
        body_font_size=8.5,
        input_color="#ffffff",
        encoder_color="#f3f4f6",
        fusion_color="#e5e7eb",
        decoder_color="#e5e7eb",
        output_color="#d1d5db",
    ),
    "presentation": ThemeStyle(
        name="presentation",
        bg_color="#181825",  # Mocha crust
        text_primary="#cdd6f4",  # Mocha text
        text_secondary="#a6adc8",  # Mocha subtext
        border_color="#45475a",
        arrow_color="#f5e0dc",
        arrow_width=2.5,
        font_family="sans-serif",
        title_font_size=20,
        header_font_size=14,
        body_font_size=10,
        input_color="#fab387",  # Peach
        encoder_color="#89b4fa",  # Blue
        fusion_color="#a6e3a1",  # Green
        decoder_color="#cba6f7",  # Mauve
        output_color="#f38ba8",  # Red
    ),
}
