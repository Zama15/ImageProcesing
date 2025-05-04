from PIL import Image, ImageDraw, ImageFont
import hashlib, random, os, sys

def generate_seed(text):
  return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

def draw_random_shapes(draw, size, seed):
  random.seed(seed)
  for _ in range(100):
    shape = random.choice(["circle", "square"])
    x0 = random.randint(0, size - 50)
    y0 = random.randint(0, size - 50)
    x1 = x0 + random.randint(20, 80)
    y1 = y0 + random.randint(20, 80)
    color = tuple(random.randint(0, 255) for _ in range(3))
    if shape == "circle":
      draw.ellipse([x0, y0, x1, y1], fill=color)
    else:
      draw.rectangle([x0, y0, x1, y1], fill=color)

def draw_centered_label(draw, size, text, font_path=None, font_size=36, bg_color=(255,255,255), text_color=(0,0,0)):
  font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
  text_bbox = draw.textbbox((0, 0), text, font=font)
  text_w = text_bbox[2] - text_bbox[0]
  text_h = text_bbox[3] - text_bbox[1]
  padding = 10
  rect_w = text_w + padding * 2
  rect_h = text_h + padding * 2
  rect_x = (size - rect_w) // 2
  rect_y = (size - rect_h) // 2
  # Rounded rectangle
  draw.rounded_rectangle([rect_x, rect_y, rect_x + rect_w, rect_y + rect_h], radius=15, fill=bg_color)
  text_x = rect_x + padding - text_bbox[0]
  text_y = rect_y + padding - text_bbox[1]
  draw.text((text_x, text_y), text, fill=text_color, font=font)

def paste_logo(img, logo_path, scale=0.15, position="bottom_right"):
  try:
    logo = Image.open(logo_path).convert("RGBA")
  except FileNotFoundError:
    print("⚠️ Logo file not found, skipping logo.")
    return

  # Resize
  target_w = int(img.size[0] * scale)
  logo = logo.resize((target_w, target_w), Image.LANCZOS)

  # Position
  margin = 10
  if position == "bottom_right":
    pos = (img.size[0] - logo.size[0] - margin, img.size[1] - logo.size[1] - margin)
  elif position == "top_left":
    pos = (margin, margin)
  elif position == "center":
    pos = ((img.size[0] - logo.size[0]) // 2, (img.size[1] - logo.size[1]) // 2)
  else:
    pos = (0, 0)

  img.paste(logo, pos, mask=logo)

def generate_ar_image(text, label, font_path=None, logo_path=None, output_path=None, size=512):
  seed = generate_seed(text)
  img = Image.new("RGB", (size, size), "white")
  draw = ImageDraw.Draw(img)

  draw_random_shapes(draw, size, seed)
  draw_centered_label(draw, size, label, font_path=font_path, font_size=96, bg_color=(255,255,255), text_color=(0,0,0))

  if logo_path:
    paste_logo(img, logo_path)

  if output_path:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, format="JPEG", quality=90)
    print(f"✅ Saved {output_path}")

# --- CLI execution ---
if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("❌ Usage: python ar_image_generator.py path/to/input.txt [path/to/logo.png]")
    sys.exit(1)

  input_label = sys.argv[1]
  input_path = sys.argv[2]
  logo_path = sys.argv[3] if len(sys.argv) > 3 else None

  with open(input_path, "r", encoding="utf-8") as f:
    content = f.read().strip()

  base = os.path.splitext(os.path.basename(input_path))[0]
  output_dir = os.path.join(os.path.dirname(__file__), "img")
  output_path = os.path.join(output_dir, f"ar-pattern-{base}.jpg")

  generate_ar_image(content, font_path="Futura Bold font.ttf", label=input_label, logo_path=logo_path, output_path=output_path)
