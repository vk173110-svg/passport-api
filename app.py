import io
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
from flask import Flask, request, send_file, jsonify, make_response
from flask_cors import CORS
from rembg import remove, new_session

app = Flask(__name__)
# सिर्फ एक जगह सही तरीके से CORS कॉन्फ़िगर करें
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET", "POST", "OPTIONS"])

# ⚡ Silueta Model Lazy Loader
ai_session = None

def get_session():
    global ai_session
    if ai_session is None:
        print("⚡ Loading Silueta 43MB Model...")
        ai_session = new_session("silueta")
        print("✅ Silueta Model Ready!")
    return ai_session

def remove_floating_artifacts(alpha_channel):
    _, binary = cv2.threshold(alpha_channel, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return alpha_channel
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + np.argmax(areas)
    clean_mask = np.zeros_like(binary)
    clean_mask[labels == max_idx] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.GaussianBlur(clean_mask, (3, 3), 0)
    return np.where(clean_mask > 25, alpha_channel, 0).astype(np.uint8)

def studio_ultra_hd_enhancer(bgr_img, alpha_mask):
    smooth = cv2.bilateralFilter(bgr_img, d=5, sigmaColor=25, sigmaSpace=25)
    gaussian = cv2.GaussianBlur(smooth, (0, 0), 1.6)
    sharpened = cv2.addWeighted(smooth, 1.45, gaussian, -0.45, 0)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)
    highlight_mask = l_float > 190
    l_float[highlight_mask] = 190 + (l_float[highlight_mask] - 190) * 0.45
    l = np.clip(l_float, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    retouched = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(retouched, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 65) & (alpha_mask > 100)
    for c in range(3):
        retouched[:, :, c] = np.where(dark_mask, np.clip(retouched[:, :, c] * 0.85, 0, 255), retouched[:, :, c])
    return retouched

def auto_crop_passport_smart_hd(pil_img, aspect_ratio=3.5/4.5):
    bbox = pil_img.getbbox()
    if not bbox:
        return pil_img
    left, top, right, bottom = bbox
    person_w = right - left
    person_h = bottom - top
    img_w, img_h = pil_img.size
    head_margin = int(person_h * 0.10)
    crop_top = max(0, top - head_margin)
    crop_bottom = min(img_h, bottom)
    crop_h = crop_bottom - crop_top
    target_w = int(crop_h * aspect_ratio)
    center_x = left + person_w // 2
    crop_left = center_x - target_w // 2
    crop_right = crop_left + target_w
    if crop_left < 0:
        crop_right += abs(crop_left)
        crop_left = 0
    if crop_right > img_w:
        crop_left = max(0, crop_left - (crop_right - img_w))
        crop_right = img_w
    cropped = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
    final_h = 1543
    final_w = int(final_h * aspect_ratio)
    return cropped.resize((final_w, final_h), Image.LANCZOS)

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "running", "message": "Ultra HD Studio API Live 24x7!"}), 200

@app.route('/api/remove-bg', methods=['POST'])
def process_auto_passport():
    if 'image' not in request.files:
        return jsonify({'error': 'कोई फ़ोटो नहीं मिली'}), 400

    file = request.files['image']
    bg_color = request.form.get('bg_color', 'white')
    crop_mode = request.form.get('crop_mode', 'passport')

    try:
        # 1. सुरक्षा: 512MB RAM क्रैश रोकने के लिए इनपुट इमेज को ऑप्टिमाइज़ करें
        pil_raw = Image.open(file.stream)
        pil_raw.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        
        in_buffer = io.BytesIO()
        pil_raw.save(in_buffer, format="PNG")
        input_bytes = in_buffer.getvalue()

        # 2. AI Background Removal
        output_bytes = remove(input_bytes, session=get_session())
        rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        rgba_np = np.array(rgba)

        # 3. Artifact cleanup & Enhancement
        clean_alpha = remove_floating_artifacts(rgba_np[:, :, 3])
        rgba_np[:, :, 3] = clean_alpha

        bgr = cv2.cvtColor(rgba_np[:, :, :3], cv2.COLOR_RGB2BGR)
        hd_bgr = studio_ultra_hd_enhancer(bgr, clean_alpha)
        rgba_np[:, :, :3] = cv2.cvtColor(hd_bgr, cv2.COLOR_BGR2RGB)

        cleaned_pil = Image.fromarray(rgba_np)

        # 4. Aspect Ratio Crop
        ratio_map = {'passport': 3.5/4.5, 'pancard': 2.5/3.5, 'stamp': 2.0/2.5, 'square': 1.0}
        passport_pil = auto_crop_passport_smart_hd(cleaned_pil, ratio_map.get(crop_mode, 3.5/4.5))

        # 5. Background Color Replacement
        bg_hex = {'white': '#ffffff', 'blue': '#2563eb', 'red': '#dc2626'}.get(bg_color, '#ffffff')
        final_canvas = Image.new("RGBA", passport_pil.size, bg_hex)

        if bg_hex == '#ffffff':
            # FIX: अल्फा मास्क को सही से निकालें ताकि 'bad transparency mask' एरर न आए
            alpha_mask = passport_pil.split()[-1]
            shadow_mask = alpha_mask.filter(ImageFilter.GaussianBlur(3))
            final_canvas.paste((215, 215, 215, 170), (0, 1), shadow_mask)

        final_canvas.paste(passport_pil, (0, 0), passport_pil)

        img_io = io.BytesIO()
        final_canvas.convert("RGB").save(img_io, format='JPEG', quality=98, subsampling=0)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
