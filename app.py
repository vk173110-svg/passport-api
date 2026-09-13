import io
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from rembg import remove, new_session

app = Flask(__name__)
CORS(app)

print("Loading Human Segmentation AI Model...")
session = new_session("u2net_human_seg")
print("Ultra HD 1200x1543 Studio AI Engine Ready!")

def remove_floating_artifacts(alpha_channel):
    """कान/गर्दन के अनचाहे धब्बे साफ़ करना"""
    _, binary = cv2.threshold(alpha_channel, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

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
    """
    ⚡ ULTRA HD SHARPNESS & TEXTURE ENGINE:
    WhatsApp से आई धुंधली फोटो को DSLR की तरह क्रिस्प, शार्प और 
    आंखों/बालों की डिटेल को शीशे जैसा साफ बनाना
    """
    # 1. स्किन से WhatsApp कम्प्रेशन का धुंधलापन हटाना (Bilateral Smoothing)
    smooth = cv2.bilateralFilter(bgr_img, d=5, sigmaColor=25, sigmaSpace=25)

    # 2. Unsharp Masking (आंखें, पुतलियां, मूंछें और बाल 100% Ultra-Sharp)
    gaussian = cv2.GaussianBlur(smooth, (0, 0), 1.6)
    sharpened = cv2.addWeighted(smooth, 1.45, gaussian, -0.45, 0)

    # 3. LAB Equalization (चेहरे की लाइटिंग बैलेंस करना)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # तेज़ लाइट को दबाना (Anti-Glare)
    l_float = l.astype(np.float32)
    highlight_mask = l_float > 190
    l_float[highlight_mask] = 190 + (l_float[highlight_mask] - 190) * 0.45
    l = np.clip(l_float, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    retouched = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. बालों को गहरा काला (Jet Black) बनाए रखना
    gray = cv2.cvtColor(retouched, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < 65) & (alpha_mask > 100)
    for c in range(3):
        retouched[:, :, c] = np.where(dark_mask, np.clip(retouched[:, :, c] * 0.85, 0, 255), retouched[:, :, c])

    return retouched

def auto_crop_passport_smart_hd(pil_img, aspect_ratio=3.5/4.5):
    """
    ⭐ ULTRA HD 1200 x 1543 RESOLUTION CROPPING:
    छोटे 400px के बजाय बड़े 1200x1543px मास्टर साइज़ में काटना ताकि फोटो कभी न फटे
    """
    bbox = pil_img.getbbox()
    if not bbox:
        return pil_img

    left, top, right, bottom = bbox
    person_w = right - left
    person_h = bottom - top
    img_w, img_h = pil_img.size

    # सिर के ऊपर 10% खाली जगह (Headroom)
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

    # 🌟 मास्टर रेजोल्यूशन: 1200 x 1543 (Full HD Studio Print Quality)
    final_h = 1543
    final_w = int(final_h * aspect_ratio) # 1200 px
    return cropped.resize((final_w, final_h), Image.LANCZOS)

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "running", "message": "Ultra HD 1200x1543 Passport Studio Active!"})

@app.route('/api/remove-bg', methods=['POST'])
def process_auto_passport():
    if 'image' not in request.files:
        return jsonify({'error': 'कोई फ़ोटो नहीं मिली'}), 400

    file = request.files['image']
    bg_color = request.form.get('bg_color', 'blue') # Default Blue / White / Red
    crop_mode = request.form.get('crop_mode', 'passport')

    try:
        input_bytes = file.read()

        # 1. AI Background Removal (Smooth Erode 15)
        output_bytes = remove(
            input_bytes,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=25,
            alpha_matting_erode_size=15
        )

        rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        rgba_np = np.array(rgba)

        # 2. अनचाहे धब्बे खुद साफ़ करना
        clean_alpha = remove_floating_artifacts(rgba_np[:, :, 3])
        rgba_np[:, :, 3] = clean_alpha

        # 3. ⚡ ULTRA HD SHARPENING & GLARE RECOVERY
        bgr = cv2.cvtColor(rgba_np[:, :, :3], cv2.COLOR_RGB2BGR)
        hd_bgr = studio_ultra_hd_enhancer(bgr, clean_alpha)
        rgba_np[:, :, :3] = cv2.cvtColor(hd_bgr, cv2.COLOR_BGR2RGB)

        cleaned_pil = Image.fromarray(rgba_np)

        # 4. ⭐ ULTRA HD 1200x1543 CROP
        if crop_mode == 'passport':
            passport_pil = auto_crop_passport_smart_hd(cleaned_pil, 3.5/4.5)
        elif crop_mode == 'pancard':
            passport_pil = auto_crop_passport_smart_hd(cleaned_pil, 2.5/3.5)
        elif crop_mode == 'stamp':
            passport_pil = auto_crop_passport_smart_hd(cleaned_pil, 2.0/2.5)
        elif crop_mode == 'square':
            passport_pil = auto_crop_passport_smart_hd(cleaned_pil, 1.0)
        else:
            passport_pil = cleaned_pil

        # 5. बैकग्राउंड रंग लगाना (300-600 DPI Crystal Clear)
        bg_hex = {'white': '#ffffff', 'blue': '#2563eb', 'red': '#dc2626'}.get(bg_color, '#2563eb')
        final_canvas = Image.new("RGBA", passport_pil.size, bg_hex)

        if bg_hex == '#ffffff':
            shadow = passport_pil.filter(ImageFilter.GaussianBlur(3))
            final_canvas.paste((215, 215, 215, 170), (0, 1), shadow)

        final_canvas.paste(passport_pil, (0, 0), passport_pil)

        img_io = io.BytesIO()
        # 100% मैक्सिमम क्वालिटी (बिना किसी कम्प्रेशन लॉस के)
        final_canvas.convert("RGB").save(img_io, format='JPEG', quality=100, subsampling=0)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)