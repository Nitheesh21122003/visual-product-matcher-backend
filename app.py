from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import sys

# --- Lazy Loading Setup ---
model = None
preprocess = None
products = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    global model, preprocess
    if model is None or preprocess is None:
        print("🔄 Loading CLIP model (lazy)...", file=sys.stderr)
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✅ CLIP model loaded successfully.", file=sys.stderr)
    return model, preprocess

def load_products():
    global products
    if products is None:
        print("📦 Loading products.json (lazy)...", file=sys.stderr)
        with open('products.json', 'r') as f:
            products = json.load(f)
        print(f"✅ Loaded {len(products)} products.", file=sys.stderr)
    return products

# ----------------------------------------------------------------------

def get_image_features(image_path_or_file):
    model, preprocess = load_model()  # Lazy load model
    if isinstance(image_path_or_file, str) and image_path_or_file.startswith('http'):
        response = requests.get(image_path_or_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    elif hasattr(image_path_or_file, "read"):
        image = Image.open(image_path_or_file).convert('RGB')
    else:
        image = Image.open(image_path_or_file).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features

def match_products(uploaded_image):
    products = load_products()  # Lazy load products
    upload_features = get_image_features(uploaded_image)
    results = []
    for product in products:
        try:
            prod_features = get_image_features(product["image"])
            similarity = (upload_features @ prod_features.T).item()
            results.append({
                "id": product["id"],
                "score": similarity,
                "name": product.get("name", ""),
                "image": product["image"],
                "category": product.get("category", "")
            })
        except Exception as e:
            print(f"Error processing product {product['id']}: {e}", file=sys.stderr)
            continue
    results.sort(key=lambda x: x["score"], reverse=True)
    THRESHOLD = 0.7  # Low for demo
    results = [r for r in results if r['score'] >= THRESHOLD]
    return results[:100]

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/api/match', methods=['POST'], strict_slashes=False)
def api_match():
    try:
        if 'image' in request.files:
            file = request.files['image']
            matches = match_products(file)
            return jsonify(matches)
        else:
            data = request.get_json() or {}
            image_url = data.get("image_url")
            if not image_url:
                return jsonify({"error": "No image file or image_url provided"}), 400
            matches = match_products(image_url)
            return jsonify(matches)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    print("🚀 Flask app running with lazy loading...")
    app.run(debug=True, host="0.0.0.0", port=5000)
