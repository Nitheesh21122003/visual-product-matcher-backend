from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
import sys
import os

# ----------------------------
# Lazy-loaded globals
# ----------------------------
model = None
preprocess = None
products = None
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", file=sys.stderr)

# ----------------------------
# Lazy model loading
# ----------------------------
def load_model():
    global model, preprocess
    if model is None or preprocess is None:
        print("🔄 Loading CLIP model...", file=sys.stderr)
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✅ CLIP model loaded.", file=sys.stderr)
    return model, preprocess

# ----------------------------
# Lazy product loading
# ----------------------------
def load_products():
    global products
    if products is None:
        print("📦 Loading products.json...", file=sys.stderr)
        path = os.path.join(os.path.dirname(__file__), "products.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"products.json not found at {path}")
        with open(path, "r") as f:
            products = json.load(f)
        print(f"✅ Loaded {len(products)} products.", file=sys.stderr)
    return products

# ----------------------------
# Feature extraction helper
# ----------------------------
def get_image_features(image_path_or_file):
    model, preprocess = load_model()
    try:
        if isinstance(image_path_or_file, str) and image_path_or_file.startswith("http"):
            response = requests.get(image_path_or_file, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        elif hasattr(image_path_or_file, "read"):
            image = Image.open(image_path_or_file).convert("RGB")
        else:
            image = Image.open(image_path_or_file).convert("RGB")
    except Exception as e:
        print(f"⚠️ Failed to load image: {e}", file=sys.stderr)
        return torch.zeros(1, 512).to(device)

    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_input)
        features /= features.norm(dim=-1, keepdim=True)
    return features

# ----------------------------
# Matching logic
# ----------------------------
def match_products(uploaded_image):
    products = load_products()
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
            print(f"⚠️ Error processing product {product.get('id', 'unknown')}: {e}", file=sys.stderr)
            continue

    # Sort and threshold
    results.sort(key=lambda x: x["score"], reverse=True)
    THRESHOLD = 0.7
    results = [r for r in results if r["score"] >= THRESHOLD]
    return results[:100]

# ----------------------------
# Flask app initialization
# ----------------------------
app = Flask(__name__)
CORS(app, origins=[
    "https://visual-product-matcher-frontend-nine.vercel.app",
    "https://visual-product-matcher-frontend-beta.vercel.app",
    "http://localhost:3000"
], supports_credentials=True)

# Preload model and products on startup
@app.before_first_request
def initialize():
    load_model()
    load_products()

# ----------------------------
# Health check
# ----------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Backend running successfully"})

# ----------------------------
# API match endpoint
# ----------------------------
@app.route("/api/match", methods=["POST"])
def api_match():
    try:
        if "image" in request.files:
            file = request.files["image"]
            matches = match_products(file)
            return jsonify(matches)

        data = request.get_json() or {}
        image_url = data.get("image_url")
        if not image_url:
            return jsonify({"error": "No image file or image_url provided"}), 400

        matches = match_products(image_url)
        return jsonify(matches)

    except Exception as exc:
        print(f"❌ Error in /api/match: {exc}", file=sys.stderr)
        return jsonify({"error": str(exc)}), 500

# ----------------------------
# Run app locally
# ----------------------------
if __name__ == "__main__":
    print("🚀 Flask backend running...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
