import os
import json

print("=" * 50)
print("PWA 配置文件生成工具")
print("=" * 50)

os.makedirs("static", exist_ok=True)
os.makedirs(".streamlit", exist_ok=True)

manifest = {
    "name": "复杂性阑尾炎预测系统",
    "short_name": "阑尾炎AI",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#ffffff",
    "theme_color": "#FF4B4B",
    "icons": [
        {"src": "/app/static/icon-192.png", "sizes": "192x192", "type": "image/png"},
        {"src": "/app/static/icon-512.png", "sizes": "512x512", "type": "image/png"}
    ]
}
with open("static/manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("✅ static/manifest.json")

sw = """const CACHE_NAME = 'appendicitis-ai-v1';
self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(['/'])));
});
self.addEventListener('fetch', e => {
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});
"""
with open("static/service-worker.js", "w", encoding="utf-8") as f:
    f.write(sw)
print("✅ static/service-worker.js")

try:
    from PIL import Image, ImageDraw
    for size, name in [(192, "icon-192.png"), (512, "icon-512.png")]:
        img = Image.new("RGB", (size, size), "#FF4B4B")
        d = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        w, h = size // 6, size // 3
        d.rectangle([cx - w, cy - h, cx + w, cy + h], fill="white")
        d.rectangle([cx - h, cy - w, cx + h, cy + w], fill="white")
        img.save(os.path.join("static", name))
        print(f"✅ static/{name}")
except ImportError:
    print("⚠️ 图标生成需要: pip install Pillow")

config = """[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200

[browser]
gatherUsageStats = false
"""
with open(".streamlit/config.toml", "w") as f:
    f.write(config)
print("✅ .streamlit/config.toml")

reqs = """streamlit>=1.28.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
shap>=0.42.0
matplotlib>=3.6.0
joblib>=1.2.0
openpyxl>=3.1.0
"""
with open("requirements.txt", "w") as f:
    f.write(reqs)
print("✅ requirements.txt")

print("\n🎉 全部完成！")
