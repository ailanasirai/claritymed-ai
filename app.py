# ClarityMed AI - Gradio app (Hugging Face Space entrypoint).
# Research prototype for clinical decision support. Not a diagnostic system.
import os, json, hashlib
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import joblib

# All inference runs on CPU; no GPU is requested.
device = torch.device("cpu")

# ZeroGPU Spaces require a declared GPU function. This no-op is never called.
# The `spaces` package does not exist on Colab or local machines.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_noop():
        return None
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

CHEST_WEIGHTS = os.path.join(MODEL_DIR, "chest_xray_densenet121_v2_best.pth")
BRAIN_WEIGHTS = os.path.join(MODEL_DIR, "brain_mri_resnet50_best.pth")
CHEST_CONFIG  = os.path.join(MODEL_DIR, "chest_xray_config.json")
TEXT_MODEL    = os.path.join(MODEL_DIR, "symptom2disease_model.joblib")
TEXT_VECTOR   = os.path.join(MODEL_DIR, "symptom2disease_vectorizer.joblib")

# Alphabetical (ImageFolder) order used in training
BRAIN_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def file_tag(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]

# Same preprocessing as training: 224x224, ImageNet normalization
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def build_densenet(num_classes=2):
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier.in_features, 256), nn.ReLU(), nn.Linear(256, num_classes))
    return m

def build_resnet_brain(num_classes=4):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 256), nn.ReLU(), nn.Linear(256, num_classes))
    return m

chest_model_v2 = build_densenet(2)
chest_model_v2.load_state_dict(torch.load(CHEST_WEIGHTS, map_location=device))
chest_model_v2.to(device).eval()

brain_model = build_resnet_brain(4)
brain_model.load_state_dict(torch.load(BRAIN_WEIGHTS, map_location=device))
brain_model.to(device).eval()

with open(CHEST_CONFIG) as f:
    saved_threshold = json.load(f)["threshold"]

text_model = joblib.load(TEXT_MODEL)
vectorizer = joblib.load(TEXT_VECTOR)

MODEL_TAG = f"chest {file_tag(CHEST_WEIGHTS)} | brain {file_tag(BRAIN_WEIGHTS)}"
print("Model build:", MODEL_TAG, "| device:", device)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    def generate(self, input_tensor):
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)
        class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        return heatmap, class_idx, probs[0, class_idx].item()

gradcam = GradCAM(chest_model_v2, chest_model_v2.features.denseblock4.denselayer16.conv2)
gradcam_brain = GradCAM(brain_model, brain_model.layer4[-1].conv3)

import os, re, tempfile, uuid
import gradio as gr
from fpdf import FPDF
from datetime import datetime
from PIL import Image as PILImage
import google.generativeai as genai

# ============ GEMINI SETUP ============
# The API key is never stored in code.
# Colab: add GEMINI_API_KEY under Secrets and enable notebook access.
# Hugging Face Spaces: add GEMINI_API_KEY under Settings -> Variables and secrets.
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
    IN_COLAB = True
except Exception:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    IN_COLAB = False

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)

FALLBACK_TEXT = "A detailed clinical explanation will be provided by your doctor during review."

def get_friendly_explanation(condition, confidence, module_name):
    if gemini_model is None:
        return FALLBACK_TEXT
    try:
        prompt = f"""You are a calm, compassionate medical communication assistant.
A patient just received this AI analysis:
Module: {module_name}
Result: {condition}
Model score: {confidence}

Write a short, warm, 2-3 sentence explanation in simple language. Do NOT diagnose or give
medical advice. Reassure them a doctor will review this. Avoid alarming language."""
        return gemini_model.generate_content(prompt).text.strip()
    except Exception as e:
        print("Gemini error:", e)
        return FALLBACK_TEXT

# ============ DECISION BANDS (chest X-ray) ============
# Probabilities between REVIEW_LOW and the saved threshold are shown as "needs review".
# Tune this on the validation set, not the test set.
REVIEW_LOW = 0.50

# ============ CSS: BLUE + BLACK ============
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=DM+Sans:wght@400;500;700&display=swap');
:root { --ink:#0a0f1a; --blue:#1d6fff; --blue-dark:#1553c7; --blue-soft:#e6efff; --paper:#f3f6fc; --card:#fff;
        --line:#d3dcee; --muted:#59657c; --risk:#d63b2f; --risk-soft:#fde9e7; --warn:#b7791f; --warn-soft:#fbf1dc; }
.gradio-container { background:var(--paper) !important; font-family:'DM Sans',system-ui,sans-serif !important; color:var(--ink); max-width:1100px !important; }

#header { background:var(--ink); color:#fff; border-radius:18px; padding:34px 38px 30px; margin-bottom:20px; position:relative; overflow:hidden; border-bottom:4px solid var(--blue); }
#header h1 { margin:0; font-family:'Instrument Serif',Georgia,serif; font-weight:400; font-size:46px; letter-spacing:-.5px; color:#fff; line-height:1.05; }
#header p { margin:10px 0 0; font-size:16px; color:#b9c7e6; max-width:60ch; }
#header .ecg { display:block; width:100%; height:54px; margin-top:22px; }
#header .ecg path { stroke:#4d94ff; stroke-width:2.2; fill:none; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray:900; stroke-dashoffset:900; animation:ecgdraw 2.4s ease-out forwards; }
@keyframes ecgdraw { to { stroke-dashoffset:0; } }
@media (max-width:700px){ #header h1{font-size:34px;} }
@media (prefers-reduced-motion:reduce){ #header .ecg path{animation:none;stroke-dashoffset:0;} }

.section-card { background:var(--card) !important; border:1px solid var(--line) !important; border-radius:14px !important; padding:24px !important; margin-bottom:20px; box-shadow:none !important; }
.section-card h3 { font-family:'Instrument Serif',Georgia,serif; font-weight:400; font-size:26px; color:var(--ink); margin:0 0 6px; }
.section-hint { color:var(--muted); font-size:14px; margin:0 0 14px; }

.tab-nav button,[role="tab"] { font-size:16px !important; font-weight:700 !important; padding:12px 18px !important; color:var(--muted) !important; }
.tab-nav button.selected,[role="tab"][aria-selected="true"] { color:var(--blue) !important; border-bottom:3px solid var(--blue) !important; }

button.primary { background:var(--blue) !important; border:none !important; border-radius:10px !important; color:#fff !important; font-weight:700 !important; font-size:15px !important; }
button.primary:hover { background:var(--blue-dark) !important; }
button.secondary { background:var(--ink) !important; color:#fff !important; border:none !important; border-radius:10px !important; font-weight:700 !important; }
button:focus-visible,input:focus-visible,textarea:focus-visible { outline:3px solid #7fb0ff !important; outline-offset:2px; }

.result-empty { border:1.5px dashed var(--line); border-radius:12px; padding:28px 20px; color:var(--muted); text-align:center; font-size:15px; }
.result-badge { border-radius:12px; padding:16px 18px; display:flex; align-items:center; gap:12px; border:1.5px solid; }
.result-badge h3 { margin:0; font-family:'Instrument Serif',Georgia,serif; font-weight:400; font-size:30px; line-height:1; }
.result-badge .dot { width:14px; height:14px; border-radius:50%; flex:none; }
.badge-risk { background:var(--risk-soft); border-color:var(--risk); color:var(--risk); } .badge-risk .dot { background:var(--risk); }
.badge-ok { background:var(--blue-soft); border-color:var(--blue); color:var(--blue-dark); } .badge-ok .dot { background:var(--blue); }
.badge-review { background:var(--warn-soft); border-color:var(--warn); color:var(--warn); } .badge-review .dot { background:var(--warn); }

.meter-wrap { margin-top:18px; }
.meter-head { display:flex; justify-content:space-between; align-items:baseline; gap:10px; }
.meter-label { font-weight:700; font-size:15px; }
.meter-value { font-family:'Instrument Serif',Georgia,serif; font-size:34px; line-height:1; }
.meter-track { position:relative; height:20px; border-radius:10px; margin-top:8px; overflow:hidden; border:1px solid var(--line);
               background:linear-gradient(90deg,var(--warn-soft) 0 50%,#edf1f9 50% 80%,var(--blue-soft) 80% 100%); }
.meter-fill { height:100%; border-radius:10px 0 0 10px; transition:width .7s cubic-bezier(.2,.8,.2,1); }
.fill-low { background:var(--warn); } .fill-mid { background:#5c86d6; } .fill-high { background:var(--blue); } .fill-risk { background:var(--risk); }
.meter-marker { position:absolute; top:-2px; bottom:-2px; width:3px; background:var(--ink); }
.meter-scale { display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-top:4px; }
.meter-level { display:inline-block; margin-top:10px; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:700; }
.lvl-low { background:var(--warn-soft); color:var(--warn); } .lvl-mid { background:#e8eef9; color:#3d5a94; } .lvl-high { background:var(--blue-soft); color:var(--blue-dark); }
.meter-note { font-size:12.5px; color:var(--muted); margin:10px 0 0; line-height:1.5; }
.explanation-box { background:var(--paper); border-left:4px solid var(--blue); border-radius:8px; padding:14px 16px; margin-top:16px; font-size:14.5px; line-height:1.6; color:var(--ink); }
.review-note { font-size:12.5px; color:var(--muted); margin-top:12px; }
#footer { text-align:center; color:var(--muted); font-size:13px; margin-top:24px; padding:18px; border-top:1px solid var(--line); line-height:1.7; }
"""

# ============ HTML HELPERS ============
EMPTY_HTML = "<div class='result-empty'>Results will appear here after you run an analysis.</div>"

def validate_patient_info(name, age, phone):
    if not name or name.strip() == "":
        return False, "⚠️ Patient name is required."
    if not age or not str(age).strip().isdigit():
        return False, "⚠️ Please enter a valid numeric age."
    if not phone or len(str(phone).strip()) < 10:
        return False, "⚠️ Please enter a valid contact number (at least 10 digits)."
    return True, ""

def confidence_level(pct):
    if pct >= 80: return "lvl-high", "fill-high", "High model confidence"
    if pct >= 60: return "lvl-mid", "fill-mid", "Moderate model confidence"
    return "lvl-low", "fill-low", "Low confidence: treat as uncertain"

def confidence_bar_html(pct, label="Model confidence", marker=None, show_level=True, risk=False):
    lvl_cls, fill_cls, lvl_text = confidence_level(pct)
    if risk: fill_cls = "fill-risk"
    marker_html = f"<div class='meter-marker' style='left:{marker:.1f}%;' title='Decision threshold'></div>" if marker is not None else ""
    level_html = f"<span class='meter-level {lvl_cls}'>{lvl_text}</span>" if show_level else ""
    note = (f"Decision threshold: {marker:.0f}% (dark line). " if marker is not None else "") + \
           "This is the model's raw score, not a calibrated probability."
    return f"""<div class='meter-wrap'>
      <div class='meter-head'><span class='meter-label'>{label}</span><span class='meter-value'>{pct:.1f}%</span></div>
      <div class='meter-track'><div class='meter-fill {fill_cls}' style='width:{pct:.1f}%;'></div>{marker_html}</div>
      <div class='meter-scale'><span>0%</span><span>50%</span><span>100%</span></div>{level_html}
      <p class='meter-note'>{note}</p></div>"""

def result_badge_html(text, state):  # state: "risk" | "ok" | "review"
    return f"<div class='result-badge badge-{state}'><span class='dot'></span><h3>{text}</h3></div>"

def explanation_html(text):
    return (f"<div class='explanation-box'>💬 {text}</div>"
            "<p class='review-note'>AI-assisted analysis. Requires clinical review.</p>")

def build_overlay(image, heatmap):
    h = cv2.resize(heatmap, (224, 224))
    h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    base = np.array(image.convert("RGB").resize((224, 224)))
    return cv2.addWeighted(base, 0.6, h, 0.4, 0)

def save_overlay(overlay):
    path = os.path.join(tempfile.gettempdir(), f"cm_{uuid.uuid4().hex[:8]}.png")
    PILImage.fromarray(overlay).save(path)
    return path

def add_log(log, entry):
    return (list(log or []) + [entry])[-12:]

# ============ SESSION PDF REPORT (blue + black) ============
BLACK, BLUE, BLUE_SOFT, GREY, RISK, AMBER = (10, 15, 26), (29, 111, 255), (230, 239, 255), (89, 101, 124), (214, 59, 47), (183, 121, 31)

def pdf_safe(text):
    # Helvetica supports Latin-1 only; replace other characters to avoid crashes.
    return str(text).encode("latin-1", "replace").decode("latin-1")

class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(*BLACK); self.rect(0, 0, 210, 30, "F")
        self.set_fill_color(*BLUE);  self.rect(0, 30, 210, 2, "F")
        self.set_text_color(255, 255, 255); self.set_font("Helvetica", "B", 20)
        self.set_xy(12, 8);  self.cell(0, 9, "ClarityMed AI")
        self.set_font("Helvetica", "", 9.5); self.set_text_color(170, 195, 245)
        self.set_xy(12, 18); self.cell(0, 6, "Clinical Decision Support Report")
        self.set_y(38)
    def footer(self):
        self.set_y(-14); self.set_font("Helvetica", "", 8); self.set_text_color(*GREY)
        self.cell(0, 6, f"ClarityMed AI | AI-assisted decision support, not a diagnosis | Page {self.page_no()}", align="C")

def make_full_report(name, age, phone, log):
    pdf = ReportPDF(); pdf.set_auto_page_break(True, 18); pdf.add_page()
    report_id = uuid.uuid4().hex[:8].upper()
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(*GREY)
    pdf.cell(0, 6, pdf_safe(f"Report ID: CM-{report_id}   |   Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}"), ln=True)
    pdf.ln(2)
    # Patient card
    y = pdf.get_y(); pdf.set_fill_color(*BLUE_SOFT); pdf.rect(10, y, 190, 26, "F")
    pdf.set_fill_color(*BLUE); pdf.rect(10, y, 2, 26, "F")
    pdf.set_xy(16, y + 3); pdf.set_font("Helvetica", "B", 12); pdf.set_text_color(*BLACK); pdf.cell(0, 6, "Patient information", ln=True)
    pdf.set_font("Helvetica", "", 10.5); pdf.set_x(16)
    pdf.cell(80, 7, pdf_safe(f"Name: {name}")); pdf.cell(40, 7, pdf_safe(f"Age: {age}")); pdf.cell(0, 7, pdf_safe(f"Contact: {phone}"), ln=True)
    pdf.set_y(y + 32)
    # Summary table
    pdf.set_font("Helvetica", "B", 13); pdf.set_text_color(*BLACK); pdf.cell(0, 8, "Summary of analyses explored", ln=True)
    pdf.set_fill_color(*BLACK); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 9.5)
    for w, t in [(38, "Module"), (42, "Result"), (48, "Model score"), (62, "Status")]:
        pdf.cell(w, 8, "  " + t, fill=True)
    pdf.ln(8); pdf.set_font("Helvetica", "", 9.5)
    for i, e in enumerate(log):
        pdf.set_fill_color(*(BLUE_SOFT if i % 2 == 0 else (255, 255, 255))); pdf.set_text_color(*BLACK)
        pdf.cell(38, 8, pdf_safe("  " + e["module"]), fill=True); pdf.cell(42, 8, pdf_safe("  " + e["result"][:24]), fill=True)
        pdf.cell(48, 8, pdf_safe("  " + e["score"]), fill=True)
        col = RISK if e["state"] == "risk" else (AMBER if e["state"] == "review" else BLUE)
        pdf.set_text_color(*col); pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(62, 8, pdf_safe("  " + e["status"]), fill=True); pdf.set_font("Helvetica", "", 9.5); pdf.ln(8)
    pdf.ln(4)
    # Detail section per module
    for e in log:
        if pdf.get_y() > 200: pdf.add_page()
        pdf.set_font("Helvetica", "B", 12); pdf.set_text_color(*BLUE); pdf.cell(0, 8, pdf_safe(e["module"]), ln=True)
        pdf.set_draw_color(*BLUE); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(3)
        y0 = pdf.get_y(); text_x = 10
        if e.get("image") and os.path.exists(e["image"]):
            pdf.image(e["image"], x=10, y=y0, w=52); text_x = 68
        pdf.set_xy(text_x, y0); pdf.set_text_color(*BLACK); pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, pdf_safe(f"Result: {e['result']}"), ln=True)
        pdf.set_x(text_x); pdf.set_font("Helvetica", "", 10); pdf.cell(0, 6, pdf_safe(f"Model score: {e['score']}   ({e['status']})"), ln=True)
        if e.get("detail"):
            pdf.set_x(text_x); pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(*GREY)
            pdf.multi_cell(190 - (text_x - 10), 5, pdf_safe(e["detail"]))
        pdf.set_x(text_x); pdf.set_font("Helvetica", "", 10); pdf.set_text_color(*BLACK)
        pdf.multi_cell(190 - (text_x - 10), 5.5, pdf_safe(e["explanation"]))
        pdf.set_y(max(pdf.get_y(), y0 + (40 if text_x > 10 else 0)) + 6)
    # Next steps and disclaimer
    if pdf.get_y() > 235: pdf.add_page()
    any_flag = any(e["state"] in ("risk", "review") for e in log)
    pdf.set_font("Helvetica", "B", 12); pdf.set_text_color(*BLACK); pdf.cell(0, 8, "Recommended next steps", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, "One or more results need clinical attention. Please consult a licensed physician and share this report. "
                           "Seek emergency care if you have severe symptoms." if any_flag else
                           "No high-risk indicators were flagged by the models. Routine follow-up with your physician is still recommended.")
    pdf.ln(5); pdf.set_font("Helvetica", "I", 8.5); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 4.5, "Disclaimer: ClarityMed is a research prototype for decision support. Model scores are raw outputs and are not "
                           "calibrated probabilities. This report does not replace professional medical diagnosis.")
    out = os.path.join(tempfile.gettempdir(), f"ClarityMed_Report_{re.sub(r'[^A-Za-z0-9_-]', '_', name.strip())}_{report_id}.pdf")
    pdf.output(out)
    return out

def generate_report(name, age, phone, log):
    valid, msg = validate_patient_info(name, age, phone)
    if not valid:
        return None, msg
    if not log:
        return None, "⚠️ Run at least one analysis first, then generate the report."
    return make_full_report(name, age, phone, log), "✅ Report ready. Download it below."

# ============ INFERENCE ============
def chest_xray_interface(image, name, age, phone, log):
    valid, msg = validate_patient_info(name, age, phone)
    if not valid: return None, msg, log
    if image is None: return None, "⚠️ Please upload a chest X-ray image.", log
    x = val_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(chest_model_v2(x), dim=1)
    p = probs[0, 1].item()
    if p >= saved_threshold:   label, state, status = "PNEUMONIA", "risk", "Flagged"
    elif p >= REVIEW_LOW:      label, state, status = "NEEDS REVIEW", "review", "Uncertain: manual review"
    else:                      label, state, status = "NORMAL", "ok", "Not flagged by model"
    heatmap, _, _ = gradcam.generate(x)
    overlay = build_overlay(image, heatmap)
    expl = get_friendly_explanation(label, f"{p*100:.1f}% pneumonia probability", "Chest X-ray")
    html = (result_badge_html(label, state) +
            confidence_bar_html(p * 100, "Pneumonia probability", marker=saved_threshold * 100, show_level=False, risk=(state == "risk")) +
            explanation_html(expl))
    entry = dict(module="Chest X-ray", result=label, score=f"{p*100:.1f}% pneumonia prob.", state=state, status=status,
                 image=save_overlay(overlay), explanation=expl,
                 detail=f"Grad-CAM overlay shown. Decision threshold {saved_threshold*100:.0f}%; review band {REVIEW_LOW*100:.0f}-{saved_threshold*100:.0f}%.")
    return overlay, html, add_log(log, entry)

def brain_mri_interface(image, name, age, phone, log):
    valid, msg = validate_patient_info(name, age, phone)
    if not valid: return None, msg, log
    if image is None: return None, "⚠️ Please upload a brain MRI image.", log
    x = val_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    heatmap, idx, conf = gradcam_brain.generate(x)
    label = BRAIN_CLASSES[idx]
    overlay = build_overlay(image, heatmap)
    low = conf < 0.60
    state = "review" if low else ("risk" if label != "notumor" else "ok")
    status = "Low confidence: manual review" if low else ("Flagged" if label != "notumor" else "Not flagged by model")
    expl = get_friendly_explanation(label, f"{conf*100:.1f}%", "Brain MRI")
    html = (result_badge_html(label.upper(), state) + confidence_bar_html(conf * 100) + explanation_html(expl))
    entry = dict(module="Brain MRI", result=label.upper(), score=f"{conf*100:.1f}%", state=state, status=status,
                 image=save_overlay(overlay), explanation=expl, detail="Grad-CAM overlay shown (inspection aid, not proof of medical reasoning).")
    return overlay, html, add_log(log, entry)

def symptom_checker_interface(symptom_text, name, age, phone, log):
    valid, msg = validate_patient_info(name, age, phone)
    if not valid: return msg, log
    if not symptom_text or symptom_text.strip() == "": return "⚠️ Please describe your symptoms.", log
    vec = vectorizer.transform([symptom_text])
    pred = text_model.predict(vec)[0]
    conf = max(text_model.predict_proba(vec)[0]) * 100
    low = conf < 60
    state, status = ("review", "Low confidence: see a doctor") if low else ("ok", "Screening result")
    expl = get_friendly_explanation(pred, f"{conf:.1f}%", "Symptom Checker")
    html = (result_badge_html(pred, state) + confidence_bar_html(conf) + explanation_html(expl))
    entry = dict(module="Symptom Checker", result=str(pred), score=f"{conf:.1f}%", state=state, status=status,
                 image=None, explanation=expl, detail=f"Patient described: {symptom_text.strip()[:160]}")
    return html, add_log(log, entry)

# ============ UI ============
HEADER_HTML = """<div id='header'>
  <h1>ClarityMed AI</h1>
  <p>Clinical decision support that shows how sure the model really is.</p>
  <svg class='ecg' viewBox='0 0 600 70' preserveAspectRatio='none' aria-hidden='true'>
    <path d='M0 40 H120 L135 40 L150 12 L168 62 L184 40 H300 L315 40 L330 20 L345 52 L358 40 H600'/></svg>
</div>"""

with gr.Blocks(title="ClarityMed AI") as demo:
    log_state = gr.State([])
    gr.HTML(HEADER_HTML)

    with gr.Column(elem_classes="section-card"):
        gr.Markdown("### Patient information")
        gr.HTML("<p class='section-hint'>Fill this in once. It is used for every analysis and for the PDF report. Demo only: please use sample data, not real patient information.</p>")
        with gr.Row():
            name_in = gr.Textbox(label="Patient name", placeholder="e.g. Ayesha Khan")
            age_in = gr.Textbox(label="Age", placeholder="e.g. 34")
            phone_in = gr.Textbox(label="Contact number", placeholder="e.g. 03001234567")

    with gr.Column(elem_classes="section-card"):
        with gr.Tabs():
            with gr.Tab("🩺 Symptom checker"):
                with gr.Row():
                    with gr.Column():
                        symptom_input = gr.Textbox(label="Describe your symptoms", lines=5, placeholder="e.g. I have joint pain and a skin rash since last week...")
                        symptom_btn = gr.Button("Check symptoms", variant="primary")
                    with gr.Column():
                        symptom_output = gr.HTML(value=EMPTY_HTML)
                symptom_btn.click(symptom_checker_interface, [symptom_input, name_in, age_in, phone_in, log_state], [symptom_output, log_state])

            with gr.Tab("🫁 Chest X-ray"):
                with gr.Row():
                    with gr.Column():
                        xray_input = gr.Image(type="pil", label="Upload chest X-ray")
                        xray_btn = gr.Button("Analyze X-ray", variant="primary")
                    with gr.Column():
                        xray_img = gr.Image(label="Where the model looked (Grad-CAM)")
                        xray_html = gr.HTML(value=EMPTY_HTML)
                xray_btn.click(chest_xray_interface, [xray_input, name_in, age_in, phone_in, log_state], [xray_img, xray_html, log_state])

            with gr.Tab("🧠 Brain MRI"):
                with gr.Row():
                    with gr.Column():
                        mri_input = gr.Image(type="pil", label="Upload brain MRI")
                        mri_btn = gr.Button("Analyze MRI", variant="primary")
                    with gr.Column():
                        mri_img = gr.Image(label="Where the model looked (Grad-CAM)")
                        mri_html = gr.HTML(value=EMPTY_HTML)
                mri_btn.click(brain_mri_interface, [mri_input, name_in, age_in, phone_in, log_state], [mri_img, mri_html, log_state])

    with gr.Column(elem_classes="section-card"):
        gr.Markdown("### Session report")
        gr.HTML("<p class='section-hint'>One PDF with the patient details and every analysis explored in this session.</p>")
        report_btn = gr.Button("Generate full report (PDF)", variant="secondary")
        report_msg = gr.Markdown()
        report_file = gr.File(label="📄 Download report")
        report_btn.click(generate_report, [name_in, age_in, phone_in, log_state], [report_file, report_msg])

    gr.HTML(f"""<div id='footer'>AI-assisted decision support. Not a replacement for clinical diagnosis.<br>
        © ClarityMed AI &nbsp;|&nbsp; Developed by Aila Nasir<br><span style='font-size:11px'>Model build: {MODEL_TAG}</span></div>""")

clean_theme = gr.themes.Base(primary_hue="blue", neutral_hue="slate").set(
    body_background_fill="#f3f6fc",
    background_fill_primary="#ffffff", background_fill_secondary="#ffffff",
    block_background_fill="#ffffff", block_border_width="0px", block_shadow="none",
    block_label_background_fill="transparent", block_label_border_width="0px",
    block_label_text_color="#0a0f1a", block_label_text_weight="700",
    block_title_background_fill="transparent", block_title_text_color="#0a0f1a",
    panel_background_fill="#ffffff", panel_border_width="0px",
    input_background_fill="#ffffff", input_border_color="#d3dcee", input_border_width="1px",
    button_primary_background_fill="#1d6fff", button_primary_background_fill_hover="#1553c7",
    button_primary_text_color="#ffffff",
)
demo.launch(share=IN_COLAB, debug=IN_COLAB, css=custom_css, theme=clean_theme)
