"""
DeepFake Forensics: Web Application Server.

A web application for facial deepfake detection and explainability,
utilizing EfficientNet-B0 as the sole primary classification model.

Features:
- Home overview and pipeline highlights
- Drag-and-drop evidence upload & quick benchmark samples
- No-face integrity guard to prevent uncropped background shortcut bias
- Comprehensive analysis results & probability distribution
- Face detection, boundary annotation, and face-crop vs full-scene comparison
- 5-Pillar Facial Manipulation Forensics (Face swap seams, ELA, symmetry, distortion, skin texture, consistency)
- Face-focused Grad-CAM with target class switching (Predicted / REAL / FAKE)
- Face-focused Occlusion Sensitivity with anatomical sector contribution analysis
- Forensic session history tracking
"""

from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import uuid

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.gradcam import GradCAM, find_target_layer
from src.explainability.occlusion import OcclusionSensitivity
from src.explainability.pipeline import ExplainabilityPipeline
from src.explainability.visualizer import GradCAMVisualizer
from src.forensics.facial_manipulation import FacialManipulationForensics
from src.image_detection.efficientnet_model import (
    CLASS_NAMES,
    DeepfakeEfficientNetB0,
    get_eval_transform,
)
from src.image_detection.predict_efficientnet import load_efficientnet_model
from src.utils.face_detector import FaceDetector

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "src" / "web" / "templates"),
    static_folder=str(PROJECT_ROOT / "src" / "web" / "static"),
)
app.config["SECRET_KEY"] = "deepfake-forensics-secret-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

SESSIONS_DIR = PROJECT_ROOT / "outputs" / "web_sessions"
HISTORY_FILE = PROJECT_ROOT / "outputs" / "web_history.json"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Lazy singletons
_PIPELINE: Optional[ExplainabilityPipeline] = None
_MODEL: Optional[DeepfakeEfficientNetB0] = None
_DEVICE: Optional[torch.device] = None
_TRANSFORM = get_eval_transform()
_DETECTOR = FaceDetector()
_FORENSICS = FacialManipulationForensics()
_VISUALIZER = GradCAMVisualizer(default_colormap="turbo", default_alpha=0.5)


def get_pipeline() -> ExplainabilityPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = ExplainabilityPipeline(
            device="auto",
            colormap="turbo",
            alpha=0.5,
        )
    return _PIPELINE


def get_model() -> Tuple[DeepfakeEfficientNetB0, torch.device]:
    global _MODEL, _DEVICE
    if _MODEL is None:
        _MODEL, _DEVICE = load_efficientnet_model()
    return _MODEL, _DEVICE


# -----------------------------------------------------------------------------
# History Management
# -----------------------------------------------------------------------------
def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.is_file():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history: List[Dict[str, Any]]) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not save history: {e}")


def add_session_to_history(session_data: Dict[str, Any]) -> None:
    history = load_history()
    # Remove existing entry if reanalyzed
    history = [h for h in history if h.get("id") != session_data.get("id")]
    # Minimal entry
    entry = {
        "id": session_data["id"],
        "filename": session_data.get("filename", "Evidence"),
        "created_at": session_data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M")),
        "predicted_class": session_data.get("predicted_class", "UNKNOWN"),
        "confidence": session_data.get("confidence", 0.0),
        "artifacts": session_data.get("artifacts", {}),
        "forensics": {
            "overall_manipulation_risk": session_data.get("forensics", {}).get("overall_manipulation_risk", "LOW"),
            "overall_anomaly_index": session_data.get("forensics", {}).get("overall_anomaly_index", 0.0),
        } if session_data.get("forensics") else None,
    }
    history.insert(0, entry)
    save_history(history[:50])  # keep last 50


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    session_file = SESSIONS_DIR / session_id / "session.json"
    if not session_file.is_file():
        return None
    try:
        with open(session_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_session(session_data: Dict[str, Any]) -> None:
    sess_dir = SESSIONS_DIR / session_data["id"]
    sess_dir.mkdir(parents=True, exist_ok=True)
    with open(sess_dir / "session.json", "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)
    add_session_to_history(session_data)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    """Page 1: Home page."""
    return render_template("index.html", active_page="home", session_id=None)


@app.route("/upload")
def upload_page():
    """Page 2: Image upload page."""
    error = request.args.get("error")
    allow_bypass = request.args.get("bypass") == "true"
    bypass_path = request.args.get("bypass_path", "")
    return render_template(
        "upload.html",
        active_page="upload",
        session_id=None,
        error_message=error,
        allow_full_image_bypass=allow_bypass,
        bypass_image_path=bypass_path,
    )


@app.route("/analyze", methods=["POST"])
def analyze_action():
    """Action endpoint: processes uploaded file or selected sample."""
    sample_path = request.form.get("sample_path", "").strip()
    force_full_image = request.form.get("force_full_image") == "true"
    eval_mode = request.form.get("eval_mode", "face_crop")
    target_class = request.form.get("target_class", "predicted")
    run_occlusion = request.form.get("run_occlusion") == "true"

    uploaded_file = request.files.get("image_file")
    session_id = str(uuid.uuid4())[:12]
    sess_dir = SESSIONS_DIR / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load source image
    if sample_path:
        src_path = PROJECT_ROOT / sample_path
        if not src_path.is_file():
            return redirect(url_for("upload_page", error=f"Sample file not found: {sample_path}"))
        orig_img = Image.open(src_path).convert("RGB")
        filename = src_path.name
    elif uploaded_file and uploaded_file.filename != "":
        filename = uploaded_file.filename
        orig_img = Image.open(uploaded_file.stream).convert("RGB")
    else:
        return redirect(url_for("upload_page", error="No image file provided. Please upload or select a sample."))

    # Save original image
    orig_path = sess_dir / "original.png"
    orig_img.save(orig_path, format="PNG")

    # 2. Face Detection
    detected_boxes = _DETECTOR.detect_faces(orig_img)

    # Integrity Guard: Check if faces detected
    if len(detected_boxes) == 0 and not force_full_image and eval_mode != "full_image":
        # Halt execution with clear warning
        temp_sample_rel = f"outputs/web_sessions/{session_id}/original.png"
        return redirect(
            url_for(
                "upload_page",
                error="No face detected in the image. DeepFake Forensics halts predictions when no face is found to prevent background shortcut learning bias.",
                bypass="true",
                bypass_path=temp_sample_rel,
            )
        )

    # 3. Determine evaluated image
    if len(detected_boxes) > 0 and not force_full_image and eval_mode != "full_image":
        primary_box = detected_boxes[0]
        face_crop_img, actual_box = _DETECTOR.crop_face(orig_img, primary_box)
        eval_img = face_crop_img
        annotated_orig = _DETECTOR.draw_face_boxes(orig_img, detected_boxes)
    else:
        # Full image fallback/bypass
        primary_box = (0, 0, orig_img.size[0], orig_img.size[1])
        actual_box = primary_box
        eval_img = orig_img
        face_crop_img = None
        annotated_orig = orig_img

    # Save face crop & annotated images
    artifacts: Dict[str, str] = {"original": "original.png"}
    if face_crop_img is not None:
        face_crop_path = sess_dir / "face_crop.png"
        face_crop_img.save(face_crop_path, format="PNG")
        artifacts["face_crop"] = "face_crop.png"

    annotated_path = sess_dir / "original_with_faces.png"
    annotated_orig.save(annotated_path, format="PNG")
    artifacts["original_with_faces"] = "original_with_faces.png"

    # 4. EfficientNet-B0 Inference & Grad-CAM on evaluated image
    model, device = get_model()
    target_layer = model.get_target_layer_for_gradcam()

    t_eval = _TRANSFORM(eval_img).unsqueeze(0).to(device)
    gradcam_engine = GradCAM(model=model, target_layer=target_layer, device=device)
    eff_result = gradcam_engine.generate(
        input_tensor=t_eval,
        target_class=target_class,
        target_size=eval_img.size,
    )

    # Save Grad-CAM artifacts
    eff_hm_rgb = _VISUALIZER.apply_colormap(eff_result.heatmap)
    eff_hm_img = Image.fromarray(eff_hm_rgb)
    eff_overlay = _VISUALIZER.create_overlay(eval_img, eff_result.heatmap)

    eff_hm_img.save(sess_dir / "efficientnet_heatmap.png", format="PNG")
    eff_overlay.save(sess_dir / "efficientnet_overlay.png", format="PNG")
    artifacts["efficientnet_heatmap"] = "efficientnet_heatmap.png"
    artifacts["efficientnet_overlay"] = "efficientnet_overlay.png"

    # Side-by-side [Face Crop | Grad-CAM Overlay]
    eff_side = _VISUALIZER.create_side_by_side(
        image=eval_img,
        heatmap_img=eff_hm_img,
        overlay_img=eff_overlay,
        result=eff_result,
        title="EfficientNet-B0 Face Grad-CAM",
    )
    eff_side.save(sess_dir / "efficientnet_side_by_side.png", format="PNG")
    artifacts["efficientnet_side_by_side"] = "efficientnet_side_by_side.png"

    # 3-Panel overview [Original with Box | Face Crop | Overlay]
    if face_crop_img is not None:
        panel3 = _VISUALIZER.create_face_side_by_side(
            annotated_original=annotated_orig,
            face_crop=face_crop_img,
            overlay=eff_overlay,
            primary_result=eff_result,
        )
        panel3.save(sess_dir / "face_side_by_side.png", format="PNG")
        artifacts["face_side_by_side"] = "face_side_by_side.png"

    # 5. Full-Scene Comparison Inference (to compare with face crop)
    full_pred = None
    if face_crop_img is not None:
        t_full = _TRANSFORM(orig_img).unsqueeze(0).to(device)
        with torch.no_grad():
            f_logits = model(t_full)
            f_probs = F.softmax(f_logits, dim=1)[0]
            f_idx = torch.argmax(f_probs).item()
        full_pred = {
            "predicted_class": CLASS_NAMES[f_idx],
            "confidence": f_probs[f_idx].item() * 100.0,
            "fake_prob": f_probs[0].item() * 100.0,
            "real_prob": f_probs[1].item() * 100.0,
        }

    # 6. Facial Manipulation Forensics (5 modules)
    forensic_res = _FORENSICS.analyze(
        face_image=eval_img,
        full_image=orig_img,
        face_bbox=actual_box if len(detected_boxes) > 0 else None,
        save_dir=sess_dir,
    )
    for art_k, art_path in forensic_res.artifact_files.items():
        artifacts[art_k] = Path(art_path).name

    # 7. Occlusion Sensitivity (Optional or on-demand)
    occlusion_data = None
    if run_occlusion:
        occ_engine = OcclusionSensitivity(model=model, transform=_TRANSFORM, device=device, patch_size=32, stride=16)
        occ_res = occ_engine.generate(eval_img, target_class=target_class, target_size=eval_img.size)
        occ_overlay = _VISUALIZER.create_overlay(eval_img, occ_res.heatmap)
        occ_overlay.save(sess_dir / "occlusion_overlay.png", format="PNG")
        artifacts["occlusion_overlay"] = "occlusion_overlay.png"
        occlusion_data = {
            "base_probability": occ_res.base_probability,
            "max_drop": occ_res.max_drop,
            "target_class": occ_res.target_class,
            "patch_size": occ_res.patch_size,
            "stride": occ_res.stride,
            "region_impacts": occ_res.region_impacts,
            "most_influential_region": occ_res.most_influential_region,
        }

    # 8. Multi-face sub-analysis records
    multi_face_records = []
    if len(detected_boxes) > 1:
        for idx, box in enumerate(detected_boxes):
            f_crop, f_box = _DETECTOR.crop_face(orig_img, box)
            t_f = _TRANSFORM(f_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                l_f = model(t_f)
                p_f = F.softmax(l_f, dim=1)[0]
                idx_f = torch.argmax(p_f).item()
            multi_face_records.append({
                "face_index": idx,
                "face_box": f_box,
                "effnet_result": {
                    "predicted_class": CLASS_NAMES[idx_f],
                    "confidence": p_f[idx_f].item() * 100.0,
                    "fake_prob": p_f[0].item() * 100.0,
                    "real_prob": p_f[1].item() * 100.0,
                },
            })

    # Assemble session document
    session_data = {
        "id": session_id,
        "filename": filename,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_class": eff_result.predicted_class,
        "confidence": eff_result.confidence,
        "fake_prob": eff_result.probabilities.get("FAKE", 0.0),
        "real_prob": eff_result.probabilities.get("REAL", 0.0),
        "target_class": eff_result.target_class,
        "target_score": eff_result.target_score,
        "orig_dimensions": list(orig_img.size),
        "crop_dimensions": list(eval_img.size),
        "faces_detected": len(detected_boxes),
        "face_box": list(actual_box),
        "full_scene_pred": full_pred,
        "forensics": forensic_res.to_dict(),
        "occlusion_data": occlusion_data,
        "faces": multi_face_records,
        "artifacts": artifacts,
    }

    save_session(session_data)
    return redirect(url_for("results_page", session_id=session_id))


@app.route("/results/<session_id>")
def results_page(session_id: str):
    """Page 3: Analysis results page."""
    session_data = load_session(session_id)
    if not session_data:
        return redirect(url_for("upload_page", error="Session not found or expired."))
    return render_template("results.html", active_page="results", session_id=session_id, session=session_data)


@app.route("/face_detection/<session_id>")
def face_detection_page(session_id: str):
    """Page 4: Face detection and cropping page."""
    session_data = load_session(session_id)
    if not session_data:
        return redirect(url_for("upload_page", error="Session not found."))
    return render_template("face_detection.html", active_page="face_detection", session_id=session_id, session=session_data)


@app.route("/forensics/<session_id>")
def forensics_page(session_id: str):
    """Page 5: Facial manipulation analysis page."""
    session_data = load_session(session_id)
    if not session_data:
        return redirect(url_for("upload_page", error="Session not found."))
    return render_template("forensics.html", active_page="forensics", session_id=session_id, session=session_data)


@app.route("/gradcam/<session_id>")
def gradcam_page(session_id: str):
    """Page 6: Face-focused Grad-CAM page."""
    session_data = load_session(session_id)
    if not session_data:
        return redirect(url_for("upload_page", error="Session not found."))
    return render_template("gradcam.html", active_page="gradcam", session_id=session_id, session=session_data)


@app.route("/gradcam/<session_id>/reanalyze", methods=["POST"])
def reanalyze_target_action(session_id: str):
    """Action endpoint: switches Grad-CAM attribution target (predicted, fake, real)."""
    session_data = load_session(session_id)
    if not session_data:
        return redirect(url_for("upload_page"))

    new_target = request.form.get("target", "predicted")
    sess_dir = SESSIONS_DIR / session_id

    # Load evaluated image (face crop or original)
    eval_filename = session_data["artifacts"].get("face_crop", session_data["artifacts"].get("original"))
    eval_img = Image.open(sess_dir / eval_filename).convert("RGB")

    model, device = get_model()
    target_layer = model.get_target_layer_for_gradcam()

    t_eval = _TRANSFORM(eval_img).unsqueeze(0).to(device)
    gradcam_engine = GradCAM(model=model, target_layer=target_layer, device=device)
    new_result = gradcam_engine.generate(
        input_tensor=t_eval,
        target_class=new_target,
        target_size=eval_img.size,
    )

    # Re-save Grad-CAM artifacts
    eff_hm_rgb = _VISUALIZER.apply_colormap(new_result.heatmap)
    eff_hm_img = Image.fromarray(eff_hm_rgb)
    eff_overlay = _VISUALIZER.create_overlay(eval_img, new_result.heatmap)

    eff_hm_img.save(sess_dir / "efficientnet_heatmap.png", format="PNG")
    eff_overlay.save(sess_dir / "efficientnet_overlay.png", format="PNG")

    eff_side = _VISUALIZER.create_side_by_side(
        image=eval_img,
        heatmap_img=eff_hm_img,
        overlay_img=eff_overlay,
        result=new_result,
        title=f"EfficientNet-B0 Face Grad-CAM ({new_target.upper()})",
    )
    eff_side.save(sess_dir / "efficientnet_side_by_side.png", format="PNG")

    # Update session
    session_data["target_class"] = new_result.target_class
    session_data["target_score"] = new_result.target_score
    save_session(session_data)

    return redirect(url_for("gradcam_page", session_id=session_id))


@app.route("/occlusion/<session_id>")
def occlusion_page(session_id: str):
    """Page 7: Face-focused occlusion analysis page."""
    session_data = load_session(session_id)
    if not session_data:
        return redirect(url_for("upload_page", error="Session not found."))
    return render_template("occlusion.html", active_page="occlusion", session_id=session_id, session=session_data)


@app.route("/occlusion/<session_id>/run", methods=["POST"])
def run_occlusion_action(session_id: str):
    """Action endpoint: triggers occlusion sensitivity calculation on the face crop."""
    session_data = load_session(session_id)
    if not session_data:
        return redirect(url_for("upload_page"))

    sess_dir = SESSIONS_DIR / session_id
    eval_filename = session_data["artifacts"].get("face_crop", session_data["artifacts"].get("original"))
    eval_img = Image.open(sess_dir / eval_filename).convert("RGB")

    model, device = get_model()
    occ_engine = OcclusionSensitivity(model=model, transform=_TRANSFORM, device=device, patch_size=32, stride=16)
    occ_res = occ_engine.generate(eval_img, target_class=session_data.get("target_class", "predicted"), target_size=eval_img.size)

    occ_overlay = _VISUALIZER.create_overlay(eval_img, occ_res.heatmap)
    occ_overlay.save(sess_dir / "occlusion_overlay.png", format="PNG")
    session_data["artifacts"]["occlusion_overlay"] = "occlusion_overlay.png"

    session_data["occlusion_data"] = {
        "base_probability": occ_res.base_probability,
        "max_drop": occ_res.max_drop,
        "target_class": occ_res.target_class,
        "patch_size": occ_res.patch_size,
        "stride": occ_res.stride,
        "region_impacts": occ_res.region_impacts,
        "most_influential_region": occ_res.most_influential_region,
    }
    save_session(session_data)

    return redirect(url_for("occlusion_page", session_id=session_id))


@app.route("/history")
def history_page():
    """Page 8: Results history page."""
    history_entries = load_history()
    return render_template("history.html", active_page="history", session_id=None, sessions=history_entries)


@app.route("/history/clear", methods=["POST"])
def clear_history_action():
    """Action endpoint: clears session history index."""
    save_history([])
    return redirect(url_for("history_page"))


@app.route("/session_file/<session_id>/<filename>")
def session_file(session_id: str, filename: str):
    """Secure endpoint: serves forensic artifacts and session images."""
    sess_dir = SESSIONS_DIR / session_id
    if not sess_dir.is_dir():
        abort(404)
    file_path = sess_dir / filename
    if not file_path.is_file():
        abort(404)
    return send_from_directory(str(sess_dir), filename)


def main():
    """Run Flask development server."""
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 70)
    print(" DEEPFAKE FORENSICS: EFFICIENTNET-B0 WEB APPLICATION")
    print("=" * 70)
    print(f" Server URL:   http://127.0.0.1:{port}")
    print(" Primary Model: EfficientNet-B0 (Single Classification Engine)")
    print(" Architecture: 8 Dedicated Forensic Interfaces:")
    print("   1. Home Page                       : /")
    print("   2. Image Upload                    : /upload")
    print("   3. Analysis Results                : /results/<session_id>")
    print("   4. Face Detection & Cropping       : /face_detection/<session_id>")
    print("   5. Facial Manipulation Forensics   : /forensics/<session_id>")
    print("   6. Face-Focused Grad-CAM           : /gradcam/<session_id>")
    print("   7. Face-Focused Occlusion Analysis : /occlusion/<session_id>")
    print("   8. Results History                 : /history")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
