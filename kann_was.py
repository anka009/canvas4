# canvas_safe_fixed.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json
import numpy as np

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < r

def dedup_points(points, min_dist=6):
    """Entfernt Punkte, die sich zu nahe sind (keine Duplikate/Geister)."""
    out = []
    for p in points:
        if not any(is_near(p, q, r=min_dist) for q in out):
            out.append(p)
    return out

def get_centers(mask, min_area=50):
    """Robuste findContours-Variante (OpenCV-Version neutral)."""
    m = mask.copy()
    res = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV: entweder (contours, hierarchy) oder (image, contours, hierarchy)
    if len(res) == 2:
        contours, _ = res
    else:
        _, contours, _ = res
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers

def compute_hsv_range(points, hsv_img, buffer_h=8, buffer_s=30, buffer_v=25):
    radius = 5  # fester Radius in Pixeln
    if not points:
        return None

    vals = []
    for (x, y) in points:
        x_min = max(0, x - radius)
        x_max = min(hsv_img.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(hsv_img.shape[0], y + radius + 1)

        region = hsv_img[y_min:y_max, x_min:x_max]
        vals.append(region.reshape(-1, 3))  # alle HSV-Werte in der Region

        vals = np.vstack(vals)
        h = vals[:, 0].astype(int)
        s = vals[:, 1].astype(int)
        v = vals[:, 2].astype(int)

        h_min = max(0, np.min(h) - buffer_h)
        h_max = min(180, np.max(h) + buffer_h)
        s_min = max(0, np.min(s) - buffer_s)
        s_max = min(255, np.max(s) + buffer_s)
        v_min = max(0, np.min(v) - buffer_v)
        v_max = min(255, np.max(v) + buffer_v)

        return (h_min, h_max, s_min, s_max, v_min, v_max)


def apply_hue_wrap(hsv_img, hmin, hmax, smin, smax, vmin, vmax):
    """Maskiert unter Berücksichtigung von Hue-Wrap-around."""
    if hmin <= hmax:
        mask = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img, np.array([0, smin, vmin]), np.array([hmax, smax, vmax]))
        mask_hi = cv2.inRange(hsv_img, np.array([hmin, smin, vmin]), np.array([180, smax, vmax]))
        mask = cv2.bitwise_or(mask_lo, mask_hi)
    return mask

def ensure_odd(k):
    if k % 2 == 1:
        return k
    return k + 1
def save_last_calibration():
    data = {
        "aec_hsv": st.session_state.aec_hsv,
        "hema_hsv": st.session_state.hema_hsv,
        "bg_hsv": st.session_state.bg_hsv
    }
    with open("kalibrierung.json", "w") as f:
        json.dump(data, f)

def load_last_calibration():
    try:
        with open("kalibrierung.json", "r") as f:
            data = json.load(f)
            st.session_state.aec_hsv = data.get("aec_hsv")
            st.session_state.hema_hsv = data.get("hema_hsv")
            st.session_state.bg_hsv = data.get("bg_hsv")
            st.success("✅ Letzte Kalibrierung geladen.")
    except FileNotFoundError:
        st.warning("⚠️ Keine gespeicherte Kalibrierung gefunden.")

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (fixed)", layout="wide")
st.title("🧬 Zellkern-Zähler – stabilisierte Version")

# -------------------- Session State --------------------
default_keys = [
    "aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema",
    "aec_hsv", "hema_hsv", "bg_hsv", "last_file", "disp_width", "last_auto_run"
]
for key in default_keys:
    if key not in st.session_state:
        if "points" in key or "manual" in key:
            st.session_state[key] = []
        elif key == "disp_width":
            st.session_state[key] = 1400
        else:
            st.session_state[key] = None
def save_last_calibration():
    import json
    import numpy as np

    def safe_list(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, list):
            return val
        else:
            return None

    data = {
        "aec_hsv": safe_list(st.session_state.get("aec_hsv")),
        "hema_hsv": safe_list(st.session_state.get("hema_hsv")),
        "bg_hsv": safe_list(st.session_state.get("bg_hsv"))
    }

    with open("kalibrierung.json", "w") as f:
        json.dump(data, f)

def load_last_calibration():
    try:
        with open("kalibrierung.json", "r") as f:
            data = json.load(f)
            st.session_state.aec_hsv = np.array(data.get("aec_hsv")) if data.get("aec_hsv") else None
            st.session_state.hema_hsv = np.array(data.get("hema_hsv")) if data.get("hema_hsv") else None
            st.session_state.bg_hsv = np.array(data.get("bg_hsv")) if data.get("bg_hsv") else None
            st.success("✅ Letzte Kalibrierung geladen.")
    except FileNotFoundError:
        st.warning("⚠️ Keine gespeicherte Kalibrierung gefunden.")

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "jpeg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Reset state on new file
if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_hsv", "hema_hsv", "bg_hsv"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400
    st.session_state.last_auto_run = 0

# -------------------- Bild vorbereiten --------------------
colW1, colW2 = st.columns([2, 1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite", 400, 2000, st.session_state.disp_width, step=100)
    st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig, W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH / W_orig
image_disp = cv2.resize(image_orig, (DISPLAY_WIDTH, int(H_orig * scale)), interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp, cv2.COLOR_RGB2HSV)

# -------------------- Parameter --------------------
st.markdown("### ⚙️ Filterparameter")
col1, col2, col3 = st.columns(3)
with col1:
    blur_kernel = st.slider("🔧 Blur (ungerade empfohlen)", 1, 21, 5, step=1)
    blur_kernel = ensure_odd(blur_kernel)  # zwingt ungerade Kernelgröße
    min_area = st.number_input("📏 Mindestfläche (px)", 10, 2000, 100)
with col2:
    alpha = st.slider("🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1)
with col3:
    circle_radius = st.slider("⚪ Kreisradius (Display-Px)", 1, 20, 5)

# -------------------- Modi (exklusiv) --------------------
st.markdown("### 🎨 Modus auswählen (exklusiv)")
mode = st.radio("Modus", [
    "Keine", "AEC markieren (Kalibrierung)", "Hämatoxylin markieren (Kalibrierung)",
    "Hintergrund markieren", "AEC manuell hinzufügen", "Hämatoxylin manuell hinzufügen",
    "Punkt löschen (alle Kategorien)"
], index=0)
# Map to internal flags
aec_mode = mode == "AEC markieren (Kalibrierung)"
hema_mode = mode == "Hämatoxylin markieren (Kalibrierung)"
bg_mode = mode == "Hintergrund markieren"
manual_aec_mode = mode == "AEC manuell hinzufügen"
manual_hema_mode = mode == "Hämatoxylin manuell hinzufügen"
delete_mode = mode == "Punkt löschen (alle Kategorien)"

# -------------------- Quick actions --------------------
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    if st.button("🧹 Alle markierten & manuellen Punkte löschen"):
        for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
            st.session_state[k] = []
        st.success("Alle Punkte gelöscht.")
with colB:
    if st.button("🧾 Kalibrierung zurücksetzen"):
        st.session_state.aec_hsv = None
        st.session_state.hema_hsv = None
        st.session_state.bg_hsv = None
        st.info("Kalibrierungswerte zurückgesetzt.")
with colC:
    # Auto-Run Button: Wir erhöhen einen Zähler, damit Reruns eindeutig sind
    if st.button("🤖 Auto-Erkennung ausführen"):
        st.session_state.last_auto_run = st.session_state.last_auto_run + 1

# -------------------- Bildanzeige (mit Markierungen) --------------------
marked_disp = image_disp.copy()

# Markiere bestehende Punkte
for points_list, color in [
    (st.session_state.aec_points, (255, 0, 0)),
    (st.session_state.hema_points, (0, 0, 255)),
    (st.session_state.manual_aec or [], (255, 165, 0)),
    (st.session_state.manual_hema or [], (128, 0, 128)),
    (st.session_state.bg_points or [], (255, 255, 0)),
]:
    for (x, y) in points_list:
        cv2.circle(marked_disp, (x, y), circle_radius, color, -1)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key=f"clickable_image_{st.session_state.last_auto_run}", width=DISPLAY_WIDTH)

# -------------------- Klicklogik (einheitlich + dedup) --------------------
if coords:
    x, y = int(coords["x"]), int(coords["y"])
    if delete_mode:
        for key in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
            st.session_state[key] = [p for p in st.session_state[key] if not is_near(p, (x, y), circle_radius)]
    elif aec_mode:
        st.session_state.aec_points = [(x, y)]  # Nur den letzten Punkt setzen
        st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points, hsv_disp)
        st.session_state.aec_points = []  # Direkt löschen
        st.success("✅ AEC-Kalibrierung durchgeführt.")
    elif hema_mode:
        st.session_state.hema_points = [(x, y)]
        st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points, hsv_disp)
        st.session_state.hema_points = []
        st.success("✅ Hämatoxylin-Kalibrierung durchgeführt.")
    elif bg_mode:
        st.session_state.bg_points = [(x, y)]
        st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points, hsv_disp)
        st.session_state.bg_points = []
        st.success("✅ Hintergrund-Kalibrierung durchgeführt.")
    
    elif manual_aec_mode:
        st.session_state.manual_aec.append((x, y))
    elif manual_hema_mode:
        st.session_state.manual_hema.append((x, y))

# dedup session lists to avoid ghosting
for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4, circle_radius//2))

# -------------------- Kalibrierung speichern (getrennt) --------------------
st.markdown("### ⚙️ Kalibrierung")

col_cal1, col_cal2, col_cal3 = st.columns(3)

with col_cal1:
    if st.button("⚡ AEC kalibrieren"):
        if st.session_state.aec_points:
            st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points, hsv_disp)
            st.session_state.aec_points = []  # Punkte direkt löschen
            st.success("✅ AEC-Kalibrierung gespeichert.")
        else:
            st.warning("⚠️ Keine AEC-Punkte vorhanden.")

with col_cal2:
    if st.button("⚡ Hämatoxylin kalibrieren"):
        if st.session_state.hema_points:
            st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points, hsv_disp)
            st.session_state.hema_points = []  # Punkte direkt löschen
            st.success("✅ Hämatoxylin-Kalibrierung gespeichert.")
        else:
            st.warning("⚠️ Keine Hämatoxylin-Punkte vorhanden.")

with col_cal3:
    if st.button("⚡ Hintergrund kalibrieren"):
        if st.session_state.bg_points:
            st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points, hsv_disp)
            st.session_state.bg_points = []  # Punkte direkt löschen
            st.success("✅ Hintergrund-Kalibrierung gespeichert.")
        else:
            st.warning("⚠️ Keine Hintergrund-Punkte vorhanden.")

st.markdown("### 💾 Kalibrierung speichern/laden")
col_save, col_load = st.columns(2)
with col_save:
    if st.button("💾 Letzte Kalibrierung speichern"):
        save_last_calibration()
with col_load:
    if st.button("📂 Letzte Kalibrierung laden"):
        load_last_calibration()

# -------------------- Auto-Erkennung (reaktiv bei last_auto_run Veränderung) --------------------
# Wenn last_auto_run > 0, führe Erkennung aus
if st.session_state.last_auto_run > 0:
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (ensure_odd(blur_kernel), ensure_odd(blur_kernel)), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    aec_detected = []
    hema_detected = []

    # Wenn Kalibrierung existiert, erstelle Masken
    if st.session_state.aec_hsv:
        hmin, hmax, smin, smax, vmin, vmax = st.session_state.aec_hsv
        mask_aec = apply_hue_wrap(hsv_proc, hmin, hmax, smin, smax, vmin, vmax)
    else:
        mask_aec = np.zeros(hsv_proc.shape[:2], dtype=np.uint8)

    if st.session_state.hema_hsv:
        hmin, hmax, smin, smax, vmin, vmax = st.session_state.hema_hsv
        mask_hema = apply_hue_wrap(hsv_proc, hmin, hmax, smin, smax, vmin, vmax)
    else:
        mask_hema = np.zeros(hsv_proc.shape[:2], dtype=np.uint8)

    # Wenn Hintergrundkalibrierung vorhanden, entferne diese Bereiche
    if st.session_state.bg_hsv:
        hmin, hmax, smin, smax, vmin, vmax = st.session_state.bg_hsv
        mask_bg = apply_hue_wrap(hsv_proc, hmin, hmax, smin, smax, vmin, vmax)
        # Masken bereinigen: setze BG-Pixel auf 0 in AEC/HEMA
        mask_aec = cv2.bitwise_and(mask_aec, cv2.bitwise_not(mask_bg))
        mask_hema = cv2.bitwise_and(mask_hema, cv2.bitwise_not(mask_bg))

    # Optional: kleine Morphologie zum Glätten
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_aec = cv2.morphologyEx(mask_aec, cv2.MORPH_OPEN, kernel)
    mask_hema = cv2.morphologyEx(mask_hema, cv2.MORPH_OPEN, kernel)

    # Konturen -> Zentren
    aec_detected = get_centers(mask_aec, int(min_area))
    hema_detected = get_centers(mask_hema, int(min_area))

    # entferne Punkte, die in der Nähe von bg_points liegen (Artefakte)
    if st.session_state.bg_points:
        cleaned_aec = []
        for p in aec_detected:
            if not any(is_near(p, bgp, r=max(6, circle_radius)) for bgp in st.session_state.bg_points):
                cleaned_aec.append(p)
        aec_detected = cleaned_aec

        cleaned_hema = []
        for p in hema_detected:
            if not any(is_near(p, bgp, r=max(6, circle_radius)) for bgp in st.session_state.bg_points):
                cleaned_hema.append(p)
        hema_detected = cleaned_hema

    # Kombiniere automatische und manuelle Punkte (manuelle ergänzen/überschreiben)
    # Wir wollen manuelle Punkte beibehalten + automatisch erkannte hinzufügen (falls nicht nahe an manuellen)
    merged_aec = list(st.session_state.manual_aec)  # manuelle behalten
    for p in aec_detected:
        if not any(is_near(p, q, r=max(6, circle_radius)) for q in merged_aec):
            merged_aec.append(p)

    merged_hema = list(st.session_state.manual_hema)
    for p in hema_detected:
        if not any(is_near(p, q, r=max(6, circle_radius)) for q in merged_hema):
            merged_hema.append(p)

    # dedup final
    st.session_state.aec_points = dedup_points(merged_aec, min_dist=max(4, circle_radius//2))
    st.session_state.hema_points = dedup_points(merged_hema, min_dist=max(4, circle_radius//2))

    # Reset run-counter so user can re-run cleanly
    st.session_state.last_auto_run = 0

# -------------------- Anzeige der Gesamtzahlen --------------------
all_aec = (st.session_state.aec_points or [])  # enthält bereits manual + auto nach merge
all_hema = (st.session_state.hema_points or [])
st.markdown(f"### 🔢 Gesamt: AEC={len(all_aec) + len(st.session_state.manual_aec)}, Hämatoxylin={len(all_hema) + len(st.session_state.manual_hema)}")

# -------------------- CSV Export --------------------
df_list = []
for (x, y) in all_aec:
    df_list.append({"X_display": x, "Y_display": y, "Type": "AEC"})
for (x, y) in all_hema:
    df_list.append({"X_display": x, "Y_display": y, "Type": "Hämatoxylin"})
if df_list:
    df = pd.DataFrame(df_list)
    df["X_original"] = (df["X_display"] / scale).round().astype("Int64")
    df["Y_original"] = (df["Y_display"] / scale).round().astype("Int64")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")
# -------------------- Punktanzahl anzeigen --------------------
auto_aec = len(st.session_state.aec_points)
auto_hema = len(st.session_state.hema_points)
manual_aec = len(st.session_state.manual_aec)
manual_hema = len(st.session_state.manual_hema)

st.markdown(f"""
### 🔢 Zellkern-Zählung
- 🧠 Auto AEC: **{auto_aec}**
- 🧠 Auto Hämatoxylin: **{auto_hema}**
- ✋ Manuell AEC: **{manual_aec}**
- ✋ Manuell Hämatoxylin: **{manual_hema}**
""")

# -------------------- Debug Info (optional) --------------------
with st.expander("🧠 Debug Info"):
    st.write({
        "aec_hsv": st.session_state.aec_hsv,
        "hema_hsv": st.session_state.hema_hsv,
        "bg_hsv": st.session_state.bg_hsv,
        "aec_points_count": len(st.session_state.aec_points),
        "hema_points_count": len(st.session_state.hema_points),
        "manual_aec": st.session_state.manual_aec,
        "manual_hema": st.session_state.manual_hema,
        "bg_points": st.session_state.bg_points,
        "last_auto_run": st.session_state.last_auto_run
    })
