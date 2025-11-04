# canvas_safe_fixed_clean_v2.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json
import io
from pathlib import Path

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
    """Robuste findContours-Variante (kompatibel mit OpenCV 4+)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00", 0) != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
    return centers


def compute_hsv_range(points, hsv_img, radius=5):
    """
    Berechne HSV-Bereich aus mehreren Kalibrierpunkten (robust gegen Ausreißer)
    - Median statt Min/Max
    - Dynamischer Radius
    - Hue-Wraparound für Rot (0°-180°)
    - Dynamische Toleranzen je nach Anzahl der Punkte
    - **Berücksichtigt die Sidebar-Buffer (buffer_h/s/v) aus session_state**
    """

    if not points:
        return None

    vals = []
    for (x, y) in points:
        x_min = max(0, x - radius)
        x_max = min(hsv_img.shape[1], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(hsv_img.shape[0], y + radius + 1)
        region = hsv_img[y_min:y_max, x_min:x_max]
        if region.size > 0:
            vals.append(region.reshape(-1, 3))

    if not vals:
        return None

    vals = np.vstack(vals)
    h = vals[:, 0].astype(int)
    s = vals[:, 1].astype(int)
    v = vals[:, 2].astype(int)

    # Medianwerte
    h_med = np.median(h)
    s_med = np.median(s)
    v_med = np.median(v)

    # Dynamische Toleranz
    n_points = len(points)
    tol_h = min(25, 10 + n_points * 3)
    tol_s = min(80, 30 + n_points * 10)
    tol_v = min(80, 30 + n_points * 10)

    # Hue-Wraparound für Rot
    if np.mean(h) > 150 or np.mean(h) < 20:
        h_med = np.median(np.where(h < 90, h + 180, h)) % 180
        tol_h = min(30, tol_h + 5)

    # Extra-Buffer aus Sidebar (jetzt wirksam)
    buffer_h = int(st.session_state.get("buffer_h", 0) or 0)
    buffer_s = int(st.session_state.get("buffer_s", 0) or 0)
    buffer_v = int(st.session_state.get("buffer_v", 0) or 0)

    h_min = int((h_med - tol_h - buffer_h) % 180)
    h_max = int((h_med + tol_h + buffer_h) % 180)
    s_min = max(0, int(s_med - tol_s - buffer_s))
    s_max = min(255, int(s_med + tol_s + buffer_s))
    v_min = max(0, int(v_med - tol_v - buffer_v))
    v_max = min(255, int(v_med + tol_v + buffer_v))

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


def remove_near(points, forbidden_points, r):
    """Hilfsfunktion: entferne Punkte, die nahe an forbidden_points liegen."""
    if not forbidden_points:
        return points
    return [p for p in points if not any(is_near(p, q, r) for q in forbidden_points)]

from pathlib import Path
import json
import numpy as np
import streamlit as st

# -------------------- Kalibrierung speichern --------------------
def save_last_calibration(filename="kalibrierung.json"):
    """Speichert die aktuelle Kalibrierung sicher in JSON."""
    data = {}
    for key in ["aec_hsv", "hema_hsv", "bg_hsv"]:
        val = st.session_state.get(key)
        if isinstance(val, np.ndarray):
            data[key] = val.tolist()
        elif isinstance(val, list):
            data[key] = val
        else:
            data[key] = None

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f)
        st.success("💾 Kalibrierung gespeichert.")
    except Exception as e:
        st.error(f"Fehler beim Speichern der Kalibrierung: {e}")

# -------------------- Kalibrierung laden --------------------
def load_last_calibration(filename="kalibrierung.json"):
    """Lädt Kalibrierung aus JSON und setzt Session State.
    Löst automatisch die Anzeige/AEC-Erkennung aus."""
    path = Path(filename)
    if not path.exists():
        st.warning(f"⚠️ Datei {filename} nicht gefunden.")
        return

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Setze Session State
        st.session_state.aec_hsv = np.array(data["aec_hsv"]) if data.get("aec_hsv") else None
        st.session_state.hema_hsv = np.array(data["hema_hsv"]) if data.get("hema_hsv") else None
        st.session_state.bg_hsv = np.array(data["bg_hsv"]) if data.get("bg_hsv") else None

        # Trigger Auto-Erkennung / Anzeige
        st.session_state.last_auto_run = 1

        st.success(f"✅ Kalibrierung geladen aus {filename}.")
        st.write({
            "aec_hsv": st.session_state.aec_hsv,
            "hema_hsv": st.session_state.hema_hsv,
            "bg_hsv": st.session_state.bg_hsv
        })  # Debug: zeigt die geladenen Werte
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Kalibrierung: {e}")


# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler (fixed)", layout="wide")
st.title("🧬 Zellkern-Zähler – 3 Punkte Kalibration (v2)")

# -------------------- Session State --------------------
default_keys = [
    "aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema",
    "aec_hsv", "hema_hsv", "bg_hsv", "last_file", "disp_width", "last_auto_run",
    # buffer sliders und last_coords für Ghost-Click-Handling
    "buffer_h", "buffer_s", "buffer_v", "last_coords"
]
for key in default_keys:
    if key not in st.session_state:
        if "points" in key or "manual" in key:
            st.session_state[key] = []
        elif key == "disp_width":
            st.session_state[key] = 1400
        elif key in ("buffer_h", "buffer_s", "buffer_v"):
            st.session_state[key] = 0
        else:
            st.session_state[key] = None

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
    st.session_state.last_coords = None

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

# -------------------- Sidebar: Parameter & Aktionen --------------------
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")

# -------------------- Filterparameter --------------------
blur_kernel = st.sidebar.slider(
    "🔧 Blur (ungerade empfohlen)", 1, 21, 5, step=1, key="blur_slider"
)
blur_kernel = ensure_odd(blur_kernel)

min_area = st.sidebar.number_input(
    "📏 Mindestfläche (px)", 10, 2000, 100, key="min_area_input"
)

alpha = st.sidebar.slider(
    "🌗 Alpha (Kontrast)", 0.1, 3.0, 1.0, step=0.1, key="alpha_slider"
)

circle_radius = st.sidebar.slider(
    "⚪ Kreisradius (Display-Px)", 1, 20, 5, key="circle_radius_slider"
)

calib_radius = st.sidebar.slider(
    "🎯 Kalibrierungsradius (Pixel)", 1, 15, 5, key="calib_radius_slider"
)

# -------------------- HSV-Toleranz --------------------
st.sidebar.markdown("### 🛠️ Kalibrierungs-Toleranz")
buffer_h = st.sidebar.slider("Hue-Toleranz", 0, 30, int(st.session_state.buffer_h or 0), key="buffer_h")
buffer_s = st.sidebar.slider("Sättigung-Toleranz", 0, 100, int(st.session_state.buffer_s or 0), key="buffer_s")
buffer_v = st.sidebar.slider("Value-Toleranz", 0, 100, int(st.session_state.buffer_v or 0), key="buffer_v")

# -------------------- Modus auswählen --------------------
st.sidebar.markdown("### 🎨 Modus auswählen (exklusiv)")
mode = st.sidebar.radio(
    "Modus",
    [
        "Keine",
        "AEC markieren (Kalibrierung)",
        "Hämatoxylin markieren (Kalibrierung)",
        "Hintergrund markieren",
        "AEC manuell hinzufügen",
        "Hämatoxylin manuell hinzufügen",
        "Punkt löschen (alle Kategorien)"
    ],
    index=0,
    key="mode_radio"
)

# interne Flags für Klicklogik
aec_mode = mode == "AEC markieren (Kalibrierung)"
hema_mode = mode == "Hämatoxylin markieren (Kalibrierung)"
bg_mode = mode == "Hintergrund markieren"
manual_aec_mode = mode == "AEC manuell hinzufügen"
manual_hema_mode = mode == "Hämatoxylin manuell hinzufügen"
delete_mode = mode == "Punkt löschen (alle Kategorien)"

# -------------------- Quick Actions --------------------
st.sidebar.markdown("### ⚡ Schnellaktionen")
if st.sidebar.button("🧹 Alle markierten & manuellen Punkte löschen", key="btn_clear_points"):
    for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
        st.session_state[k] = []
    st.success("Alle Punkte gelöscht.")

if st.sidebar.button("🧾 Kalibrierung zurücksetzen", key="btn_reset_calib"):
    st.session_state.aec_hsv = None
    st.session_state.hema_hsv = None
    st.session_state.bg_hsv = None
    st.info("Kalibrierungswerte zurückgesetzt.")

# Auto-Run: inkrementiere und rerun, damit Ergebnis sofort sichtbar ist
if st.sidebar.button("🤖 Auto-Erkennung ausführen", key="btn_auto_run"):
    st.session_state.last_auto_run = (st.session_state.last_auto_run or 0) + 1
    st.rerun()

# -------------------- Kalibrierung speichern/laden --------------------
st.sidebar.markdown("### 💾 Kalibrierung")
if st.sidebar.button("💾 Letzte Kalibrierung speichern", key="btn_save_calib"):
    save_last_calibration()

if st.sidebar.button("📂 Letzte Kalibrierung laden", key="btn_load_calib"):
    load_last_calibration()

# -------------------- Bildanzeige (mit Markierungen) --------------------
marked_disp = image_disp.copy()

# Markiere bestehende Punkte
for points_list, color in [
    (st.session_state.aec_points, (255, 100, 100)),      # AEC = hellrot
    (st.session_state.hema_points, (100, 100, 255)),     # Hämatoxylin = hellblau
    (st.session_state.bg_points, (255, 255, 0)),         # Hintergrund = gelb
    (st.session_state.manual_aec or [], (255, 165, 0)),  # manuell AEC = orange
    (st.session_state.manual_hema or [], (128, 0, 128)), # manuell Hämatoxylin = lila
]:
    for (x, y) in points_list:
        cv2.circle(marked_disp, (x, y), circle_radius, color, -1)


coords = streamlit_image_coordinates(Image.fromarray(marked_disp), key=f"clickable_image_{st.session_state.last_auto_run}", width=DISPLAY_WIDTH)

# -------------------- Klicklogik (mehrpunktfähig + dedup + Ghost-Click fix) --------------------
if coords:
    # coords liefert {'x':..., 'y':...}
    x, y = int(coords["x"]), int(coords["y"])
    last = st.session_state.get("last_coords")
    if last is None or (x, y) != last:
        # verarbeite Klick nur, wenn er neu ist
        if delete_mode:
            for key in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
                st.session_state[key] = [p for p in st.session_state[key] if not is_near(p, (x, y), circle_radius)]

        elif aec_mode:
            st.session_state.aec_points.append((x, y))
            st.info(f"📍 AEC-Punkt hinzugefügt ({x}, {y})")

        elif hema_mode:
            st.session_state.hema_points.append((x, y))
            st.info(f"📍 Hämatoxylin-Punkt hinzugefügt ({x}, {y})")

        elif bg_mode:
            st.session_state.bg_points.append((x, y))
            st.info(f"📍 Hintergrund-Punkt hinzugefügt ({x}, {y})")

        elif manual_aec_mode:
            st.session_state.manual_aec.append((x, y))

        elif manual_hema_mode:
            st.session_state.manual_hema.append((x, y))

        # speichere die letzte verarbeitete Koordinate
        st.session_state.last_coords = (x, y)
else:
    # kein Klick -> clear last_coords damit neuer Klick wieder verarbeitet wird
    st.session_state.last_coords = None

# dedup session lists to avoid ghosting
for k in ["aec_points", "hema_points", "bg_points", "manual_aec", "manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k], min_dist=max(4, circle_radius // 2))

# -------------------- Kalibrierung speichern (Mehrpunkt) --------------------
st.markdown("### ⚙️ Kalibrierung")

col_cal1, col_cal2, col_cal3 = st.columns(3)

with col_cal1:
    if st.button("⚡ AEC kalibrieren"):
        if st.session_state.aec_points:
            st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points, hsv_disp)
            count = len(st.session_state.aec_points)
            st.session_state.aec_points = []
            st.success(f"✅ AEC-Kalibrierung aus {count} Punkten gespeichert.")
        else:
            st.warning("⚠️ Keine AEC-Punkte vorhanden.")

with col_cal2:
    if st.button("⚡ Hämatoxylin kalibrieren"):
        if st.session_state.hema_points:
            st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points, hsv_disp)
            count = len(st.session_state.hema_points)
            st.session_state.hema_points = []
            st.success(f"✅ Hämatoxylin-Kalibrierung aus {count} Punkten gespeichert.")
        else:
            st.warning("⚠️ Keine Hämatoxylin-Punkte vorhanden.")

with col_cal3:
    if st.button("⚡ Hintergrund kalibrieren"):
        if st.session_state.bg_points:
            st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points, hsv_disp)
            count = len(st.session_state.bg_points)
            st.session_state.bg_points = []
            st.success(f"✅ Hintergrund-Kalibrierung aus {count} Punkten gespeichert.")
        else:
            st.warning("⚠️ Keine Hintergrund-Punkte vorhanden.")

# -------------------- Auto-Erkennung (reaktiv bei last_auto_run Veränderung) --------------------
# Wenn last_auto_run > 0, führe Erkennung aus
if st.session_state.last_auto_run and st.session_state.last_auto_run > 0:
    proc = cv2.convertScaleAbs(image_disp, alpha=alpha, beta=0)
    if blur_kernel > 1:
        proc = cv2.GaussianBlur(proc, (ensure_odd(blur_kernel), ensure_odd(blur_kernel)), 0)
    hsv_proc = cv2.cvtColor(proc, cv2.COLOR_RGB2HSV)

    aec_detected = []
    hema_detected = []

    # Wenn Kalibrierung existiert, erstelle Masken (mit None-Checks)
    if st.session_state.aec_hsv is not None:
        hmin, hmax, smin, smax, vmin, vmax = map(int, st.session_state.aec_hsv)
        mask_aec = apply_hue_wrap(hsv_proc, hmin, hmax, smin, smax, vmin, vmax)
    else:
        mask_aec = np.zeros(hsv_proc.shape[:2], dtype=np.uint8)

    if st.session_state.hema_hsv is not None:
        hmin, hmax, smin, smax, vmin, vmax = map(int, st.session_state.hema_hsv)
        mask_hema = apply_hue_wrap(hsv_proc, hmin, hmax, smin, smax, vmin, vmax)
    else:
        mask_hema = np.zeros(hsv_proc.shape[:2], dtype=np.uint8)

    # Wenn Hintergrundkalibrierung vorhanden, entferne diese Bereiche
    if st.session_state.bg_hsv is not None:
        hmin, hmax, smin, smax, vmin, vmax = map(int, st.session_state.bg_hsv)
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
        aec_detected = remove_near(aec_detected, st.session_state.bg_points, r=max(6, circle_radius))
        hema_detected = remove_near(hema_detected, st.session_state.bg_points, r=max(6, circle_radius))

    # Kombiniere automatische und manuelle Punkte (manuelle behalten)
    merged_aec = list(st.session_state.manual_aec)
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

    # Hinweis: wir setzen last_auto_run **nicht** zurück, damit das Bild direkt
    # und konsistent angezeigt bleibt. Falls du einen Reset willst, benutze einen
    # eigenen Knopf "Reset Auto-Run".

# -------------------- Anzeige der Gesamtzahlen --------------------
# st.session_state.aec_points / hema_points enthalten nach Merge bereits manuelle+auto
all_aec = (st.session_state.aec_points or [])
all_hema = (st.session_state.hema_points or [])
st.markdown(f"### 🔢 Gesamt: AEC={len(all_aec)}, Hämatoxylin={len(all_hema)}")

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
        "last_auto_run": st.session_state.last_auto_run,
        "last_coords": st.session_state.last_coords
    })
