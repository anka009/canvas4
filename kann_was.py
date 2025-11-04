# canvas_fixed_full.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import pandas as pd
import json
from pathlib import Path

# -------------------- Hilfsfunktionen --------------------
def is_near(p1, p2, r=10):
    return np.linalg.norm(np.array(p1)-np.array(p2)) < r

def dedup_points(points, min_dist=6):
    out = []
    for p in points:
        if not any(is_near(p, q, r=min_dist) for q in out):
            out.append(p)
    return out

def get_centers(mask, min_area=50):
    res = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    centers = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            M = cv2.moments(c)
            if M.get("m00",0) != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centers.append((cx,cy))
    return centers

def compute_hsv_range(points, hsv_img, radius=5):
    if not points:
        return None
    vals = []
    for x,y in points:
        x_min, x_max = max(0,x-radius), min(hsv_img.shape[1],x+radius+1)
        y_min, y_max = max(0,y-radius), min(hsv_img.shape[0],y+radius+1)
        region = hsv_img[y_min:y_max,x_min:x_max]
        if region.size > 0:
            vals.append(region.reshape(-1,3))
    if not vals:
        return None
    vals = np.vstack(vals)
    h,s,v = vals[:,0].astype(int), vals[:,1].astype(int), vals[:,2].astype(int)
    h_med, s_med, v_med = np.median(h), np.median(s), np.median(v)
    n_points = len(points)
    tol_h = min(25,10+n_points*3)
    tol_s = min(80,30+n_points*10)
    tol_v = min(80,30+n_points*10)
    # Hue-Wraparound für Rot
    if np.mean(h) > 150 or np.mean(h)<20:
        h_med = np.median(np.where(h<90,h+180,h))%180
        tol_h = min(30,tol_h+5)
    h_min = int((h_med-tol_h)%180)
    h_max = int((h_med+tol_h)%180)
    s_min = max(0,int(s_med-tol_s))
    s_max = min(255,int(s_med+tol_s))
    v_min = max(0,int(v_med-tol_v))
    v_max = min(255,int(v_med+tol_v))
    return (h_min,h_max,s_min,s_max,v_min,v_max)

def apply_hue_wrap(hsv_img,hmin,hmax,smin,smax,vmin,vmax):
    if hmin<=hmax:
        mask = cv2.inRange(hsv_img,np.array([hmin,smin,vmin]),np.array([hmax,smax,vmax]))
    else:
        mask_lo = cv2.inRange(hsv_img,np.array([0,smin,vmin]),np.array([hmax,smax,vmax]))
        mask_hi = cv2.inRange(hsv_img,np.array([hmin,smin,vmin]),np.array([180,smax,vmax]))
        mask = cv2.bitwise_or(mask_lo,mask_hi)
    return mask

def ensure_odd(k):
    return k if k%2==1 else k+1

def remove_near(points, forbidden_points, r):
    if not forbidden_points:
        return points
    return [p for p in points if not any(is_near(p,q,r) for q in forbidden_points)]

def save_calibration(filename="kalibrierung.json"):
    data = {
        "aec_hsv": st.session_state.aec_hsv.tolist() if st.session_state.aec_hsv is not None else None,
        "hema_hsv": st.session_state.hema_hsv.tolist() if st.session_state.hema_hsv is not None else None,
        "bg_hsv": st.session_state.bg_hsv.tolist() if st.session_state.bg_hsv is not None else None
    }
    with open(filename,"w",encoding="utf-8") as f:
        json.dump(data,f)
    st.success("💾 Kalibrierung gespeichert.")

def load_calibration(filename="kalibrierung.json"):
    path = Path(filename)
    if not path.exists():
        st.warning("⚠️ Keine gespeicherte Kalibrierung gefunden.")
        return
    try:
        with path.open("r",encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.aec_hsv = np.array(data.get("aec_hsv")) if data.get("aec_hsv") else None
        st.session_state.hema_hsv = np.array(data.get("hema_hsv")) if data.get("hema_hsv") else None
        st.session_state.bg_hsv = np.array(data.get("bg_hsv")) if data.get("bg_hsv") else None
        st.success("✅ Letzte Kalibrierung geladen.")
    except Exception as e:
        st.error(f"Fehler beim Laden der Kalibrierung: {e}")

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Zellkern-Zähler Fixed", layout="wide")
st.title("🧬 Zellkern-Zähler – 3 Punkte Kalibrierung")

# -------------------- Session State --------------------
default_keys = [
    "aec_points","hema_points","bg_points","manual_aec","manual_hema",
    "aec_hsv","hema_hsv","bg_hsv","last_file","disp_width","last_auto_run",
    "last_click"
]
for key in default_keys:
    if key not in st.session_state:
        if "points" in key or "manual" in key:
            st.session_state[key] = []
        elif key=="disp_width":
            st.session_state[key] = 1400
        else:
            st.session_state[key] = None

# -------------------- File upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg","jpeg","png","tif","tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

if uploaded_file.name != st.session_state.last_file:
    for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
        st.session_state[k] = []
    for k in ["aec_hsv","hema_hsv","bg_hsv"]:
        st.session_state[k] = None
    st.session_state.last_file = uploaded_file.name
    st.session_state.disp_width = 1400
    st.session_state.last_auto_run = 0

# -------------------- Bild vorbereiten --------------------
colW1,colW2 = st.columns([2,1])
with colW1:
    DISPLAY_WIDTH = st.slider("📐 Bildbreite",400,2000,st.session_state.disp_width,step=100)
    st.session_state.disp_width = DISPLAY_WIDTH

image_orig = np.array(Image.open(uploaded_file).convert("RGB"))
H_orig,W_orig = image_orig.shape[:2]
scale = DISPLAY_WIDTH/W_orig
image_disp = cv2.resize(image_orig,(DISPLAY_WIDTH,int(H_orig*scale)),interpolation=cv2.INTER_AREA)
hsv_disp = cv2.cvtColor(image_disp,cv2.COLOR_RGB2HSV)

# -------------------- Sidebar --------------------
st.sidebar.markdown("### ⚙️ Filter & Kalibrierung")
blur_kernel = ensure_odd(st.sidebar.slider("🔧 Blur (ungerade)",1,21,5,step=1))
min_area = st.sidebar.number_input("📏 Mindestfläche (px)",10,2000,100)
alpha = st.sidebar.slider("🌗 Alpha (Kontrast)",0.1,3.0,1.0,step=0.1)
circle_radius = st.sidebar.slider("⚪ Kreisradius (Display-Px)",1,20,5)
calib_radius = st.sidebar.slider("🎯 Kalibrierungsradius (Pixel)",1,15,5)

st.sidebar.markdown("### 🛠️ Kalibrierungs-Toleranz")
buffer_h = st.sidebar.slider("Hue-Toleranz",1,30,8)
buffer_s = st.sidebar.slider("Sättigung-Toleranz",1,100,30)
buffer_v = st.sidebar.slider("Value-Toleranz",1,100,25)

st.sidebar.markdown("### 🎨 Modus auswählen")
mode = st.sidebar.radio(
    "Modus",
    ["Keine","AEC markieren (Kalibrierung)","Hämatoxylin markieren (Kalibrierung)",
     "Hintergrund markieren","AEC manuell hinzufügen","Hämatoxylin manuell hinzufügen",
     "Punkt löschen (alle Kategorien)"]
)

# interne Flags
aec_mode = mode=="AEC markieren (Kalibrierung)"
hema_mode = mode=="Hämatoxylin markieren (Kalibrierung)"
bg_mode = mode=="Hintergrund markieren"
manual_aec_mode = mode=="AEC manuell hinzufügen"
manual_hema_mode = mode=="Hämatoxylin manuell hinzufügen"
delete_mode = mode=="Punkt löschen (alle Kategorien)"

st.sidebar.markdown("### ⚡ Schnellaktionen")
if st.sidebar.button("🧹 Alle Punkte löschen"):
    for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
        st.session_state[k] = []

if st.sidebar.button("🧾 Kalibrierung zurücksetzen"):
    st.session_state.aec_hsv = None
    st.session_state.hema_hsv = None
    st.session_state.bg_hsv = None

if st.sidebar.button("🤖 Auto-Erkennung ausführen"):
    st.session_state.last_auto_run += 1

st.sidebar.markdown("### 💾 Kalibrierung")
if st.sidebar.button("💾 Speichern"):
    save_calibration()
if st.sidebar.button("📂 Laden"):
    load_calibration()

# -------------------- Bildanzeige --------------------
marked_disp = image_disp.copy()
for points_list,color in [
    (st.session_state.aec_points,(255,100,100)),
    (st.session_state.hema_points,(100,100,255)),
    (st.session_state.bg_points,(255,255,0)),
    (st.session_state.manual_aec,(255,165,0)),
    (st.session_state.manual_hema,(128,0,128))
]:
    for x,y in points_list:
        cv2.circle(marked_disp,(x,y),circle_radius,color,-1)

coords = streamlit_image_coordinates(Image.fromarray(marked_disp),key="clickable_image",width=DISPLAY_WIDTH)

# -------------------- Klicklogik --------------------
if coords and st.session_state.last_click != (coords["x"],coords["y"]):
    x,y = int(coords["x"]),int(coords["y"])
    st.session_state.last_click = (x,y)

    if delete_mode:
        for key in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
            st.session_state[key] = [p for p in st.session_state[key] if not is_near(p,(x,y),circle_radius)]
    elif aec_mode:
        st.session_state.aec_points.append((x,y))
    elif hema_mode:
        st.session_state.hema_points.append((x,y))
    elif bg_mode:
        st.session_state.bg_points.append((x,y))
    elif manual_aec_mode:
        st.session_state.manual_aec.append((x,y))
    elif manual_hema_mode:
        st.session_state.manual_hema.append((x,y))

# dedup
for k in ["aec_points","hema_points","bg_points","manual_aec","manual_hema"]:
    st.session_state[k] = dedup_points(st.session_state[k],min_dist=max(4,circle_radius//2))

# -------------------- Kalibrierung Buttons --------------------
col1,col2,col3 = st.columns(3)
with col1:
    if st.button("⚡ AEC kalibrieren"):
        if st.session_state.aec_points:
            st.session_state.aec_hsv = compute_hsv_range(st.session_state.aec_points,hsv_disp)
            st.session_state.aec_points=[]
with col2:
    if st.button("⚡ Hämatoxylin kalibrieren"):
        if st.session_state.hema_points:
            st.session_state.hema_hsv = compute_hsv_range(st.session_state.hema_points,hsv_disp)
            st.session_state.hema_points=[]
with col3:
    if st.button("⚡ Hintergrund kalibrieren"):
        if st.session_state.bg_points:
            st.session_state.bg_hsv = compute_hsv_range(st.session_state.bg_points,hsv_disp)
            st.session_state.bg_points=[]

# -------------------- Auto-Erkennung --------------------
if st.session_state.last_auto_run>0:
    proc = cv2.convertScaleAbs(image_disp,alpha=alpha,beta=0)
    if blur_kernel>1:
        proc = cv2.GaussianBlur(proc,(ensure_odd(blur_kernel),ensure_odd(blur_kernel)),0)
    hsv_proc = cv2.cvtColor(proc,cv2.COLOR_RGB2HSV)
    aec_detected=[]
    hema_detected=[]
    # Masken
    mask_aec = apply_hue_wrap(hsv_proc,*map(int,st.session_state.aec_hsv)) if st.session_state.aec_hsv is not None else np.zeros(hsv_proc.shape[:2],dtype=np.uint8)
    mask_hema = apply_hue_wrap(hsv_proc,*map(int,st.session_state.hema_hsv)) if st.session_state.hema_hsv is not None else np.zeros(hsv_proc.shape[:2],dtype=np.uint8)
    if st.session_state.bg_hsv is not None:
        mask_bg = apply_hue_wrap(hsv_proc,*map(int,st.session_state.bg_hsv))
        mask_aec = cv2.bitwise_and(mask_aec,cv2.bitwise_not(mask_bg))
        mask_hema = cv2.bitwise_and(mask_hema,cv2.bitwise_not(mask_bg))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_aec=cv2.morphologyEx(mask_aec,cv2.MORPH_OPEN,kernel)
    mask_hema=cv2.morphologyEx(mask_hema,cv2.MORPH_OPEN,kernel)
    aec_detected=get_centers(mask_aec,int(min_area))
    hema_detected=get_centers(mask_hema,int(min_area))
    # entfernen nahe BG
    if st.session_state.bg_points:
        aec_detected=remove_near(aec_detected,st.session_state.bg_points,r=max(6,circle_radius))
        hema_detected=remove_near(hema_detected,st.session_state.bg_points,r=max(6,circle_radius))
    # Merge mit manuell
    merged_aec = st.session_state.manual_aec.copy()
    for p in aec_detected:
        if not any(is_near(p,q,r=max(6,circle_radius)) for q in merged_aec):
            merged_aec.append(p)
    merged_hema = st.session_state.manual_hema.copy()
    for p in hema_detected:
        if not any(is_near(p,q,r=max(6,circle_radius)) for q in merged_hema):
            merged_hema.append(p)
    st.session_state.aec_points=dedup_points(merged_aec,min_dist=max(4,circle_radius//2))
    st.session_state.hema_points=dedup_points(merged_hema,min_dist=max(4,circle_radius//2))
    st.session_state.last_auto_run=0

# -------------------- Anzeige & CSV Export --------------------
all_aec = st.session_state.aec_points
all_hema = st.session_state.hema_points

st.markdown(f"### 🔢 Gesamt: AEC={len(all_aec)}, Hämatoxylin={len(all_hema)}")

df_list=[]
for x,y in all_aec: df_list.append({"X_display":x,"Y_display":y,"Type":"AEC"})
for x,y in all_hema: df_list.append({"X_display":x,"Y_display":y,"Type":"Hämatoxylin"})
if df_list:
    df=pd.DataFrame(df_list)
    df["X_original"]=(df["X_display"]/scale).round().astype("Int64")
    df["Y_original"]=(df["Y_display"]/scale).round().astype("Int64")
    csv=df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren",data=csv,file_name="zellkerne.csv",mime="text/csv")
