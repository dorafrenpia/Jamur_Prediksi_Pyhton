import joblib
import pandas as pd
import sys
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox


# Muat model dan encoders
try:
    model = joblib.load('decision_tree_model.pkl')
    feature_encoders = joblib.load('feature_encoders.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    ttk.Window().withdraw()
    messagebox.showerror("Error", f"Gagal memuat model/encoder:\n{e}")
    sys.exit()

# Daftar fitur
features = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]

# Keterangan masing-masing fitur
feature_descriptions = {
    'cap-shape': "Bentuk topi jamur",
    'cap-surface': "Permukaan topi",
    'cap-color': "Warna topi jamur",
    'bruises': "Ada memar atau tidak",
    'odor': "Bau jamur",
    'gill-attachment': "Tipe keterikatan insang",
    'gill-spacing': "Jarak antar insang",
    'gill-size': "Ukuran insang",
    'gill-color': "Warna insang",
    'stalk-shape': "Bentuk batang",
    'stalk-root': "Jenis akar batang",
    'stalk-surface-above-ring': "Permukaan batang di atas cincin",
    'stalk-surface-below-ring': "Permukaan batang di bawah cincin",
    'stalk-color-above-ring': "Warna batang di atas cincin",
    'stalk-color-below-ring': "Warna batang di bawah cincin",
    'veil-type': "Tipe selubung",
    'veil-color': "Warna selubung",
    'ring-number': "Jumlah cincin",
    'ring-type': "Tipe cincin",
    'spore-print-color': "Warna cetakan spora",
    'population': "Jumlah populasi di habitat",
    'habitat': "Jenis habitat"
}

# Fungsi prediksi
def predict():
    try:
        input_data = {}
        for feature in features:
            val = vars_inputs[feature].get().strip()
            if not val:
                raise ValueError(f"Field '{feature}' tidak boleh kosong.")
            
            # --- Tambahkan bagian ini ---
            if "(" in val and ")" in val:
                code = val.split("(")[-1].replace(")", "").strip()
            else:
                code = val.strip()
            # --- Sampai sini ---

            if feature in feature_encoders:
                enc = feature_encoders[feature]
                try:
                    input_data[feature] = enc.transform([code])[0]
                except ValueError:
                    raise ValueError(
                        f"'{code}' tidak dikenali untuk '{feature}'.\nPilihan: {list(enc.classes_)}"
                    )
            else:
                input_data[feature] = code

        X = pd.DataFrame([input_data])
        pred_code = model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_code])[0]
        result = (
            "🍄✅ Jamur ini bisa DIMAKAN (Edible)!"
            if pred_label == 'e'
            else "☠️❌ Jamur ini BERACUN (Poisonous)!"
        )
        messagebox.showinfo("Hasil Prediksi", result)
    except Exception as e:
        messagebox.showerror("Error Prediksi", str(e))
        
        # Fungsi untuk mengganti tema
def change_theme(theme_name):
    app.style.theme_use(theme_name)


# --- Setup GUI ---
app = ttk.Window(themename="darkly")  # Menggunakan ttk.Window untuk ttkbootstrap
app.title("Prediksi Jamur Edible atau Poisonous")
app.geometry('850x800')  # Lebarkan supaya keterangan cukup
from PIL import Image, ImageTk

try:
    # Ganti dengan path ikon .ico
    app.iconbitmap(r"C:\Users\Frendy\Desktop\Kuliah\DataScience2\Aplikasi\logohanz.ico")  # Menggunakan format .ico
except Exception as e:
    print(f"Gagal memuat logo untuk tab: {e}")

# Header
header = ttk.Label(
    app,
    text="🍄 Prediksi Jamur Edible atau Poisonous ☠️",
    font=("Segoe UI", 20, "bold"),
    bootstyle=PRIMARY
)
header.pack(pady=20)
# Frame untuk tombol ganti tema (DIBAWAH HEADER)
theme_frame = ttk.Frame(app)
theme_frame.pack(pady=10)

# Menambahkan tema baru: minty, sandstone
themes = ['darkly', 'flatly', 'solar', 'cyborg', 'minty', 'sandstone']
for theme in themes:
    btn = ttk.Button(
        theme_frame,
        text=theme.capitalize(),
        command=lambda t=theme: change_theme(t),
        bootstyle=INFO,
        width=10
    )
    btn.pack(side='left', padx=5)


# Frame scrollable untuk input
container = ttk.Frame(app)
canvas = ttk.Canvas(container)

scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview, bootstyle=SECONDARY)
input_frame = ttk.Frame(canvas)

canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side='right', fill='y')
canvas.pack(side='left', fill='both', expand=True)
container.pack(fill='both', expand=True, padx=10, pady=10)

canvas.create_window((0, 0), window=input_frame, anchor='nw')
input_frame.bind(
    '<Configure>',
    lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
)
feature_choices = {
    'cap-shape': {
        'Bell': 'b',
        'Conical': 'c',
        'Flat': 'f',
        'Knobbed': 'k',
        'Sunken': 's',
        'Convex': 'x'
    },
    'cap-surface': {
        'Fibrous': 'f',
        'Grooves': 'g',
        'Scaly': 'y',
        'Smooth': 's'
    },
    'cap-color': {
        'Brown': 'n',
        'Buff': 'b',
        'Cinnamon': 'c',
        'Gray': 'g',
        'Green': 'r',
        'Pink': 'p',
        'Purple': 'u',
        'Red': 'e',
        'White': 'w',
        'Yellow': 'y'
    },
    'bruises': {
        'Yes': 't',
        'No': 'f'
    },
    'odor': {
        'Almond': 'a',
        'Anise': 'l',
        'Creosote': 'c',
        'Fishy': 'y',
        'Foul': 'f',
        'Musty': 'm',
        'None': 'n',
        'Pungent': 'p',
        'Spicy': 's'
    },
    'gill-attachment': {
        'Attached': 'a',
        
        'Free': 'f',
        
    },
    'gill-spacing': {
        'Close': 'c',
        'Crowded': 'w',
    },
    'gill-size': {
        'Broad': 'b',
        'Narrow': 'n'
    },
    'gill-color': {
        'Black': 'k',
        'Brown': 'n',
        'Buff': 'b',
        'Chocolate': 'h',
        'Gray': 'g',
        'Green': 'r',
        'Orange': 'o',
        'Pink': 'p',
        'Purple': 'u',
        'Red': 'e',
        'White': 'w',
        'Yellow': 'y'
    },
    'stalk-shape': {
        'Enlarging': 'e',
        'Tapering': 't'
    },
    'stalk-root': {
        'Bulbous': 'b',
        'Club': 'c',
        'Equal': 'e',
        'Rooted': 'r',
    },
    'stalk-surface-above-ring': {
        'Fibrous': 'f',
        'Scaly': 'y',
        'Smooth': 's',
        'Silky': 'k'  # Ditambahkan
    },
    'stalk-surface-below-ring': {
        'Fibrous': 'f',
        'Scaly': 'y',
        'Smooth': 's',
        'Silky': 'k'  # Ditambahkan
    },
    'stalk-color-above-ring': {
        'Brown': 'n',
        'Buff': 'b',
        'Cinnamon': 'c',
        'Gray': 'g',
        'Green': 'r',
        'Orange': 'o',
        'Pink': 'p',
        'Red': 'e',
        'White': 'w',
        'Yellow': 'y'
    },
    'stalk-color-below-ring': {
        'Brown': 'n',
        'Buff': 'b',
        'Cinnamon': 'c',
        'Gray': 'g',
        'Orange': 'o',
        'Pink': 'p',
        'Red': 'e',
        'White': 'w',
        'Yellow': 'y'
    },
    'veil-type': {
        'Partial': 'p'
    },
    'veil-color': {
        'Brown': 'n',
        'Orange': 'o',
        'White': 'w',
        'Yellow': 'y'
    },
    'ring-number': {
        'None': 'n',
        'One': 'o',
        'Two': 't'
    },
    'ring-type': {
        'Evanescent': 'e',
        'Flaring': 'f',
        'Large': 'l',
        'None': 'n',
        'Pendant': 'p',
        
        
    },
    'spore-print-color': {
        'Black': 'k',
        'Brown': 'n',
        'Buff': 'b',
        'Chocolate': 'h',
        'Green': 'r',
        'Orange': 'o',
        'Purple': 'u',
        'White': 'w',
        'Yellow': 'y'
    },
    'population': {
        'Abundant': 'a',
        'Clustered': 'c',
        'Numerous': 'n',
        'Scattered': 's',
        'Several': 'v',
        'Solitary': 'y'
    },
    'habitat': {
        'Grasses': 'g',
        'Leaves': 'l',
        'Meadows': 'm',
        'Paths': 'p',
        'Urban': 'u',
        'Waste': 'w',  # Ditambahkan
        'Woods': 'd'
    }
}


# Buat input fields
vars_inputs = {}
for idx, feature in enumerate(features):
    # Label nama fitur
    label = ttk.Label(input_frame, text=feature.replace("-", " ").title(), font=("Segoe UI", 11))
    label.grid(row=idx, column=0, sticky='w', padx=5, pady=6)

    var = ttk.StringVar()

    if feature in feature_choices:
        choices = [f"{name} ({code})" for name, code in feature_choices[feature].items()]

        entry = ttk.Combobox(
            input_frame,
            textvariable=var,
            values=choices,
            state='readonly',
            font=("Segoe UI", 10),
            bootstyle=SUCCESS
        )
        var.set(choices[0])  # Set default choice
    else:
        entry = ttk.Entry(
            input_frame,
            textvariable=var,
            font=("Segoe UI", 10),
            bootstyle=SUCCESS
        )

    entry.grid(row=idx, column=1, sticky='ew', padx=5, pady=6)
    vars_inputs[feature] = var
    input_frame.columnconfigure(1, weight=1)

    # Label keterangan samping input
    desc = feature_descriptions.get(feature, "Keterangan tidak tersedia")
    description_label = ttk.Label(input_frame, text=desc, font=("Segoe UI", 9), bootstyle=SECONDARY)
    description_label.grid(row=idx, column=2, sticky='w', padx=10, pady=6)

# Tombol prediksi
predict_btn = ttk.Button(
    app,
    text='🔮 Prediksi Sekarang!',
    command=predict,
    bootstyle=SUCCESS,
    width=20
)
predict_btn.pack(pady=20)

# Footer
footer = ttk.Label(
    app,
    text="Frendy Hansung - 2355202038",
    font=("Segoe UI", 9),
    bootstyle=SECONDARY
)
footer.pack(side='bottom', pady=10)

app.mainloop()
