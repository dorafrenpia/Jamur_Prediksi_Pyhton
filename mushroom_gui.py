import numpy as np
import ttkbootstrap as ttk
from ttkbootstrap.constants import PRIMARY, SUCCESS, DANGER
from tkinter import filedialog, messagebox
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
import pickle

# Load model feature extractor dan decision tree
feature_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(128, 128, 3))

with open('mushroom_dtc_model.pkl', 'rb') as f:
    clf = pickle.load(f)

class MushroomApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🍄 Mushroom Classifier - Aman atau Beracun?")
        self.root.geometry("450x600")
        self.root.resizable(False, False)

        style = ttk.Style("litera")

        # Judul aplikasi
        self.title_label = ttk.Label(
            root,
            text="🌿 Klasifikasi Jamur",
            font=("Segoe UI", 20, "bold"),
            anchor="center"
        )
        self.title_label.pack(pady=(20, 10))

        # Instruksi
        self.label = ttk.Label(
            root,
            text="Silakan unggah gambar jamur (jpg/png):",
            font=("Segoe UI", 12)
        )
        self.label.pack(pady=10)

        # Tombol pilih gambar
        self.btn = ttk.Button(
            root,
            text="📁 Pilih Gambar",
            command=self.load_image,
            bootstyle=PRIMARY
        )
        self.btn.pack(pady=10)

        # Tempat gambar ditampilkan
        self.canvas = ttk.Label(root)
        self.canvas.pack(pady=10)

        # Label hasil prediksi
        self.result_label = ttk.Label(
            root,
            text="",
            font=("Segoe UI", 16, "bold")
        )
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            try:
                # Tampilkan gambar di canvas
                img = Image.open(file_path).convert("RGB")
                img_resized_display = img.resize((250, 250))
                self.tk_img = ImageTk.PhotoImage(img_resized_display)
                self.canvas.config(image=self.tk_img)

                # Proses untuk prediksi
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = preprocess_input(img_array)
                features = feature_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
                prediction = clf.predict(features)

                if prediction[0] == 0:
                    label = "✅ Aman dimakan (Edible)"
                    color = SUCCESS
                else:
                    label = "☠️ Beracun (Poisonous)"
                    color = DANGER

                self.result_label.config(text=f"Hasil: {label}", bootstyle=color)

            except Exception as e:
                messagebox.showerror("Error", f"Gagal memproses gambar: {str(e)}")

if __name__ == "__main__":
    root = ttk.Window(themename="litera")  # bisa juga "cosmo", "morph", dll.
    app = MushroomApp(root)
    root.mainloop()
