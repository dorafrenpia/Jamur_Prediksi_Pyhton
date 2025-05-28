import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
import pickle

# Load MobileNetV2 feature extractor
feature_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(128,128,3))

# Load model Decision Tree dari file .pkl
with open('mushroom_dtc_model.pkl', 'rb') as f:
    clf = pickle.load(f)

class MushroomApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mushroom Classifier 🍄")
        self.root.geometry("400x500")

        self.label = tk.Label(root, text="Unggah gambar jamur", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn = tk.Button(root, text="Pilih Gambar", command=self.load_image)
        self.btn.pack(pady=10)

        self.canvas = tk.Label(root)
        self.canvas.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 16, "bold"))
        self.result_label.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            try:
                img = Image.open(file_path).convert("RGB")
                img_resized_display = img.resize((200, 200))
                self.tk_img = ImageTk.PhotoImage(img_resized_display)
                self.canvas.config(image=self.tk_img)

                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = preprocess_input(img_array)
                features = feature_model.predict(np.expand_dims(img_array, axis=0), verbose=0)
                prediction = clf.predict(features)
                label = "✅ Edible" if prediction[0] == 0 else "☠️ Poisonous"
                self.result_label.config(text=f"Hasil: {label}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memproses gambar: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MushroomApp(root)
    root.mainloop()
