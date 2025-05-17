import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from image_processing import (
    load_image, display_1d_rgb_histograms, compute_2d_histogram, compute_3d_histogram_rgb,
    search_similar_images, euclidean_distance, normalize_histogram, analyze_rgb_distribution,
    display_1d_hsv_histograms, display_2d_hsv_histograms, display_3d_histogram_hsv,
    rgb_2d_all_planes_histogram_for_comparison
)
import numpy as np
import os

def show_image(img_path, panel):
    try:
        img = Image.open(img_path)
        orig_w, orig_h = img.size
        max_w, max_h = 400, 400
        ratio = min(max_w / orig_w, max_h / orig_h, 1.0)
        new_size = (int(orig_w * ratio), int(orig_h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk, width=new_size[0], height=new_size[1])
        panel.image = img_tk
    except Exception as e:
        messagebox.showerror("Error", f"Error displaying image: {e}")

def gui():
    root = tk.Tk()
    root.title("Color-based Image Retrieval")
    
    img_panel = tk.Label(root)
    img_panel.pack()
    
    selected_img = {'path': None, 'rgb': None}
    
    results_frame = tk.Frame(root)
    results_frame.pack(pady=10)
    result_labels = []
    result_imgs = []
    
    def clear_results():
        for lbl in result_labels:
            lbl.destroy()
        for img in result_imgs:
            img.destroy()
        result_labels.clear()
        result_imgs.clear()

    def select_image():
        initial_dir = os.path.abspath("tests/test_images")
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("JPEG files", "*.jpeg"),
                ("Bitmap files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            selected_img['path'] = file_path
            selected_img['rgb'] = load_image(file_path)
            show_image(file_path, img_panel)
            clear_results()
    
    def show_1d_hist():
        if selected_img['rgb'] is not None:
            display_1d_rgb_histograms(selected_img['rgb'], title="1D RGB Histogram")
    
    def show_2d_hist():
        if selected_img['rgb'] is not None and selected_img['path'] is not None:
            analyze_rgb_distribution(selected_img['path'])
    
    def show_3d_hist():
        if selected_img['rgb'] is not None:
            from mpl_toolkits.mplot3d import Axes3D
            hist3d = compute_3d_histogram_rgb(selected_img['rgb'], bins=8)
            hist_norm = hist3d / np.max(hist3d) if np.max(hist3d) > 0 else hist3d
            x, y, z = np.where(hist_norm > 0.01)
            values = hist_norm[x, y, z]
            colors = np.stack([x, y, z], axis=1) / 7.0
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c=colors, s=values * 200, alpha=0.6)
            ax.set_xlabel('R')
            ax.set_ylabel('G')
            ax.set_zlabel('B')
            ax.set_title("3D RGB Histogram")
            plt.tight_layout()
            plt.show()
    
    def show_1d_hist_hsv():
        if selected_img['rgb'] is not None:
            display_1d_hsv_histograms(selected_img['rgb'], title="1D HSV Histogram")

    def show_2d_hist_hsv():
        if selected_img['rgb'] is not None:
            display_2d_hsv_histograms(selected_img['rgb'], title="2D HSV Histograms")

    def show_3d_hist_hsv():
        if selected_img['rgb'] is not None:
            display_3d_histogram_hsv(selected_img['rgb'], title="3D HSV Histogram")
    
    # --- BUTTON FRAMES ---
    btn_frame_top = tk.Frame(root)
    btn_frame_top.pack(pady=10)
    tk.Button(btn_frame_top, text="Select Image", command=select_image).pack(side=tk.LEFT, padx=5)

    btn_frame_rgb = tk.LabelFrame(root, text="RGB Visualization", padx=10, pady=5)
    btn_frame_rgb.pack(pady=5, fill="x")
    tk.Button(btn_frame_rgb, text="1D RGB Histogram", command=show_1d_hist).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame_rgb, text="2D RGB Histogram", command=show_2d_hist).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame_rgb, text="3D RGB Histogram", command=show_3d_hist).pack(side=tk.LEFT, padx=5)

    btn_frame_hsv = tk.LabelFrame(root, text="HSV Visualization", padx=10, pady=5)
    btn_frame_hsv.pack(pady=5, fill="x")
    tk.Button(btn_frame_hsv, text="1D HSV Histogram", command=show_1d_hist_hsv).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame_hsv, text="2D HSV Histogram", command=show_2d_hist_hsv).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame_hsv, text="3D HSV Histogram", command=show_3d_hist_hsv).pack(side=tk.LEFT, padx=5)

    # --- SEARCH SECTION ---
    search_frame = tk.LabelFrame(root, text="Search Similar Images", padx=10, pady=5)
    search_frame.pack(pady=10, fill="x")
    search_types = [
        "1D RGB", "2D RGB", "3D RGB",
        "1D HSV", "2D HSV", "3D HSV",
        "All RGB", "All HSV", "All Combined"
    ]
    search_var = tk.StringVar(value=search_types[0])
    tk.Label(search_frame, text="Histogram Type:").pack(side=tk.LEFT, padx=5)
    search_menu = tk.OptionMenu(search_frame, search_var, *search_types)
    search_menu.pack(side=tk.LEFT, padx=5)
    tk.Button(search_frame, text="Search Similar Images", command=lambda: search_similar_hist(search_var.get())).pack(side=tk.LEFT, padx=5)

    def search_similar_hist(hist_type):
        clear_results()
        if selected_img['path'] is not None:
            test_dir = os.path.abspath("tests/test_images")
            image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            image_files = [f for f in image_files if os.path.abspath(f) != os.path.abspath(selected_img['path'])]
            from image_processing import (
                compute_1d_histogram, compute_2d_histogram, compute_3d_histogram_rgb,
                convert_to_hsv, compute_3d_histogram_hsv, compute_2d_histogram_hsv
            )
            def rgb_1d(image):
                return np.concatenate([
                    compute_1d_histogram(image, 0),
                    compute_1d_histogram(image, 1),
                    compute_1d_histogram(image, 2)
                ])
            def rgb_2d(image):
                return rgb_2d_all_planes_histogram_for_comparison(image, bins=32)
            def rgb_3d(image):
                return compute_3d_histogram_rgb(image, bins=8).flatten()
            def hsv_1d(image):
                hsv = convert_to_hsv(image)
                return np.concatenate([
                    compute_1d_histogram(hsv, 0),
                    compute_1d_histogram(hsv, 1),
                    compute_1d_histogram(hsv, 2)
                ])
            def hsv_2d(image):
                hsv = convert_to_hsv(image)
                return np.concatenate([
                    compute_2d_histogram_hsv(hsv, 0, 1, 180, 256).flatten(),
                    compute_2d_histogram_hsv(hsv, 0, 2, 180, 256).flatten(),
                    compute_2d_histogram_hsv(hsv, 1, 2, 256, 256).flatten()
                ])
            def hsv_3d(image):
                return compute_3d_histogram_hsv(image, bins=(18, 16, 16)).flatten()
            def all_rgb(image):
                return np.concatenate([rgb_1d(image), rgb_2d(image), rgb_3d(image)])
            def all_hsv(image):
                return np.concatenate([hsv_1d(image), hsv_2d(image), hsv_3d(image)])
            def all_combined(image):
                return np.concatenate([rgb_1d(image), rgb_2d(image), rgb_3d(image), hsv_1d(image), hsv_2d(image), hsv_3d(image)])
            hist_func_map = {
                "1D RGB": rgb_1d,
                "2D RGB": rgb_2d,
                "3D RGB": rgb_3d,
                "1D HSV": hsv_1d,
                "2D HSV": hsv_2d,
                "3D HSV": hsv_3d,
                "All RGB": all_rgb,
                "All HSV": all_hsv,
                "All Combined": all_combined
            }
            hist_func = hist_func_map.get(hist_type, rgb_1d)
            results = search_similar_images(
                query_image_path=selected_img['path'],
                image_paths=image_files,
                histogram_func=hist_func,
                distance_func=euclidean_distance,
                top_n=3,
                normalize=True
            )
            for idx, (path, dist) in enumerate(results):
                try:
                    img = Image.open(path)
                    img = img.resize((100, 100), Image.LANCZOS)
                    img_tk = ImageTk.PhotoImage(img)
                    img_label = tk.Label(results_frame, image=img_tk)
                    img_label.image = img_tk
                    img_label.grid(row=0, column=idx, padx=5)
                    result_imgs.append(img_label)
                    text = f"{os.path.basename(path)}\ndistance: {dist:.5f}"
                    lbl = tk.Label(results_frame, text=text)
                    lbl.grid(row=1, column=idx, padx=5)
                    result_labels.append(lbl)
                except Exception as e:
                    lbl = tk.Label(results_frame, text=f"Display error: {e}")
                    lbl.grid(row=1, column=idx, padx=5)
                    result_labels.append(lbl)

    root.mainloop()

if __name__ == "__main__":
    gui() 