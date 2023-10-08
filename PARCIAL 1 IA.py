import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import skfuzzy as fuzz
import tkinter as tk
from tkinter import filedialog
import colorsys

def fuzzy_clustering(image_path, num_clusters):
    image = io.imread(image_path)
    
    if image.shape[2] == 4:
        image = image[:, :, :3]
    
    image_lab = color.rgb2lab(image)

    data = image_lab.reshape(-1, 3)

    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data.T, num_clusters, 2, error=0.005, maxiter=1000
    )

    cluster_membership = np.argmax(u, axis=0)
    clustered_data = cntr[cluster_membership]

    clustered_image_lab = clustered_data.reshape(image_lab.shape)
    clustered_image_rgb = color.lab2rgb(clustered_image_lab)

    return clustered_image_rgb

def open_and_segment():
    image_path = filedialog.askopenfilename()
    if image_path:
        num_clusters = int(cluster_entry.get())
        segmented_image = fuzzy_clustering(image_path, num_clusters)

        plt.imshow(segmented_image)
        plt.axis('off')
        plt.show()

        # Calcular colores presentes en la imagen
        unique_colors = np.unique(segmented_image.reshape(-1, 3), axis=0)

        # Crear ventana emergente para mostrar colores
        color_window = tk.Toplevel(root)
        color_window.title("Colores Presentes")

        for idx, color_rgb in enumerate(unique_colors):
            color_hex = '#{:02x}{:02x}{:02x}'.format(int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255))

            label_text = f"Color {idx + 1}:"
            color_label = tk.Label(color_window, text=label_text)
            color_label.pack()

            color_box = tk.Label(color_window, bg=color_hex, width=10, height=2)
            color_box.pack()

            color_hex_label = tk.Label(color_window, text=color_hex)
            color_hex_label.pack()

root = tk.Tk()
root.title("Fuzzy Clustering de Colores")

title_label = tk.Label(root, text="Segmentación de Colores con Clustering Difuso")
title_label.pack()

cluster_label = tk.Label(root, text="Número de Clusters:")
cluster_label.pack()

cluster_entry = tk.Entry(root)
cluster_entry.pack()

open_button = tk.Button(root, text="Abrir y Segmentar", command=open_and_segment)
open_button.pack()

root.mainloop()

