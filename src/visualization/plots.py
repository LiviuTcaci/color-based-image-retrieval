import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import cv2
import os

def plot_2d_histogram(hist: np.ndarray, 
                     title: str,
                     xlabel: str,
                     ylabel: str,
                     figsize: Tuple[int, int] = (8, 6),
                     cmap: str = 'viridis',
                     save_path: Optional[str] = None) -> None:
    """
    Vizualizează o histogramă 2D folosind matplotlib.
    
    Args:
        hist: Histograma 2D de vizualizat
        title: Titlul graficului
        xlabel: Eticheta pentru axa X
        ylabel: Eticheta pentru axa Y
        figsize: Dimensiunea figurii (width, height)
        cmap: Schema de culori pentru vizualizare
        save_path: Calea unde să se salveze imaginea (opțional)
    """
    plt.figure(figsize=figsize)
    plt.imshow(hist, cmap=cmap, origin='lower')
    plt.colorbar(label='Frecvență')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_2d_rgb_histograms(hist_rg: np.ndarray,
                               hist_rb: np.ndarray,
                               hist_gb: np.ndarray,
                               figsize: Tuple[int, int] = (15, 5),
                               save_path: Optional[str] = None) -> None:
    """
    Vizualizează toate cele trei histograme 2D RGB (RG, RB, GB) într-o singură figură.
    
    Args:
        hist_rg: Histograma 2D pentru canalele Red-Green
        hist_rb: Histograma 2D pentru canalele Red-Blue
        hist_gb: Histograma 2D pentru canalele Green-Blue
        figsize: Dimensiunea figurii (width, height)
        save_path: Calea unde să se salveze imaginea (opțional)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # RG Histogram
    im0 = axes[0].imshow(hist_rg, cmap='Reds', origin='lower')
    axes[0].set_title('Red-Green Histogram')
    axes[0].set_xlabel('Green')
    axes[0].set_ylabel('Red')
    plt.colorbar(im0, ax=axes[0], label='Frecvență')
    
    # RB Histogram
    im1 = axes[1].imshow(hist_rb, cmap='Blues', origin='lower')
    axes[1].set_title('Red-Blue Histogram')
    axes[1].set_xlabel('Blue')
    axes[1].set_ylabel('Red')
    plt.colorbar(im1, ax=axes[1], label='Frecvență')
    
    # GB Histogram
    im2 = axes[2].imshow(hist_gb, cmap='Greens', origin='lower')
    axes[2].set_title('Green-Blue Histogram')
    axes[2].set_xlabel('Blue')
    axes[2].set_ylabel('Green')
    plt.colorbar(im2, ax=axes[2], label='Frecvență')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def create_2d_histogram_visualization(hist_rg: np.ndarray,
                                    hist_rb: np.ndarray,
                                    hist_gb: np.ndarray,
                                    output_size: Tuple[int, int] = (800, 200),
                                    save_path: Optional[str] = None) -> np.ndarray:
    """
    Creează o imagine combinată cu toate cele trei histograme 2D RGB.
    
    Args:
        hist_rg: Histograma 2D pentru canalele Red-Green
        hist_rb: Histograma 2D pentru canalele Red-Blue
        hist_gb: Histograma 2D pentru canalele Green-Blue
        output_size: Dimensiunea imaginii de ieșire (width, height)
        save_path: Calea unde să se salveze imaginea (opțional)
        
    Returns:
        np.ndarray: Imaginea combinată cu histogramele
    """
    # Normalizăm histogramele pentru vizualizare
    def normalize_hist(hist):
        hist_norm = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)
        return hist_norm.astype(np.uint8)
    
    # Convertim histogramele în imagini color
    hist_rg_vis = cv2.applyColorMap(normalize_hist(hist_rg), cv2.COLORMAP_JET)
    hist_rb_vis = cv2.applyColorMap(normalize_hist(hist_rb), cv2.COLORMAP_JET)
    hist_gb_vis = cv2.applyColorMap(normalize_hist(hist_gb), cv2.COLORMAP_JET)
    
    # Redimensionăm histogramele pentru a se potrivi în imaginea finală
    target_height = output_size[1]
    target_width = output_size[0] // 3
    
    hist_rg_vis = cv2.resize(hist_rg_vis, (target_width, target_height))
    hist_rb_vis = cv2.resize(hist_rb_vis, (target_width, target_height))
    hist_gb_vis = cv2.resize(hist_gb_vis, (target_width, target_height))
    
    # Combinăm histogramele
    combined = np.hstack((hist_rg_vis, hist_rb_vis, hist_gb_vis))
    
    # Adăugăm text pentru etichete
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    cv2.putText(combined, 'RG', (target_width//2 - 20, 30), font, font_scale, color, thickness)
    cv2.putText(combined, 'RB', (target_width + target_width//2 - 20, 30), font, font_scale, color, thickness)
    cv2.putText(combined, 'GB', (2*target_width + target_width//2 - 20, 30), font, font_scale, color, thickness)
    
    if save_path:
        cv2.imwrite(save_path, combined)
    
    return combined

def visualize_3d_rgb_histogram_planes(hist_3d: np.ndarray, save_dir: str, prefix: str = "hist_3d_plan", plane_axis: str = 'B') -> None:
    """
    Vizualizează histograma 3D RGB ca un set de plane 2D (implicit, pentru fiecare valoare a canalului Blue).
    Salvează fiecare plan ca imagine separată.

    Args:
        hist_3d: Histograma 3D RGB (shape: bins x bins x bins)
        save_dir: Directorul unde se salvează imaginile
        prefix: Prefixul fișierelor salvate
        plane_axis: Canalul fixat pentru fiecare plan ('R', 'G', 'B')
    """
    os.makedirs(save_dir, exist_ok=True)
    bins = hist_3d.shape[0]
    axis_map = {'R': 0, 'G': 1, 'B': 2}
    axis = axis_map.get(plane_axis.upper(), 2)
    
    for i in range(bins):
        if axis == 0:
            plane = hist_3d[i, :, :]
            title = f"Green-Blue plane (Red={i})"
            xlabel, ylabel = 'Blue', 'Green'
        elif axis == 1:
            plane = hist_3d[:, i, :]
            title = f"Red-Blue plane (Green={i})"
            xlabel, ylabel = 'Blue', 'Red'
        else:
            plane = hist_3d[:, :, i]
            title = f"Red-Green plane (Blue={i})"
            xlabel, ylabel = 'Green', 'Red'
        
        plt.figure(figsize=(5, 4))
        plt.imshow(plane, cmap='jet', origin='lower')
        plt.colorbar(label='Frecvență')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        out_path = os.path.join(save_dir, f"{prefix}_{i}.jpg")
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close()

def visualize_hsv_histograms(hist_h: np.ndarray, hist_s: np.ndarray, hist_v: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Vizualizează histogramele HSV:
      - Hue: diagramă circulară (polar)
      - Saturation: bar plot pe scală de gri
      - Value: bar plot pe scală de gri
    Toate în aceeași figură, cu titluri și etichete clare.

    Args:
        hist_h: Histograma pentru canalul Hue
        hist_s: Histograma pentru canalul Saturation
        hist_v: Histograma pentru canalul Value
        save_path: Calea unde să se salveze figura (opțional)
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Hue - polar plot
    ax1 = plt.subplot(1, 3, 1, polar=True)
    bins_h = len(hist_h)
    theta = np.linspace(0.0, 2 * np.pi, bins_h, endpoint=False)
    radii = hist_h / (hist_h.max() if hist_h.max() > 0 else 1)
    bars = ax1.bar(theta, radii, width=2*np.pi/bins_h, bottom=0.0, color=plt.cm.hsv(theta/(2*np.pi)), edgecolor='k')
    ax1.set_title('Histograma Hue (circulară)', va='bottom')
    ax1.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
    ax1.set_yticklabels([])
    
    # Saturation - bar plot (grayscale)
    ax2 = plt.subplot(1, 3, 2)
    bins_s = len(hist_s)
    ax2.bar(np.arange(bins_s), hist_s, color='gray', edgecolor='black')
    ax2.set_title('Histograma Saturation')
    ax2.set_xlabel('Saturație')
    ax2.set_ylabel('Frecvență')
    
    # Value - bar plot (grayscale)
    ax3 = plt.subplot(1, 3, 3)
    bins_v = len(hist_v)
    ax3.bar(np.arange(bins_v), hist_v, color='gray', edgecolor='black')
    ax3.set_title('Histograma Value')
    ax3.set_xlabel('Valoare')
    ax3.set_ylabel('Frecvență')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
