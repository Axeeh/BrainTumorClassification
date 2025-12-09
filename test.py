import numpy as np
from skimage import io as skio
import os
import sys
import io

# Fix Unicode per Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_if_grayscale_rgb(image_path):
    """Verifica se un'immagine è caricata come RGB ma ha tutti i canali identici"""
    img = skio.imread(image_path)

    info = {
        'file': os.path.basename(image_path),
        'shape': img.shape,
        'dtype': img.dtype,
        'is_rgb_3d': len(img.shape) == 3 and img.shape[2] == 3,
        'all_channels_identical': False
    }

    if info['is_rgb_3d']:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        channels_identical = np.allclose(r, g) and np.allclose(g, b)
        info['all_channels_identical'] = channels_identical

    return info


def check_all_images_in_dataset(dataset_path):
    """Controlla tutte le immagini in un dataset - SOLO ANALISI"""
    results = {
        'total': 0,
        'grayscale_actual': 0,
        'rgb_necessary': 0,
        'rgb_unnecessary': 0,
        'images_info': []
    }

    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            img_path = os.path.join(dataset_path, filename)
            try:
                info = check_if_grayscale_rgb(img_path)
                results['images_info'].append(info)
                results['total'] += 1

                if info['is_rgb_3d']:
                    if info['all_channels_identical']:
                        results['rgb_unnecessary'] += 1
                    else:
                        results['rgb_necessary'] += 1
                else:
                    results['grayscale_actual'] += 1

            except Exception as e:
                print(f"Errore leggendo {filename}: {e}")

    return results


# ===== MAIN - SOLO ANALISI =====

if __name__ == "__main__":
    dataset_path = "Dataset/test"

    print("\n" + "="*70)
    print("ANALISI IMMAGINI RGB - VERIFICA SE GRAYSCALE INUTILE")
    print("="*70 + "\n")

    # Controlla tutte le immagini
    results = check_all_images_in_dataset(dataset_path)

    print(f"STATISTICHE:")
    print(f"  Totale immagini: {results['total']}")
    print(f"  Grayscale vere (H, W): {results['grayscale_actual']}")
    print(f"  RGB con canali diversi: {results['rgb_necessary']}")
    print(f"  RGB inutile (canali identici): {results['rgb_unnecessary']}")

    if results['rgb_unnecessary'] > 0:
        print(f"\n[ATTENZIONE] {results['rgb_unnecessary']} immagini hanno RGB inutile!")
        print("Se lo desideri, puoi convertirle a grayscale in seguito.")
        
        print("\nPrimi 5 esempi di RGB inutile:")
        count = 0
        for info in results['images_info']:
            if info['all_channels_identical']:
                print(f"  - {info['file']}: shape={info['shape']}")
                count += 1
                if count >= 5:
                    break
    else:
        print(f"\n[OK] Tutte le immagini RGB hanno canali diversi.")

    print("\n" + "="*70)
