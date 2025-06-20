import gradio as gr
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageEnhance
import torch
import clip
import cv2
import wikipedia

# === Setup ===
wikipedia.set_lang("en")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load models
unet_model = load_model("C:\\Users\\vinay\\OneDrive\\Desktop\\major_p\\unet_best_model.h5", compile=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# === Monument Labels ===
monument_names = [
    'Ajanta Caves', 'Charar-E- Sharif', 'Chhota Imambara', 'Ellora Caves',
    'Fatehpur Sikri', 'Gateway of India', 'Humayun\'s Tomb', 'India Gate',
    'Khajuraho', 'Sun Temple Konark', 'Alai Darwaza', 'Alai Minar',
    'Basilica of Bom Jesus', 'Charminar', 'Golden Temple', 'Hawa Mahal',
    'Iron Pillar', 'Jamali Kamali Tomb', 'Lotus Temple', 'Mysore Palace',
    'Qutub Minar', 'Taj Mahal', 'Tanjavur Temple', 'Victoria Memorial'
]

# === Function: Restore and Identify Monuments ===
def gradio_restore_and_identify(damaged_image):
    try:
        img_size = (256, 256)
        base = os.path.basename(damaged_image)
        restored_path = os.path.join("outputs", f"restored_{base}")

        # Restore using U-Net
        img = load_img(damaged_image, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        restored = unet_model.predict(img_array)[0]
        restored = (restored * 255).astype(np.uint8)
        Image.fromarray(restored).save(restored_path)

        # Identify using CLIP
        image = preprocess(Image.open(restored_path)).unsqueeze(0).to(device)
        text = clip.tokenize(monument_names).to(device)
        with torch.no_grad():
            logits_per_image, _ = clip_model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        best_idx = probs[0].argmax()
        predicted = monument_names[best_idx]
        confidence = probs[0][best_idx]

        # Get Wikipedia Description
        try:
            description = wikipedia.summary(predicted, sentences=2)
        except:
            description = "No Wikipedia description found."

        # Plot Result
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(plt.imread(damaged_image))
        axs[0].set_title("Damaged Image")
        axs[0].axis('off')
        axs[1].imshow(restored)
        axs[1].set_title(f"Restored → {predicted}")
        axs[1].axis('off')
        fig_path = os.path.join("outputs", "comparison_plot.png")
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

        return fig_path, f"{predicted} (Confidence: {confidence:.2%})", description

    except Exception as e:
        return None, "Error", f"Error: {str(e)}"

# === Function: Enhance Manuscript Images ===
def enhance_image(image_pil):
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.7)
    enhancer = ImageEnhance.Color(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Brightness(enhanced_pil)
    enhanced_pil = enhancer.enhance(1.1)

    enhanced_np = np.array(enhanced_pil)
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_image = cv2.transform(enhanced_np, sepia_matrix)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    final = cv2.addWeighted(enhanced_np, 0.7, sepia_image, 0.3, 0)
    return Image.fromarray(final)

def process_multiple(images):
    results = []
    for img in images:
        enhanced = enhance_image(img)
        results.append((img, enhanced))
    return results

def handle_manuscript_files(files):
    images = [Image.open(f).convert("RGB") for f in files]
    results = process_multiple(images)
    gallery_images = [img for pair in results for img in pair]
    return gallery_images

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("# 🏛️ Monument & 📜 Manuscript Image Processor")

    with gr.Tabs():
        with gr.TabItem("🛠️ Monument Restoration"):
            gr.Markdown("Upload a damaged Indian monument image. It will be restored and identified using a U-Net + CLIP pipeline, along with a Wikipedia description.")
            monument_input = gr.Image(type="filepath", label="Upload Damaged Monument Image")
            monument_output_img = gr.Image(type="filepath", label="Restoration Result")
            monument_output_text = gr.Textbox(label="Predicted Monument")
            monument_output_desc = gr.Textbox(label="Wikipedia Description", lines=4)
            monument_input.change(
                gradio_restore_and_identify,
                inputs=monument_input,
                outputs=[monument_output_img, monument_output_text, monument_output_desc]
            )

        with gr.TabItem("📜 Manuscript Enhancement"):
            gr.Markdown("Upload one or more manuscript images to enhance their readability and visual quality.")
            manuscript_input = gr.File(type="filepath", file_types=['.jpg', '.png', '.jpeg'], file_count="multiple", label="Upload Images")
            manuscript_gallery = gr.Gallery(label="Original and Enhanced Pairs", columns=2)
            manuscript_input.change(
                handle_manuscript_files,
                inputs=manuscript_input,
                outputs=manuscript_gallery
            )

demo.launch(debug=True)
# run this using python app_gradio.py
