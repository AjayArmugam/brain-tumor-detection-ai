import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time

model = tf.keras.models.load_model("brain_tumor_model.h5")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_and_explain(img):

    # Simulate loading (for spinner UX)
    time.sleep(1)

    img = img.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))
    label = class_names[class_idx]

    # 🎯 Badge + color logic
    if label == "notumor":
        badge = "🟢 No Tumor Detected"
        color = "green"
    else:
        badge = "🔴 Tumor Detected"
        color = "red"

    result_text = f"<h2 style='color:{color}'>{badge}</h2>"

    # Grad-CAM
    base_model = model.get_layer("efficientnetb0")
    last_conv_layer = base_model.get_layer("top_conv")

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Overlay
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    output_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

    return result_text, label, confidence, output_img

# 🎨 UI DESIGN
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🧠 Brain Tumor Detection AI
    ### Upload MRI scan for prediction & explanation
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload MRI Scan")
            submit_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column():
            badge_output = gr.HTML(label="Result")
            prediction_text = gr.Textbox(label="Prediction")
            confidence_text = gr.Number(label="Confidence")
            output_image = gr.Image(label="Grad-CAM Explanation")

    gr.Markdown("""
    ## 📘 About
    🔴 Red = important regions  
    🔵 Blue = less important  

    ⚠️ This is for educational purposes only.
    """)

    submit_btn.click(
        fn=predict_and_explain,
        inputs=image_input,
        outputs=[badge_output, prediction_text, confidence_text, output_image],
        show_progress=True   # 🔥 spinner enabled
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
