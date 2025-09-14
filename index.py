import gradio as gr
from backend import recognize_plate

demo = gr.Interface(
    fn=recognize_plate,
    inputs=gr.Image(
        type="numpy",
        sources="upload",
        image_mode="RGB",
        width=800,
        height=500,
        label="Upload Car Image"
    ),
    outputs=[
        gr.Text(label="Detection Result"),
        gr.Image(label="Detected License Plate (Cropped)")
    ],
    title="Car Plate Recognition System",
    description="Upload a car image and get the license plate number along with the cropped detected plate.",
    flagging_mode="never"  # This removes the flag button
)

if __name__ == "__main__":
    demo.launch()