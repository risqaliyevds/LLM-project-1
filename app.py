import gradio as gr
from gradio.components import Audio
from model import STT_with_Summary

output_1 = gr.Textbox(label="STT:")
output_2 = gr.Textbox(label="Suhbatning qisqacha mazmuni:")

app = gr.Interface(
    title="Audio xabar va uning qisqacha mazmuni.",
    fn=STT_with_Summary,
    inputs=[Audio(source="upload", type="filepath")],
    outputs=[output_1, output_2],
    live=False
)

app.launch(share=True)

