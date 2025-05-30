import gradio as gr
import numpy as np
import torch
from PIL import Image
import io
import base64, os
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")

DEVICE = torch.device('cuda')

def process_image(image_input, box_threshold, iou_threshold, use_paddleocr, imgsz):
    if image_input is None:
        return None, "Please upload an image first."
    
    try:
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_input, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_input, 
            yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=caption_model_processor, 
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz,
        )  
        
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        parsed_content_text = '\n'.join([f'icon {i}: ' + str(v) for i, v in enumerate(parsed_content_list)])
        
        return image, parsed_content_text
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Create interface with explicit types
with gr.Blocks(title="OmniParser Demo") as demo:
    gr.Markdown("# OmniParser for Pure Vision Based General GUI Agent")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type='pil', label='Upload image')
            box_threshold = gr.Slider(
                minimum=0.01, maximum=1.0, step=0.01, value=0.05,
                label='Box Threshold'
            )
            iou_threshold = gr.Slider(
                minimum=0.01, maximum=1.0, step=0.01, value=0.1,
                label='IOU Threshold'
            )
            use_paddleocr = gr.Checkbox(label='Use PaddleOCR', value=True)
            imgsz = gr.Slider(
                minimum=640, maximum=1920, step=32, value=640,
                label='Icon Detect Image Size'
            )
            submit_btn = gr.Button('Process Image', variant='primary')
        
        with gr.Column():
            image_output = gr.Image(type='pil', label='Processed Image')
            text_output = gr.Textbox(
                label='Parsed Screen Elements', 
                placeholder='Results will appear here...',
                lines=10
            )

    submit_btn.click(
        fn=process_image,
        inputs=[image_input, box_threshold, iou_threshold, use_paddleocr, imgsz],
        outputs=[image_output, text_output]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_port=7862, server_name='0.0.0.0')
