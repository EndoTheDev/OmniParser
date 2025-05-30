import streamlit as st
import torch
from PIL import Image
import io
import base64
import numpy as np
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

@st.cache_resource
def load_models():
    """Load models once and cache them"""
    yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path="weights/icon_caption_florence"
    )
    return yolo_model, caption_model_processor

def process_image(image, box_threshold, iou_threshold, use_paddleocr, imgsz):
    """Process image with OmniParser"""
    try:
        yolo_model, caption_model_processor = load_models()
        
        box_overlay_ratio = image.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # OCR detection
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # Get labeled image
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image, 
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
        
        # Convert result
        result_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        parsed_content_text = '\n'.join([f'Element {i+1}: {str(v)}' for i, v in enumerate(parsed_content_list)])
        
        return result_image, parsed_content_text, len(parsed_content_list)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, 0

def main():
    st.set_page_config(
        page_title="OmniParser Demo", 
        page_icon="🔍", 
        layout="wide"
    )
    
    # Header
    st.title("🔍 OmniParser for Pure Vision Based General GUI Agent")
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        box_threshold = st.slider(
            "Box Threshold", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.05, 
            step=0.01,
            help="Threshold for removing bounding boxes with low confidence"
        )
        
        iou_threshold = st.slider(
            "IOU Threshold", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.1, 
            step=0.01,
            help="Threshold for removing bounding boxes with large overlap"
        )
        
        use_paddleocr = st.checkbox(
            "Use PaddleOCR", 
            value=True,
            help="Use PaddleOCR for text detection"
        )
        
        imgsz = st.slider(
            "Icon Detect Image Size", 
            min_value=640, 
            max_value=1920, 
            value=640, 
            step=32,
            help="Input image size for icon detection"
        )
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Input")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a screenshot to analyze"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("🚀 Process Image", type="primary"):
                with st.spinner("Processing image... This may take a moment."):
                    result_image, parsed_content, num_elements = process_image(
                        image, box_threshold, iou_threshold, use_paddleocr, imgsz
                    )
                
                if result_image is not None:
                    st.session_state['result_image'] = result_image
                    st.session_state['parsed_content'] = parsed_content
                    st.session_state['num_elements'] = num_elements
                    st.success(f"✅ Successfully detected {num_elements} screen elements!")
    
    with col2:
        st.header("📤 Results")
        
        if 'result_image' in st.session_state:
            st.image(
                st.session_state['result_image'], 
                caption=f"Processed Image ({st.session_state['num_elements']} elements detected)", 
                use_column_width=True
            )
            
            # Download button for processed image
            img_buffer = io.BytesIO()
            st.session_state['result_image'].save(img_buffer, format='PNG')
            st.download_button(
                label="📥 Download Processed Image",
                data=img_buffer.getvalue(),
                file_name="omniparser_result.png",
                mime="image/png"
            )
        else:
            st.info("👆 Upload an image and click 'Process Image' to see results here.")
    
    # Parsed content section
    if 'parsed_content' in st.session_state:
        st.header("📋 Detected Screen Elements")
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Elements", st.session_state['num_elements'])
        with col2:
            text_elements = st.session_state['parsed_content'].count("'type': 'text'")
            st.metric("Text Elements", text_elements)
        with col3:
            icon_elements = st.session_state['num_elements'] - text_elements
            st.metric("Icon Elements", icon_elements)
        
        # Detailed results
        with st.expander("📖 View Detailed Results", expanded=False):
            st.text_area(
                "Parsed Content", 
                st.session_state['parsed_content'], 
                height=400,
                help="Detailed information about each detected element"
            )

if __name__ == "__main__":
    main()
