#!/usr/bin/env python3
"""
Simple test script to verify OmniParser functionality without Gradio
"""

import torch
from PIL import Image
import io
import base64
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

def test_omniparser():
    print("🔄 Loading models...")
    
    # Load models
    yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path="weights/icon_caption_florence"
    )
    
    print("✅ Models loaded successfully!")
    
    # Test with a demo image
    try:
        # Load demo image
        demo_image_path = "imgs/demo_image.jpg"
        image = Image.open(demo_image_path)
        print(f"📸 Loaded test image: {demo_image_path}")
        
        # Set parameters
        box_threshold = 0.05
        iou_threshold = 0.1
        use_paddleocr = True
        imgsz = 640
        
        print("🔄 Processing image...")
        
        # Process with OmniParser
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
        print(f"📝 OCR detected {len(text)} text elements")
        
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
        
        # Save result
        result_image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        result_image.save("test_output.png")
        
        parsed_content_text = '\n'.join([f'Element {i}: {str(v)}' for i, v in enumerate(parsed_content_list)])
        
        print("✅ Processing completed successfully!")
        print(f"📊 Detected {len(parsed_content_list)} screen elements")
        print(f"💾 Result saved as 'test_output.png'")
        
        # Print first few results
        print("\n📋 Sample results:")
        for i, element in enumerate(parsed_content_list[:5]):
            print(f"  {i+1}. {element}")
        
        if len(parsed_content_list) > 5:
            print(f"  ... and {len(parsed_content_list) - 5} more elements")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing OmniParser functionality...")
    success = test_omniparser()
    
    if success:
        print("\n🎉 SUCCESS: OmniParser is working correctly!")
        print("📝 The Gradio interface has a bug, but the core functionality works.")
        print("💡 You can use the core functions directly or try alternative UI frameworks.")
    else:
        print("\n💥 FAILED: There are issues with the core functionality.")
