#!pip install diffusers transformers accelerate torch torchvision  if you are running this in Google Colab
# Import necessary libraries


from diffusers import StableDiffusionPipeline
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch
import os


if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU not available! Please enable GPU in Colab: Runtime > Change runtime type > GPU")

device = torch.device("cuda")
print(f"✅ Using device: {device}")


huggingface_token = "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"


def generate_scene(prompt, output_path, hf_token):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        use_auth_token=hf_token,
        torch_dtype=torch.float16
    ).to(device)

    image = pipe(prompt).images[0]
    image.save(output_path)
    print(f"✅ Scene generated and saved to {output_path}")


def detect_objects(image_path, output_annotated_path="annotated_scene.png"):
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = model.config.id2label[label.item()]
        box = [round(i, 2) for i in box.tolist()]
        detections.append((label_name, score.item(), box))


        x0, y0, x1, y1 = box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, y0), f"{label_name} {score:.2f}", fill="white", font=font)

    image.save(output_annotated_path)
    print(f"🖼️ Annotated image saved to {output_annotated_path}")
    return detections


def decide_navigation(detections):
    obstacles = ["car", "truck", "bus", "person", "tree", "building", "traffic light", "stop sign"]
    action_taken = False

    for label, score, box in detections:
        if label.lower() in obstacles:
            print(f"🚧 Obstacle detected: {label} at {box} (confidence {round(score, 2)})")
            action_taken = True

    if action_taken:
        print("🔀 Decision: Obstacle detected → Turn left or right to avoid.")
    else:
        print("✅ Path is clear → Go forward.")


prompt = "a top-down view of 3 cars driving on a road with buildings and trees around"
image_file = "scene.png"

generate_scene(prompt, image_file, huggingface_token)
detections = detect_objects(image_file, "annotated_scene.png")
decide_navigation(detections)

from IPython.display import Image as IPImage
IPImage("annotated_scene.png")