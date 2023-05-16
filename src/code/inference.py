import base64
import torch
import requests
from PIL import Image
from io import BytesIO
import boto3
from datetime import datetime
import uuid
from diffusers import StableDiffusionInpaintPipeline

s3 = boto3.client('s3')

saving_bucket = "cloudbeer-llm-models"
key_prefix = "works_result/"
cloudfront_url = "https://d1ffqcflvp9rc.cloudfront.net/"



def model_fn(model_dir):
    # print(model_dir)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker = None
    )
    pipe.to("cuda")
    return pipe

def s3_to_cf_url(s3_url):
    o = split_s3_path(s3_url)
    return cloudfront_url + o[1]


def split_s3_path(s3_path):
    path_parts=s3_path.replace("s3://","").split("/")
    bucket=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket, key

def download_image(url, w, h):
    o = split_s3_path(url)
    response = s3.get_object(Bucket=o[0], Key=o[1])['Body'].read()
    res_img = Image.open(BytesIO(response)).convert("RGB")
    res_img = res_img.resize((w, h))
    return res_img

def download_image_http(url, w, h):
    response = requests.get(url)
    res_img = Image.open(BytesIO(response.content)).convert("RGB")
    res_img = res_img.resize((w, h))
    return res_img


def gen(data, pipe):
    prompt = data.pop("prompt", data)
    image_url = data.pop("image_url", data)
    mask_url = data.pop("mask_url", None)
    width = data.pop("width", 384)
    height = data.pop("height", 512)
    
    image_ori = download_image(image_url, width, height)
    if mask_url: 
        mask_image = download_image(mask_url, width, height)
    else:
        mask_url = image_url
        mask_image = image_ori

    num_inference_steps = data.pop("num_inference_steps", 30)
    guidance_scale = data.pop("guidance_scale", 7.5)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)
    prompt_suffix = ",fine skin,masterpiece,cinematic light, ultra high res, film grain, perfect anatomy, best shadow, delicate,(photorealistic:1.4),(extremely intricate:1.2)"
    nprompt = 'bad_legs,bad_fingers,(semi_realistic,cgi,3d,render,sketch,cartoon,drawing,anime:1.4),text,cropped,out_of_frame,worst_quality,low_quality,jpeg_artifacts,ugly,duplicate,morbid,mutilated,extra_fingers,mutated_hands,poorly_drawn_hands,poorly_drawn_face,mutation,deformed,blurry,dehydrated,bad_anatomy,bad_proportions,extra_limbs,cloned_face,disfigured,gross_proportions,malformed_limbs,missing_arms,missing_legs,extra_arms,extra_legs,fused_fingers,too_many_fingers,long_neck,signature'


    now = datetime.now() 
    date_str = now.strftime("%Y-%m-%d")

    html = "<html><head><title>图片生成" + date_str + "</title><link href='../main.css' rel='stylesheet'></head><body>"
    html += "<h1>图片生成" + date_str + "</h1>"
    html += "<h4>提示词: " + prompt + prompt_suffix + "</h4>"
    
    cf_in_url = s3_to_cf_url(image_url)
    cf_msk_url = s3_to_cf_url(mask_url)
    html += "<div><a href='" + cf_in_url + "' target='_blank'><img src='" + cf_in_url + "' /></a>"
    html += "<a href='" + cf_msk_url + "' target='_blank'><img src='" + cf_msk_url + "' /></a></div>"


    for i in range(num_images_per_prompt):
        generated_images = pipe(
            prompt=prompt + prompt_suffix,
            negative_prompt=nprompt,
            image=image_ori, 
            mask_image=mask_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
        )["images"]

        for image in generated_images:
            file_name = str(uuid.uuid4()) + ".jpg"
            key = key_prefix + date_str + '/' + file_name
            in_mem_file = BytesIO()
            image.save(in_mem_file, format="JPEG")
            in_mem_file.seek(0)
            s3.upload_fileobj(
                in_mem_file, 
                saving_bucket, 
                key,
                ExtraArgs={
                    'ContentType': 'image/jpeg'
                }
            )
            html += "<a href='" + file_name + "' target='_blank'><img src='" + file_name + "' /></a>"
    html += "</body></html>"

    index_file_name = 'index-' + str(uuid.uuid4()) + '.html'
    s3.put_object(
        Bucket='cloudbeer-llm-models',
        Key=key_prefix + date_str + '/' + index_file_name,
        Body=html.encode('utf-8'),
        ContentType='text/html'
    )

    return cloudfront_url + key_prefix + date_str + '/' + index_file_name

def predict_fn(data, pipe):
    index_url = gen(data, pipe)
    return index_url
    # generated_images = gen(data, pipe)
    # encoded_images = []
    # for image in generated_images:
    #     buffered = BytesIO()
    #     image.save(buffered, format="JPEG")
    #     encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    # return {"generated_images": encoded_images}