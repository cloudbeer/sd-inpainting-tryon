cd ./src

rm stable-diffusion-inpainting-tryon.tar.gz
tar zcvf stable-diffusion-inpainting-tryon.tar.gz *

aws s3 cp stable-diffusion-inpainting-tryon.tar.gz \
  s3://cloudbeer-llm-models/diffusers/stable-diffusion-inpainting-tryon.tar.gz