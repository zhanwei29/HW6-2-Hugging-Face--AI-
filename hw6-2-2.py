from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os
import re  # 用於處理非法文件名字符


# 第一步：使用 Hugging Face 翻譯管線翻譯句子
def translate_text(input_text, source_lang="zh", target_lang="en"):
    """
    使用 Hugging Face 翻譯模型翻譯文本
    """
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
    result = translator(input_text)
    translated_text = result[0]['translation_text']
    print(f"Translated Text: {translated_text}")
    return translated_text


# 第二步：使用 Stable Diffusion 模型生成圖片
def generate_image(prompt):
    """
    使用 Hugging Face 文本生成圖像模型
    """
    # 加載 Stable Diffusion 模型
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")  # 強制使用 CPU

    # 生成圖片
    image = pipe(prompt).images[0]
    return image


# 處理文件名，移除非法字符
def sanitize_filename(filename):
    """
    移除文件名中的非法字符
    """
    return re.sub(r'[\\/*?:"<>|]', "", filename)


# 第三步：主函數
def main():
    input_text = input("請輸入中文描述（例如：一個狗和一隻貓）: ")
    print(f"Input Text: {input_text}")

    # 翻譯句子
    translated_text = translate_text(input_text)

    # 處理文件名
    sanitized_filename = sanitize_filename(translated_text)

    # 使用翻譯結果生成圖像
    print(f"Generating image for: {translated_text}")
    image = generate_image(translated_text)

    # 確定保存路徑
    save_directory = r"D:\hw6-2\generated_picture"  # 確保目錄為 D:\hw6-2
    os.makedirs(save_directory, exist_ok=True)  # 如果目錄不存在，則創建
    output_path = os.path.join(save_directory, f"{sanitized_filename}.png")

    # 保存生成的圖像
    image.save(output_path)
    print(f"Image saved at: {output_path}")

    # 顯示生成的圖像
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated Image: {translated_text}")
    plt.show()


# 啟動程式
if __name__ == "__main__":
    main()
