import io
import base64
import os
from PIL import Image
from openai import OpenAI


model_name_map = {
    "GPT-5": "ep-e7dat9-XXXX",
}


class WangQingClient:
    def __init__(self, model_name, timeout=300, max_size=1024, max_images=20):
        self.client = OpenAI(
            api_key="XXXX",
            base_url="http://wanqing.XXXX",
            timeout=timeout
        )

        assert model_name in model_name_map.keys(), f"model_name {model_name} not in {model_name_map.keys()}"

        self.model_name_human = model_name
        self.model_name = model_name_map[model_name]
        self.max_size = max_size
        self.max_images = max_images  

        self.base_url = "NA"

    def resize_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                if img.mode in ('RGBA', 'P', 'LA'):
                    img = img.convert('RGB')
                
                img.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
                
                output_buffer = io.BytesIO()
                img.save(output_buffer, format='JPEG', quality=85, optimize=True)
                
                return output_buffer.getvalue()
        
        except Exception as e:
            with open(image_path, "rb") as f:
                return f.read()

    def encode_image_to_base64(self, image_path):
        compressed_image_data = self.resize_image(image_path)
        return base64.b64encode(compressed_image_data).decode('utf-8')

    def get_image_paths(self, image_dir_path):
        image_paths = []
        
        for image_id in range(1, 999):
            if image_id < 10:
                image_id_str = f"00000{image_id}"
            elif image_id < 100 and image_id >= 10:
                image_id_str = f"0000{image_id}"
            elif image_id < 1000 and image_id >= 100:
                image_id_str = f"000{image_id}"
            else:
                break

            image_path = f'{image_dir_path}/frame_{image_id_str}.jpg'
            if os.path.exists(image_path):
                image_paths.append(image_path)
            else:
                break
        
        if len(image_paths) > self.max_images:
            step = len(image_paths) / self.max_images
            sampled_paths = []
            
            for i in range(self.max_images):
                index = int(i * step)
                sampled_paths.append(image_paths[index])
            
            image_paths = sampled_paths
        else:
            print(f" {len(image_paths)} ")
            
        return image_paths

    def call_openai_vl(self, prompt, image_dir_path=None, saved_image_messages=None, return_image_messages=False):

        assert image_dir_path is not None or saved_image_messages is not None, " image_dir_path  saved_image_messages"
        assert image_dir_path is None or saved_image_messages is None, " image_dir_path  saved_image_messages "

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]
            }
        ]
        
        if saved_image_messages is None:
            image_paths = self.get_image_paths(image_dir_path)
            
            saved_image_messages = []
            for image_path in image_paths:
                saved_image_messages.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image_to_base64(image_path)}"}
                    }
                )
        messages[0]["content"].extend(saved_image_messages)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False
        )
        content = response.choices[0].message.content
        if return_image_messages:
            return content, response, saved_image_messages
        else:
            return content, response

    def call_openai_vl_single(self, prompt, image_path):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]
            }
        ]
        
        saved_image_messages = []
        saved_image_messages.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image_to_base64(image_path)}"}
            }
        )
        messages[0]["content"].extend(saved_image_messages)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False
        )
        content = response.choices[0].message.content

        return content, response

    def call_openai(self, prompt):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False
        )
        content = response.choices[0].message.content

        return content, response