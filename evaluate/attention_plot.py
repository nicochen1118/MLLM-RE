import os
import argparse
import json

import torch

from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import base64
import io
import re
import random

from scipy.ndimage.filters import gaussian_filter
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from utils import (
    aggregate_llm_attention, aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image
)

# from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus,AblationCAM, \
#                             XGradCAM, EigenCAM, EigenGradCAM,LayerCAM,FullGrad
# from pytorch_grad_cam import GuidedBackpropReLUModel
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

prompt = '''
Head: {}
Tail: {}

The known relationships are [race, place_of_birth, present_in, alternate_names, parent, locate_at, neighbor, part_of, alumni, siblings, religion, place_of_residence, nationality, contain, couple, awarded, subsidiary, peer, held_on, member_of, charges, None]. 

Given the input image, 'head', and 'tail', identify the appropriate relationship between 'head' and 'tail' from the above list. The 'head' and 'tail' are provided as input entities. 

Select the relationship that best describes the connection between 'head' and 'tail' based on the provided image and output the result in JSON format with 'relation', 'head', and 'tail'. 

For example:
Output: ['relation': '', 'head': '', 'tail': '']
 '''

info_dict = {
    "race": 1,
    "place_of_birth": 2,
    "present_in": 3,
    "alternate_names": 4,
    "parent": 5,
    "locate_at": 6,
    "neighbor": 7,
    "part_of": 8,
    "alumni": 9,
    "siblings": 10,
    "religion": 11,
    "place_of_residence": 12,
    "nationality": 13,
    "contain": 14,
    "couple": 15,
    "awarded": 16,
    "subsidiary": 17,
    "peer": 18,
    "held_on": 19,
    "member_of": 20,
    "charges": 21,
    "None": 22  # 特殊值None，没有对应数字
}
def get_answer(outputs):
    outputs = outputs.replace('\\', '')

    for key in info_dict:
        if re.search(key, outputs, re.IGNORECASE):
            return str(info_dict[key])  # 返回匹配键对应的值
    return "22"
class Aligngpt:
    def __init__(self, model_path) -> None:
        import sys
        sys.path.append('/home/nfs02/xingsy/code/AlignGPT')
        from AlignGPT.src.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, INFERENCE
        from AlignGPT.src.utils.conversation import conv_templates, SeparatorStyle
        from AlignGPT.src.model.builder import load_pretrained_model
        from AlignGPT.src.utils.general_utils import disable_torch_init
        from AlignGPT.src.utils.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, "aligngpt-7b",device="cuda")
        self.model.set_stage(INFERENCE)


    def get_save_path(self, save_path, suffix='_v'):
        if os.path.exists(save_path):
            filename, ext = os.path.splitext(save_path)
            version = 1
            new_save_path = f"{filename}{suffix}{version}{ext}"
            while os.path.exists(new_save_path):
                version += 1
                new_save_path = f"{filename}{suffix}{version}{ext}"
            return new_save_path
        return save_path


    def chat(self, input):
        from AlignGPT.src.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, INFERENCE
        from AlignGPT.src.utils.conversation import conv_templates, SeparatorStyle
        from AlignGPT.src.model.builder import load_pretrained_model
        from AlignGPT.src.utils.general_utils import disable_torch_init
        from AlignGPT.src.utils.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        msgs = input['question']
        # TODO: check version

        conv_mode = "llava_v0"


        conv = conv_templates[conv_mode].copy()

        roles = conv.roles

        image = Image.open(input['image']).convert('RGB')

        # Similar operation in model_worker.py
        from AlignGPT.src.utils.mm_utils import process_images
        image_tensor, images = process_images([image], self.image_processor, self.model.config)
        image = images[0]
        image_size = image.size
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        inp = msgs
        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        from transformers import TextStreamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.model.set_stage('inference')
        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False ,
                temperature=0.0,
                max_new_tokens=200,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=True
               )

        aggregated_prompt_attention = []
        for i, layer in enumerate(output["attentions"][0]):
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            cur = attns_per_head[:-1].cpu().clone()
            # following the practice in `aggregate_llm_attention`
            # we are zeroing out the attention to the first <bos> token
            # for the first row `cur[0]` (corresponding to the next token after <bos>), however,
            # we don't do this because <bos> is the only token that it can attend to
            cur[1:, 0] = 0.
            cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
            aggregated_prompt_attention.append(cur)
        aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

        # llm_attn_matrix will be of torch.Size([N, N])
        # where N is the total number of input (both image and text ones) + output tokens
        llm_attn_matrix = heterogenous_stack(
            [torch.tensor([1])]
            + list(aggregated_prompt_attention) 
            + list(map(aggregate_llm_attention, output["attentions"]))
        )

        # visualize the llm attention matrix
        # ===> adjust the gamma factor to enhance the visualization
        #      higer gamma brings out more low attention values
        gamma_factor = 1
        enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)

        fig, ax = plt.subplots(figsize=(10, 20), dpi=150)
        ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")

        
        # identify length or index of tokens

        input_token_len = self.model.get_vision_tower().num_patches + len(input_ids[0]) - 1 # -1 for the <image> token
        vision_token_start = len(self.tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
        vision_token_end = vision_token_start + self.model.get_vision_tower().num_patches
        output_token_len = len(output["sequences"][0])
        # print(output_token_len)
        output_token_start = input_token_len

        output_token_end = input_token_len + output_token_len

        # connect with the vision encoder attention
        # to visualize the attention over the image

        # vis_attn_matrix will be of torch.Size([N, N])
        # where N is the number of vision tokens/patches
        # `all_prev_layers=True` will average attention from all layers until the selected layer
        # otherwise only the selected layer's attention will be used
        vis_attn_matrix = aggregate_vit_attention(
            self.model.get_vision_tower().image_attentions,
            select_layer=self.model.get_vision_tower().select_layer,
            all_prev_layers=True
        )
        grid_size = self.model.get_vision_tower().num_patches_per_side

        num_image_per_row = 8
        
        image_ratio = image_size[0] / image_size[1]

        num_rows = (output_token_len - input_ids.shape[1]) // num_image_per_row + (1 if (output_token_len - input_ids.shape[1])  % num_image_per_row != 0 else 0)
        
        fig, axes = plt.subplots(
            num_rows, num_image_per_row, 
            figsize=(10, (10 / num_image_per_row) * image_ratio * num_rows), 
            dpi=150
        )
        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        # whether visualize the attention heatmap or 
        # the image with the attention heatmap overlayed
        vis_overlayed_with_attn = True

        output_token_inds = list(range(output_token_start, output_token_end))
        for i, ax in enumerate(axes.flatten()):
            if i + input_ids.shape[1] >= output_token_len:
                break
            
            target_token_ind = input_ids.shape[1] + i
            attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
            attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

            attn_over_image = []
            for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
                vis_attn = vis_attn.reshape(grid_size, grid_size)
                # vis_attn = vis_attn / vis_attn.max()
                attn_over_image.append(vis_attn * weight)
            attn_over_image = torch.stack(attn_over_image).sum(dim=0)
            attn_over_image = attn_over_image / attn_over_image.max()

            attn_over_image = F.interpolate(
                attn_over_image.unsqueeze(0).unsqueeze(0), 
                size=image_size, 
                mode='nearest', 
                # mode='bicubic', align_corners=False
            ).squeeze()

            np_img = np.array(image)[:, :, ::-1]
            img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())
            ax.imshow(heatmap if not vis_overlayed_with_attn else img_with_attn)
            ax.set_title(
                self.tokenizer.decode(output["sequences"][0, input_ids.shape[1] + i : input_ids.shape[1] + i + 1]).strip(),
                fontsize=7,
                pad=1
            )
    # 示例使用
            ax.axis("off")
        
            
        base_path = '/home/nfs02/xingsy/code/new_attention/test/'
        img_id = input['data'] 
        temp = base_path + img_id

        save_path = self.get_save_path(temp)
        plt.savefig(save_path, dpi=200)
        outputs = self.tokenizer.decode(output["sequences"][0, input_ids.shape[1]:]).strip()
        print(outputs)
        return get_answer(outputs)
        # for i, attention in enumerate(output["attentions"][0]):
        #     print(f"Attention layer {i} shape: {attention.shape}")
        # attention = torch.mean(output.attentions[-1].squeeze(0), dim=0)
        # attention = attention[358:, :]
        # attention_image = attention[:, 98:354]
        # logits = output.logits[:, 358:, :]
        # for i in range(len(attention)):
        #     logit = logits[:, i, :]
        #     token_id = torch.argmax(logit, dim=1).item()
        #     token = self.tokenizer.decode([token_id], skip_special_tokens=True)

        #     attention_hallu = attention[i, 98:354]
        #     attention_hallu = torch.softmax(attention_hallu*200, dim=0).view(16, 16)
        #     attention_hallu = np.array(attention_hallu.cpu(), dtype=np.float32)*100

        #     img = Image.open(input['image'])
        #     resized_attention = np.array(Image.fromarray((attention_hallu*255).astype(np.uint8)).resize(img.size, resample=Image.BILINEAR))
        #     smoothed_attention = gaussian_filter(resized_attention, sigma=2)
        #     heatmap = np.uint8(smoothed_attention)
        #     heatmap = plt.cm.jet(heatmap)[:, :, :3] * 255
        #     heatmap = Image.fromarray(np.uint8(heatmap)).resize(img.size)
        #     result = Image.blend(img.convert('RGBA'), heatmap.convert('RGBA'), alpha=0.5)
        #     result.convert('RGB').save('/home/nfs02/xingsy/code/temp.jpg')

        # attention_image = torch.mean(attention_image, dim=1).unsqueeze(1)
        # attention = torch.cat([attention_image, attention[:, 359:]], dim=1).T
        # l = len(attention)
        # y = ["<Image>"]
        # x = []
        # for i in range(l-1):
        #     logits = output.logits[:, 358+i, :]
        #     token_id = torch.argmax(logits, dim=1).item()
        #     token = self.tokenizer.decode([token_id], skip_special_tokens=True)
        #     y.append(token)
        #     x.append(token)
        # y = y[:-1]
        # attention = attention.tolist()
        # for i in range(len(attention[0])):
        #     attention[0][i] *= 1
        # sns.heatmap(attention, cmap="Blues", xticklabels=x, yticklabels=y)
        # plt.gcf().set_size_inches(10, 8)
        # plt.savefig('/home/nfs02/xingsy/code/temp1.jpg', dpi=200)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='/home/nfs02/xingsy/code/output/temp.txt')
    parser.add_argument('--file_path', default='/home/nfs02/xingsy/code/data/MNRE/mnre_txt/mnre_test.txt')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of GPU to use')
    parser.add_argument('--random_flag', type=int, default=0, help='Index of GPU to use')
    args = parser.parse_args()

    random_flag = 0
    device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    
    chat_model = Aligngpt('/home/nfs02/xingsy/code/AlignGPT/playground/model/aligngpt-7b')
    output_file = args.output_path
    file_path = args.file_path
    count = 0
    with open(output_file, "a+") as f:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Processing", unit=" lines"):  # 使用 tqdm 包装文件迭代
                # 解析每一行的JSON数据
                data = eval(line.strip())
                # 处理文本数据
                sentence = ' '.join(data['token'])
                head = data['h']['name']
                tail = data['t']['name']

                text = prompt.format(sentence, head, tail)
                # 处理图像数据（示例中仅打开图像，实际应用中可能需要进行预处理）

                image_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_image/img_org/test/' + data['img_id']
                inputs = {"image": image_path, "question": text, "data": data['img_id']}
                answer = chat_model.chat(inputs)
                outputs = answer.replace("\n", "")
                result = " ### ".join([data['img_id'], outputs])
                print(result)
                # f.write(result + "\n")
                # image_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_image/img_org/mask_test/' + data['img_id']

                # if not os.path.exists(image_path):
                #     count += 1
                #     print(image_path)
                #     # 如果图像文件不存在，使用备用路径
                #     image_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_image/img_org/test/' + data['img_id']
                # inputs = {"image": image_path, "question": text, "data": data['img_id']}
                # answer = chat_model.chat(inputs)
                # outputs = answer.replace("\n", "")
                # result = " ### ".join([data['img_id'], outputs])
                # print("mask",result)
                # f.write(result + "\n")
